/*
 *  Copyright (C) 2017 Jesus Martin Berlanga. All Rights Reserved.
 *
 *  This file is part of Paracrypt.
 *
 *  Paracrypt is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Paracrypt is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Paracrypt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include "SharedIO.hpp"
#include <boost/thread/locks.hpp>
#ifdef DEBUG
	#include "nvToolsExt.h"
#endif

// Reader thread
void paracrypt::SharedIO::reading() {
	chunk c;
	c.status = OK;
	while(c.status != END && !(*finishThreading)) {
		boost::unique_lock<boost::mutex> lock(chunk_access);
			if(this->emptyChunks->size() <= 0) {
				this->thereAreEmptyChunks.wait(lock);
				if(*finishThreading)
					break;
			}
			c = this->emptyChunks->dequeue();
		lock.unlock();

#ifdef DEBUG
	    nvtxRangeId_t id1 = nvtxRangeStartA("Chunk read");
#endif
		c.nBlocks = this->inFileRead(c.data,this->getChunkSize(),&c.status,&c.blockOffset,&c.padding);
#ifdef DEBUG
	    nvtxRangeEnd(id1);
#endif


		lock.lock();
			readyToReadChunks->enqueue(c);
			if(this->readyToReadChunks->size() == 1) {
				this->thereAreChunksToRead.notify_one();
			}
		lock.unlock();
	}
}

// Main thread
paracrypt::BlockIO::chunk paracrypt::SharedIO::read()
{
	boost::unique_lock<boost::mutex> lock(chunk_access);
		if(this->readyToReadChunks->size() <= 0) {
			this->thereAreChunksToRead.wait(lock);
		}
		chunk c = this->readyToReadChunks->dequeue();
		return c;
}
void paracrypt::SharedIO::dump(chunk c)
{
	boost::unique_lock<boost::mutex> lock(chunk_access);
		outputChunks->enqueue(c);
		if(this->outputChunks->size() == 1) {
			this->thereAreChunksToWrite.notify_one();
		}
}

// Writter thread
void paracrypt::SharedIO::writing() {
	chunk c;
	c.status = OK;
	while(!(*finishThreading) || this->outputChunks->size() > 0) {
		boost::unique_lock<boost::mutex> lock(chunk_access);
			if(this->outputChunks->size() <= 0) {
				if(*finishThreading) {
					break;
				}
				this->thereAreChunksToWrite.wait(lock);
				if(this->outputChunks->size() == 0 && *finishThreading) {
					break;
				}
			}
			c = this->outputChunks->dequeue();
		lock.unlock();

#ifdef DEBUG
	    nvtxRangeId_t id1 = nvtxRangeStartA("Chunk write");
#endif
		this->outFileWrite(c.data,c.nBlocks,c.blockOffset,c.padding);
#ifdef DEBUG
	    nvtxRangeEnd(id1);
#endif

		lock.lock();
			emptyChunks->enqueue(c);
			if(this->emptyChunks->size() == 1) {
				this->thereAreEmptyChunks.notify_one();
			}
		lock.unlock();
	}
}

paracrypt::SharedIO::SharedIO(
		std::string inFilename,
		std::string outFilename,
		unsigned int blockSize,
		std::streampos begin,
		std::streampos end
		)
: paracrypt::BlockIO::BlockIO(inFilename, outFilename, blockSize, begin, end)
{
//	this->finishThreading = NULL;
}

paracrypt::SharedIO::~SharedIO() {}

void paracrypt::SharedIO::construct(unsigned int nChunks, rlim_t bufferSizeLimit) {
    // allocate buffer chunks
	// TODO when the program runs not all the chunks will be
	//  used concurrently and we will waste both memory and
	//  time at cudaHostMalloc(). For this reason we stablish
	//  a hard-limit, a maximum number of chunks.
	this->bufferSize = nChunks;
    rlim_t buffersTotalSize = this->getPinned()->getReasonablyBigChunkOfRam(bufferSizeLimit);

    if(this->getEnd() != NO_RANDOM_ACCESS) {
    	rlim_t maxRead = this->getMaxBlocksRead()*this->getBlockSize();
    	// do not allocate memory we will not use
    	buffersTotalSize = std::min(buffersTotalSize, maxRead);
    }

    // Align to block/nchunks size in excess
    //  in this way we assure that we reserve
    //  enough memory to process an entire file
    //  delimited by bufferSizeLimit at once.
    {
		rlim_t remaining = buffersTotalSize % this->getBlockSize();
		buffersTotalSize += remaining;
		remaining = (buffersTotalSize/this->getBlockSize()) % nChunks;
		buffersTotalSize += remaining*nChunks;
    }

    // at least one block per chunk
    buffersTotalSize = std::max(buffersTotalSize, ((rlim_t)this->getBlockSize())*nChunks);

    bool allocSuccess = false;
    std::streamsize bufferSizeBytes;
    do {
		if(buffersTotalSize < this->getBlockSize()*nChunks) {
			// exit with error
		  std::cout << "Couldn't allocate SharedIO internal buffer.\n"
			    << std::endl;
		}
    	this->chunkSize = (buffersTotalSize / this->getBlockSize() / nChunks);
    	// bufferSize aligned to chunk size
        bufferSizeBytes = this->getBufferSize()*this->getChunkSize()*this->getBlockSize();
    	allocSuccess = this->getPinned()->alloc((void**)&this->chunksData,bufferSizeBytes);
    	if(!allocSuccess) {
    		buffersTotalSize -= this->chunkSize;
    		// minimum 1 block per chunk
    		if(buffersTotalSize >= this->getBlockSize()*nChunks) {
    		}
    	}
    }
    while(!allocSuccess);

    // initialize chunks
    this->chunks = new chunk[nChunks];
    this->emptyChunks = new LimitedQueue<chunk>(this->getBufferSize());
    for(unsigned int i = 0; i < nChunks; i++) {
    	unsigned char* chunkData = this->chunksData + this->getChunkSize()*this->getBlockSize()*i;
    	this->chunks[i].data = chunkData;
    	this->emptyChunks->enqueue(this->chunks[i]);
    }


    // launch reader and writer threads
    this->readyToReadChunks = new LimitedQueue<chunk>(this->getBufferSize());
    this->outputChunks = new LimitedQueue<chunk>(this->getBufferSize());
    this->finishThreading = new bool();
    *finishThreading = false;
    this->reader = new boost::thread(boost::bind(&paracrypt::SharedIO::reading, this));
    this->writer = new boost::thread(boost::bind(&paracrypt::SharedIO::writing, this));
}
void paracrypt::SharedIO::destruct() {

	boost::unique_lock<boost::mutex> lock(chunk_access);
		*finishThreading = true;
		this->thereAreEmptyChunks.notify_all();
		this->thereAreChunksToWrite.notify_all();
	lock.unlock();
	this->reader->join();
	this->writer->join();

	delete this->reader;
	delete this->writer;
	delete this->finishThreading;
	delete this->emptyChunks;
	delete this->readyToReadChunks;
	delete this->outputChunks;
	delete[] this->chunks;
	this->getPinned()->free((void*)this->chunksData);
}

const std::streamsize paracrypt::SharedIO::getBufferSize() {
	return this->bufferSize;
}

const std::streamsize paracrypt::SharedIO::getChunkSize() {
	return this->chunkSize;
}
