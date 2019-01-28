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

#include "GPUCipherDevice.hpp"
#include <math.h> 

template < typename S, typename F >
int paracrypt::GPUCipherDevice < S, F >::getNWarpsPerBlock()
{
    return this->nWarpsPerBlock;
}

template < typename S, typename F >
int paracrypt::GPUCipherDevice < S, F >::getThreadsPerThreadBlock()
{
    return this->nThreadsPerThreadBlock;
}

template < typename S, typename F >
void paracrypt::GPUCipherDevice < S, F >::setThreadsPerThreadBlock(int tptb)
{
    this->nThreadsPerThreadBlock = tptb;
}

template < typename S, typename F >
int paracrypt::GPUCipherDevice < S, F >::getMaxBlocksPerSM()
{
    return this->maxBlocksPerSM;
}

template < typename S, typename F >
int paracrypt::GPUCipherDevice < S, F >::getConcurrentKernels()
{
	if(this->maxConcurrentKernels == -1)
		return this->nConcurrentKernels;
	else 
		return std::min(this->maxConcurrentKernels,this->nConcurrentKernels);
}

template < typename S, typename F >
    paracrypt::GPUCipherDevice < S, F >::~GPUCipherDevice()
{
	typename boost::unordered_map<int,S>::iterator iter;
    for(iter = this->streams.begin(); iter != this->streams.end(); ++iter)
    {
          delStream(iter->first);
    }
}

template < typename S, typename F >
    int paracrypt::GPUCipherDevice < S, F >::getGridSize(int n_blocks,
							 int
							 threadsPerCipherBlock)
{
	int tptb = this->getThreadsPerThreadBlock();
	float cipherBlocksPerThreadBlock = tptb / threadsPerCipherBlock;
	if(std::fmod(cipherBlocksPerThreadBlock,1) != 0) {
		int newTptb = threadsPerCipherBlock;
		int newTptbAux = newTptb;
		do {
			newTptb = newTptbAux;
			newTptbAux *= 2;
		}while(newTptbAux < tptb);
		this->setThreadsPerThreadBlock(newTptb);
	}
    float fGridSize =
	n_blocks * threadsPerCipherBlock /
	(float) tptb;
    int gridSize = ceil(fGridSize);
    return gridSize;
}

template < typename S, typename F >
    int paracrypt::GPUCipherDevice < S, F >::addStream()
{
    int id = this->streams.size();
    this->streams[id] = newStream();
    return id;
}

template < typename S, typename F >
    void paracrypt::GPUCipherDevice < S, F >::delStream(int stream_id)
{
    freeStream(this->streams[stream_id]);
    this->streams.erase(stream_id);
}

template < typename S, typename F >
    S paracrypt::GPUCipherDevice < S, F >::acessStream(int stream_id)
{
    return this->streams[stream_id];
}

template < typename S, typename F >
 int paracrypt::GPUCipherDevice < S, F >::maxConcurrentKernels = -1;

template < typename S, typename F >
    void paracrypt::GPUCipherDevice < S, F >::limitConcurrentKernels(int limit)
{
    maxConcurrentKernels = limit;
}
