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

#include "Paracrypt.hpp"
#include "Launcher.hpp"
#include "BlockCipher.hpp"
#include "CudaAesVersions.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

void hexdump(std::string title, const unsigned char *s, int length)
{
  std::stringstream stream;
  for (int n = 0; n < length; ++n) {
    if ((n % 16) == 0)
      stream << "\ntitle " << (int)n;
    stream << std::hex << std::setw(4) << (int)s[n];
  }
  stream << std::endl;
  std::cerr << stream.str() << std::endl;
}

void fdump(std::string title, std::string filename)
{
  std::ifstream f(filename.c_str(),std::ifstream::binary);
  if(f.is_open()) {
    std::stringstream stream;
    std::streampos n = 0;
    std::streampos nInc = 1;
    while(!f.fail() || !f.eof()) {
      char buff[16];
      f.read(buff,16);
      unsigned int readed = 16;
      if(f.fail() && f.eof()) {
	readed = f.gcount();
      }
      stream << "\ntitle " << (int)n;
      for(unsigned int i = 0; i < readed; i++) {
	stream << std::hex << std::setw(4) <<  (int)buff[i];
	n = n + nInc;
      }
    }
    stream << std::endl;
    std::cerr << stream.str() << std::endl;
  }
}

void paracrypt::exec(paracrypt::config_t c) {

	// convert API public types to internal types
	paracrypt::Launcher::operation_t op = (paracrypt::Launcher::operation_t) c.op;
	paracrypt::BlockCipher::Mode m = (paracrypt::BlockCipher::Mode) c.m;

	hexdump("key",c.key,c.key_bits/8);
	if(c.ivBits != 0)
		hexdump("iv",c.iv,c.ivBits/8);

	if(c.stagingLimit != 0) {
		paracrypt::Launcher::limitStagging(c.stagingLimit);
	}

	if(c.kernelParalellismLimit != -1) {
		paracrypt::CUDACipherDevice::limitConcurrentKernels(c.kernelParalellismLimit);
	}

#define LAUNCH_SHARED_IO_CUDA_AES(implementation) \
		paracrypt::Launcher::launchSharedIOCudaAES<implementation>( \
		   		op, \
		   		c.inFile, c.outFile, \
		   		c.key, c.key_bits, \
		   		c.constantKey, c.constantTables, \
		   		m, c.iv, c.ivBits, \
		   		c.outOfOrder, \
		   		c.begin, c.end \
		);

	// Only CUDA AES is supported in this version
	switch(c.c)	{
		case paracrypt::AES16B:
		  LAUNCH_SHARED_IO_CUDA_AES(paracrypt::CudaAES16B);
		  break;
	}
}
