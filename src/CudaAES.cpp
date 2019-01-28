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

#include "CudaAES.hpp"
#include "CUDACipherDevice.hpp"
#include "CUDABlockCipher.hpp"
#include "BlockCipher.hpp"
#include "CudaConstant.cuh"
#include "CudaPinned.hpp"
#include "CudaAESTTables.cpp"

/* Just using the GCM Counter Mode */
paracrypt::CudaAES::CudaAES()
  : paracrypt::CUDABlockCipher::CUDABlockCipher()
{
  this->deviceEKeyConstant = false;
  this->deviceDKeyConstant = false;
  this->useConstantKey = false;
  this->useConstantTables = false;
  this->enInstantiatedButInOtherDevice = false;
  this->deInstantiatedButInOtherDevice = false;
  this->inPlace = true; // use neighbours by default
  this->pin = new CudaPinned();
  this->ivIsCopy = false;
}

paracrypt::CudaAES::CudaAES(CudaAES* aes)
  : paracrypt::CudaAES::AES(aes), paracrypt::CUDABlockCipher::CUDABlockCipher(aes)
{
  this->setDevice(aes->device);
  this->deviceEKey = aes->deviceEKey;
  this->deviceEKeyConstant = aes->deviceEKeyConstant;
  this->deviceDKey = aes->deviceDKey;
  this->deviceDKeyConstant = aes->deviceDKeyConstant;
  this->deviceTe0 = aes->deviceTe0;
  this->deviceTe1 = aes->deviceTe1;
  this->deviceTe2 = aes->deviceTe2;
  this->deviceTe3 = aes->deviceTe3;
  this->deviceTd0 = aes->deviceTd0;
  this->deviceTd1 = aes->deviceTd1;
  this->deviceTd2 = aes->deviceTd2;
  this->deviceTd3 = aes->deviceTd3;
  this->deviceTd4 = aes->deviceTd4;
  this->useConstantKey = aes->useConstantKey;
  this->useConstantTables = aes->useConstantTables;
  //	this->malloc(aes->n_blocks);
  this->enInstantiatedButInOtherDevice = false;
  this->deInstantiatedButInOtherDevice = false;
  this->deviceIV = aes->deviceIV;
  this->ivIsCopy = true;
  this->pin = aes->pin;
}

paracrypt::CudaAES::~CudaAES()
{
  if (!this->isCopy && this->deviceEKey != NULL && !deviceEKeyConstant) {
    this->getDevice()->free(this->deviceEKey);
  }
  if (!this->isCopy && this->deviceDKey != NULL && !deviceDKeyConstant) {
    this->getDevice()->free(this->deviceDKey);
  }
  if (this->data != NULL) {
    this->getDevice()->free(this->data);
    this->getDevice()->delStream(this->stream);
  }
  if (!this->isCopy && this->deviceTe0 != NULL) {
    this->getDevice()->free(this->deviceTe0);
  }
  if (!this->isCopy && this->deviceTe1 != NULL) {
    this->getDevice()->free(this->deviceTe1);
  }
  if (!this->isCopy && this->deviceTe2 != NULL) {
    this->getDevice()->free(this->deviceTe2);
  }
  if (!this->isCopy && this->deviceTe3 != NULL) {
    this->getDevice()->free(this->deviceTe3);
  }
  if (!this->isCopy && this->deviceTd0 != NULL) {
    this->getDevice()->free(this->deviceTd0);
  }
  if (!this->isCopy && this->deviceTd1 != NULL) {
    this->getDevice()->free(this->deviceTd1);
  }
  if (!this->isCopy && this->deviceTd2 != NULL) {
    this->getDevice()->free(this->deviceTd2);
  }
  if (!this->isCopy && this->deviceTd3 != NULL) {
    this->getDevice()->free(this->deviceTd3);
  }
  if (!this->isCopy && this->deviceTd4 != NULL) {
    this->getDevice()->free(this->deviceTd4);
  }
  if (!this->ivIsCopy && this->deviceIV != NULL) {
    this->getDevice()->free(this->deviceIV);
  }
  if(this->neighborsDev != NULL) {
    this->getDevice()->free(this->neighborsDev);
  }
  if(this->neighborsPin != NULL) {
    this->pin->free((void*)this->neighborsPin);
  }
}

// must be called after setKey
void paracrypt::CudaAES::setDevice(CUDACipherDevice * device)
{
  if (!this->isCopy && this->deviceEKey != NULL && !deviceEKeyConstant) {
    this->getDevice()->free(this->deviceEKey);
    this->deviceEKey = NULL;
  }
  if (!this->isCopy && this->deviceDKey != NULL && !deviceDKeyConstant) {
    this->getDevice()->free(this->deviceDKey);
    this->deviceDKey = NULL;
  }
  if (this->data != NULL) {
    this->getDevice()->free(this->data);
    this->data = NULL;
    this->getDevice()->delStream(this->stream);
  }
  if (!this->isCopy && this->deviceTe0 != NULL) {
    this->getDevice()->free(this->deviceTe0);
    this->deviceTe0 = NULL;
  }
  if (!this->isCopy && this->deviceTe1 != NULL) {
    this->getDevice()->free(this->deviceTe1);
    this->deviceTe1 = NULL;
  }
  if (!this->isCopy && this->deviceTe2 != NULL) {
    this->getDevice()->free(this->deviceTe2);
    this->deviceTe2 = NULL;
  }
  if (!this->isCopy && this->deviceTe3 != NULL) {
    this->getDevice()->free(this->deviceTe3);
    this->deviceTe3 = NULL;
  }
  if (!this->isCopy && this->deviceTd0 != NULL) {
    this->getDevice()->free(this->deviceTd0);
    this->deviceTd0 = NULL;
  }
  if (!this->isCopy && this->deviceTd1 != NULL) {
    this->getDevice()->free(this->deviceTd1);
    this->deviceTd1 = NULL;
  }
  if (!this->isCopy && this->deviceTd2 != NULL) {
    this->getDevice()->free(this->deviceTd2);
    this->deviceTd2 = NULL;
  }
  if (!this->isCopy && this->deviceTd3 != NULL) {
    this->getDevice()->free(this->deviceTd3);
    this->deviceTd3 = NULL;
  }
  if (!this->isCopy && this->deviceTd4 != NULL) {
    this->getDevice()->free(this->deviceTd4);
    this->deviceTd4 = NULL;
  }
  if (!this->ivIsCopy && this->deviceIV != NULL) {
    this->getDevice()->free(this->deviceIV);
  }
  if(this->neighborsDev != NULL) {
    this->getDevice()->free(this->neighborsDev);
    this->neighborsDev = NULL;
  }
  if(this->neighborsPin != NULL) {
    this->pin->free((void*)this->neighborsPin);
    this->neighborsPin = NULL;
  }
  this->device = device;
  this->stream = this->getDevice()->addStream();
}

paracrypt::CUDACipherDevice * paracrypt::CudaAES::getDevice()
{
  return this->device;
}

void paracrypt::CudaAES::initDeviceEKey() {
  if(this->deInstantiatedButInOtherDevice || (!this->isCopy && this->deviceEKey == NULL)) {
    if(this->constantKey()) {
      int nKeyWords = (4 * (this->getEncryptionExpandedKey()->rounds + 1));
      this->deviceEKey = __setAesKey__(this->getEncryptionExpandedKey()->rd_key,nKeyWords);
      deviceEKeyConstant = true;
    }
    else {
      size_t keySize =
	(4 * (this->getEncryptionExpandedKey()->rounds + 1)) * sizeof(uint32_t);
      this->getDevice()->malloc((void **) &(this->deviceEKey), keySize);
      this->getDevice()->malloc((void **) &(this->deviceEKey), keySize);
      // copy to default stream so that all kernels in other streams can access the key
      this->getDevice()->memcpyTo(this->getEncryptionExpandedKey()->rd_key,
				  this->deviceEKey, keySize);
      deviceEKeyConstant = false;
    }
  }
}

void paracrypt::CudaAES::initDeviceDKey() {
  // CTR and CFB modes use the encryption function even for decryption
  initDeviceEKey();
}


// Only instantiate key when it is needed,
//  avoid instantiating both encryption/decryption
//  keys and wasting GPU mem. resources.
uint32_t* paracrypt::CudaAES::getDeviceEKey()
{
  this->initDeviceEKey();
  return this->deviceEKey;
}

uint32_t* paracrypt::CudaAES::getDeviceDKey()
{
  this->initDeviceDKey();
  return this->deviceDKey;
}

void paracrypt::CudaAES::malloc(unsigned int n_blocks, bool isInplace)
{
  if (this->data != NULL) {
    this->getDevice()->free(this->data);
  }
  int dataSize = AES_BLOCK_SIZE_B * n_blocks;
  this->getDevice()->malloc((void **) &(this->data), dataSize);

  if(this->neighborsDev != NULL) {
    this->getDevice()->free(this->neighborsDev);
  }
  if(this->neighborsPin != NULL) {
    this->pin->free((void*)this->neighborsPin);
  }
  this->inPlace = isInplace;
}

void paracrypt::CudaAES::setMode(Mode m)
{
  paracrypt::BlockCipher::setMode(m);
  if(this->neighborsDev != NULL && (m != CBC || m != CFB)) {
    this->getDevice()->free(this->neighborsDev);
    this->neighborsDev = NULL;
  }
}

void paracrypt::CudaAES::initDeviceTe()
{
  if(!this->constantTables()) {
    if (!this->isCopy && this->deviceTe0 == NULL)
      {
	this->getDevice()->malloc((void **) &(this->deviceTe0), TTABLE_SIZE); // 1024 = 256*4
	// memcpy to general stream 0 so that all copies of CudaAES can reutilize this table.
	this->getDevice()->memcpyTo((void*)Te0,this->deviceTe0, TTABLE_SIZE);
      }
    if (!this->isCopy && this->deviceTe1 == NULL)
      {
	this->getDevice()->malloc((void **) &(this->deviceTe1), TTABLE_SIZE);
	this->getDevice()->memcpyTo((void*)Te1,this->deviceTe1, TTABLE_SIZE);
      }
    if (!this->isCopy && this->deviceTe2 == NULL)
      {
	this->getDevice()->malloc((void **) &(this->deviceTe2), TTABLE_SIZE);
	this->getDevice()->memcpyTo((void*)Te2,this->deviceTe2, TTABLE_SIZE);
      }
    if (!this->isCopy && this->deviceTe3 == NULL)
      {
	this->getDevice()->malloc((void **) &(this->deviceTe3), TTABLE_SIZE);
	this->getDevice()->memcpyTo((void*)Te3,this->deviceTe3, TTABLE_SIZE);
      }
  }
}

void paracrypt::CudaAES::initDeviceTd()
{
  // CTR and CFB modes use the encryption function even for decryption
  initDeviceTe();
}

AES_KEY *paracrypt::CudaAES::getDecryptionExpandedKey()
{
  return paracrypt::AES::getEncryptionExpandedKey();
}

int paracrypt::CudaAES::setDecryptionKey(AES_KEY * expandedKey) {
  paracrypt::AES::setEncryptionKey(expandedKey);
  return 0;
}

void paracrypt::CudaAES::setIV(const unsigned char iv[], int bits)
{
  if(bits != 128) {
    std::cerr << "Wrong IV size for AES (an 128 bit input vector is required)."
	 << std::endl;
  }
  if ((!this->ivIsCopy && this->deviceIV == NULL) || this->isCopy) {
    this->deviceIV = NULL;
    this->getDevice()->malloc((void **) &(this->deviceIV), 16);
    this->ivIsCopy = false;
  }
  if (!this->ivIsCopy && this->deviceIV != NULL) {
    // TODO do not copy to device!! wait and copy at the same time
    //  we copy data in one single transference. This will produce
    //  a notable performance improvement in CBC and CFB modes.
    this->getDevice()->memcpyTo((void*)iv,(void*)this->deviceIV, 16);
  }
}

unsigned char* paracrypt::CudaAES::getIV()
{
  return this->deviceIV;
}

uint32_t*  paracrypt::CudaAES::getDeviceTe0()
{
  if(this->constantTables()) {
    return __Te0__();
  }
  else {
    this->initDeviceTe();
    return this->deviceTe0;
  }
}

uint32_t*  paracrypt::CudaAES::getDeviceTe1()
{
  if(this->constantTables()) {
    return __Te1__();
  }
  else {
    this->initDeviceTe();
    return this->deviceTe1;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTe2()
{
  if(this->constantTables()) {
    return __Te2__();
  }
  else {
    this->initDeviceTe();
    return this->deviceTe2;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTe3()
{
  if(this->constantTables()) {
    return __Te3__();
  }
  else {
    this->initDeviceTe();
    return this->deviceTe3;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTd0()
{
  if(this->constantTables()) {
    return __Td0__();
  }
  else {
    this->initDeviceTd();
    return this->deviceTd0;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTd1()
{
  if(this->constantTables()) {
    return __Td1__();
  }
  else {
    this->initDeviceTd();
    return this->deviceTd1;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTd2()
{
  if(this->constantTables()) {
    return __Td2__();
  }
  else {
    this->initDeviceTd();
    return this->deviceTd2;
  }
}

uint32_t* paracrypt::CudaAES::getDeviceTd3()
{
  if(this->constantTables()) {
    return __Td3__();
  }
  else {
    this->initDeviceTd();
    return this->deviceTd3;
  }
}

uint8_t* paracrypt::CudaAES::getDeviceTd4()
{
  if(this->constantTables()) {
    return __Td4__();
  }
  else {
    this->initDeviceTd();
    return this->deviceTd4;
  }
}

void paracrypt::CudaAES::constantKey(bool val){
  this->useConstantKey = val;
}
void paracrypt::CudaAES::constantTables(bool val){
  this->useConstantTables = val;
}
bool paracrypt::CudaAES::constantKey() {
  return this->useConstantKey;
}
bool paracrypt::CudaAES::constantTables(){
  return this->useConstantTables;
}

int paracrypt::CudaAES::setBlockSize(int bits) {
  return 0;
}

unsigned int paracrypt::CudaAES::getBlockSize() {
  return AES_BLOCK_SIZE;
}

int paracrypt::CudaAES::setKey(const unsigned char key[], int bits) {
  return paracrypt::AES::setKey(key,bits);
}

void paracrypt::CudaAES::waitFinish() {
  this->getDevice()->waitMemcpyFrom(this->stream);
}

bool paracrypt::CudaAES::checkFinish() {
  return this->getDevice()->checkMemcpyFrom(this->stream);
}

bool paracrypt::CudaAES::isInplace() {
  return this->inPlace;
}

// TODO do not copy to device!! wait and copy at the same time
//  we copy data in one single transference. This will produce
//  a notable performance improvement in CBC and CFB modes.
void paracrypt::CudaAES::transferNeighborsToGPU(
						const unsigned char blocks[],
						std::streamsize n_blocks)
{
  unsigned int blockSizeBytes = this->getBlockSize()/8;
  for(unsigned int i = 0; i < this->nNeighbors; i++) {
    // neighBlock calculation works for both CBC and CFB
    unsigned int neighBlock = ((i+1)*this->cipherBlocksPerThreadBlock)-1;
    unsigned int despPin = i*blockSizeBytes;
    unsigned int despBlocks = neighBlock*blockSizeBytes;
    std::memcpy(neighborsPin+despPin,blocks+despBlocks,blockSizeBytes);
  }
  this->getDevice()->memcpyTo(this->neighborsPin, this->neighborsDev, this->neighSize, this->stream);
}

int paracrypt::CudaAES::encrypt(const unsigned char in[],
				const unsigned char out[],
				std::streamsize n_blocks)
{
  int threadsPerCipherBlock = this->getThreadsPerCipherBlock();
  int gridSize = this->getDevice()->getGridSize(n_blocks, threadsPerCipherBlock);
  int threadsPerBlock = this->getDevice()->getThreadsPerThreadBlock();
  size_t dataSize = n_blocks * AES_BLOCK_SIZE_B;
  uint32_t *key = this->getDeviceEKey();
  assert(key != NULL);
  int rounds = this->getEncryptionExpandedKey()->rounds;

  this->getDevice()->memcpyTo((void *) in, this->data, dataSize,
			      this->stream);

  this->getDevice()->memcpyFrom(this->data, (void *) out, dataSize,
				this->stream);

  paracrypt::BlockCipher::encrypt(in,out,n_blocks); // increment block offset
  return 0;
}

int paracrypt::CudaAES::decrypt(const unsigned char in[],
				const unsigned char out[],
				std::streamsize n_blocks)
{
  // CTR and CFB modes use the encryption function even for decryption
  this->setInitialBlockOffset(this->getDecryptBlockOffset());
  return encrypt(in, out, n_blocks);
}
