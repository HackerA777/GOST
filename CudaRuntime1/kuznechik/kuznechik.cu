#include "kuznechik.cuh"
#include <random>
#include <chrono>
#include <cuda/std/array>

#define cudaCheck(e) cudaCheck__((e), __FILE__, __LINE__)

__device__ static constexpr uint8_t sTable[256] = {
	0xFC, 0xEE, 0xDD, 0x11, 0xCF, 0x6E, 0x31, 0x16,
	0xFB, 0xC4, 0xFA, 0xDA, 0x23, 0xC5, 0x04, 0x4D,
	0xE9, 0x77, 0xF0, 0xDB, 0x93, 0x2E, 0x99, 0xBA,
	0x17, 0x36, 0xF1, 0xBB, 0x14, 0xCD, 0x5F, 0xC1,
	0xF9, 0x18, 0x65, 0x5A, 0xE2, 0x5C, 0xEF, 0x21,
	0x81, 0x1C, 0x3C, 0x42, 0x8B, 0x01, 0x8E, 0x4F,
	0x05, 0x84, 0x02, 0xAE, 0xE3, 0x6A, 0x8F, 0xA0,
	0x06, 0x0B, 0xED, 0x98, 0x7F, 0xD4, 0xD3, 0x1F,
	0xEB, 0x34, 0x2C, 0x51, 0xEA, 0xC8, 0x48, 0xAB,
	0xF2, 0x2A, 0x68, 0xA2, 0xFD, 0x3A, 0xCE, 0xCC,
	0xB5, 0x70, 0x0E, 0x56, 0x08, 0x0C, 0x76, 0x12,
	0xBF, 0x72, 0x13, 0x47, 0x9C, 0xB7, 0x5D, 0x87,
	0x15, 0xA1, 0x96, 0x29, 0x10, 0x7B, 0x9A, 0xC7,
	0xF3, 0x91, 0x78, 0x6F, 0x9D, 0x9E, 0xB2, 0xB1,
	0x32, 0x75, 0x19, 0x3D, 0xFF, 0x35, 0x8A, 0x7E,
	0x6D, 0x54, 0xC6, 0x80, 0xC3, 0xBD, 0x0D, 0x57,
	0xDF, 0xF5, 0x24, 0xA9, 0x3E, 0xA8, 0x43, 0xC9,
	0xD7, 0x79, 0xD6, 0xF6, 0x7C, 0x22, 0xB9, 0x03,
	0xE0, 0x0F, 0xEC, 0xDE, 0x7A, 0x94, 0xB0, 0xBC,
	0xDC, 0xE8, 0x28, 0x50, 0x4E, 0x33, 0x0A, 0x4A,
	0xA7, 0x97, 0x60, 0x73, 0x1E, 0x00, 0x62, 0x44,
	0x1A, 0xB8, 0x38, 0x82, 0x64, 0x9F, 0x26, 0x41,
	0xAD, 0x45, 0x46, 0x92, 0x27, 0x5E, 0x55, 0x2F,
	0x8C, 0xA3, 0xA5, 0x7D, 0x69, 0xD5, 0x95, 0x3B,
	0x07, 0x58, 0xB3, 0x40, 0x86, 0xAC, 0x1D, 0xF7,
	0x30, 0x37, 0x6B, 0xE4, 0x88, 0xD9, 0xE7, 0x89,
	0xE1, 0x1B, 0x83, 0x49, 0x4C, 0x3F, 0xF8, 0xFE,
	0x8D, 0x53, 0xAA, 0x90, 0xCA, 0xD8, 0x85, 0x61,
	0x20, 0x71, 0x67, 0xA4, 0x2D, 0x2B, 0x09, 0x5B,
	0xCB, 0x9B, 0x25, 0xD0, 0xBE, 0xE5, 0x6C, 0x52,
	0x59, 0xA6, 0x74, 0xD2, 0xE6, 0xF4, 0xB4, 0xC0,
	0xD1, 0x66, 0xAF, 0xC2, 0x39, 0x4B, 0x63, 0xB6 };

__device__ static constexpr uint8_t revSTable[256] = {
	0xA5, 0x2D, 0x32, 0x8F, 0x0E, 0x30, 0x38, 0xC0,
	0x54, 0xE6, 0x9E, 0x39, 0x55, 0x7E, 0x52, 0x91,
	0x64, 0x03, 0x57, 0x5A, 0x1C, 0x60, 0x07, 0x18,
	0x21, 0x72, 0xA8, 0xD1, 0x29, 0xC6, 0xA4, 0x3F,
	0xE0, 0x27, 0x8D, 0x0C, 0x82, 0xEA, 0xAE, 0xB4,
	0x9A, 0x63, 0x49, 0xE5, 0x42, 0xE4, 0x15, 0xB7,
	0xC8, 0x06, 0x70, 0x9D, 0x41, 0x75, 0x19, 0xC9,
	0xAA, 0xFC, 0x4D, 0xBF, 0x2A, 0x73, 0x84, 0xD5,
	0xC3, 0xAF, 0x2B, 0x86, 0xA7, 0xB1, 0xB2, 0x5B,
	0x46, 0xD3, 0x9F, 0xFD, 0xD4, 0x0F, 0x9C, 0x2F,
	0x9B, 0x43, 0xEF, 0xD9, 0x79, 0xB6, 0x53, 0x7F,
	0xC1, 0xF0, 0x23, 0xE7, 0x25, 0x5E, 0xB5, 0x1E,
	0xA2, 0xDF, 0xA6, 0xFE, 0xAC, 0x22, 0xF9, 0xE2,
	0x4A, 0xBC, 0x35, 0xCA, 0xEE, 0x78, 0x05, 0x6B,
	0x51, 0xE1, 0x59, 0xA3, 0xF2, 0x71, 0x56, 0x11,
	0x6A, 0x89, 0x94, 0x65, 0x8C, 0xBB, 0x77, 0x3C,
	0x7B, 0x28, 0xAB, 0xD2, 0x31, 0xDE, 0xC4, 0x5F,
	0xCC, 0xCF, 0x76, 0x2C, 0xB8, 0xD8, 0x2E, 0x36,
	0xDB, 0x69, 0xB3, 0x14, 0x95, 0xBE, 0x62, 0xA1,
	0x3B, 0x16, 0x66, 0xE9, 0x5C, 0x6C, 0x6D, 0xAD,
	0x37, 0x61, 0x4B, 0xB9, 0xE3, 0xBA, 0xF1, 0xA0,
	0x85, 0x83, 0xDA, 0x47, 0xC5, 0xB0, 0x33, 0xFA,
	0x96, 0x6F, 0x6E, 0xC2, 0xF6, 0x50, 0xFF, 0x5D,
	0xA9, 0x8E, 0x17, 0x1B, 0x97, 0x7D, 0xEC, 0x58,
	0xF7, 0x1F, 0xFB, 0x7C, 0x09, 0x0D, 0x7A, 0x67,
	0x45, 0x87, 0xDC, 0xE8, 0x4F, 0x1D, 0x4E, 0x04,
	0xEB, 0xF8, 0xF3, 0x3E, 0x3D, 0xBD, 0x8A, 0x88,
	0xDD, 0xCD, 0x0B, 0x13, 0x98, 0x02, 0x93, 0x80,
	0x90, 0xD0, 0x24, 0x34, 0xCB, 0xED, 0xF4, 0xCE,
	0x99, 0x10, 0x44, 0x40, 0x92, 0x3A, 0x01, 0x26,
	0x12, 0x1A, 0x48, 0x68, 0xF5, 0x81, 0x8B, 0xC7,
	0xD6, 0x20, 0x0A, 0x08, 0x00, 0x4C, 0xD7, 0x74
};

__device__ static constexpr uint8_t lVector[16] = { 1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148 };

__device__ uint8_t multiplicationGalua(uint8_t first, uint8_t second) {
	uint8_t result = 0;
	uint8_t hiBit;
	for (int i = 0; i < 8; i++) {
		if (second & 1) {
			result ^= first;
		}
		hiBit = first & 0x80;
		first <<= 1;
		if (hiBit) {
			first ^= 0xc3;
		}
		second >>= 1;
	}
	return result;
}

__device__ kuznechikByteVector XOR(const kuznechikByteVector src1, const kuznechikByteVector src2) {
	//kuznechikHalfVector lo = src1.halfsData.lo.halfVector ^ src2.halfsData.lo.halfVector;
	//kuznechikHalfVector hi = src1.halfsData.hi.halfVector ^ src2.halfsData.hi.halfVector;
	return kuznechikByteVector(src1.halfsData.lo.halfVector ^ src2.halfsData.lo.halfVector, src1.halfsData.hi.halfVector ^ src2.halfsData.hi.halfVector);
}

__device__ kuznechikByteVector transformationS(const kuznechikByteVector src) {
	kuznechikByteVector tmp{};
	for (size_t i = 0; i < 16; i++) {
		tmp.bytes[i] = sTable[src.bytes[i]];
	}
	return tmp;
}

__device__ kuznechikByteVector transformationR(const kuznechikByteVector src) {
	uint8_t a_15 = 0;
	kuznechikByteVector internal = { 0, 0 };
	for (int i = 15; i >= 0; i--) {
		if (i == 0)
		{
			internal.bytes[15] = src.bytes[i];
		}
		else
		{
			internal.bytes[i - 1] = src.bytes[i];
		}
		a_15 ^= multiplicationGalua(src.bytes[i], lVector[i]);
	}
	internal.bytes[15] = a_15;
	return internal;
}

__device__ kuznechikByteVector transformaionL(const kuznechikByteVector& inData) {
	kuznechikByteVector tmp = inData;
	for (int i = 0; i < 16; i++) {
		tmp = transformationR(tmp);
	}
	return tmp;
}

__device__ kuznechikByteVector transformationF(const kuznechikByteVector src, const kuznechikByteVector cons) {
	kuznechikByteVector tmp;
	tmp = XOR(src, cons);
	tmp = transformationS(tmp);
	kuznechikByteVector d;
	d = transformaionL(tmp);
	return d;
}

__device__  cuda::std::array <kuznechikByteVector, 32> getConstTableKuz() {
	cuda::std::array <kuznechikByteVector, 32> constTable;
	kuznechikByteVector numberIter = { kuznechikHalfVector(0), kuznechikHalfVector(0) };
	numberIter.bytes[0] += 0x01;
	for (int i = 0; i < 32; i++) {
		kuznechikByteVector result = { 0, 0 };
		result = transformaionL(numberIter);
		constTable[i] = result;
		numberIter.bytes[0] += 0x01;
	}
	return constTable;
}

__global__  void getRoundKeys(const kuznechikKeys& mainKey, kuznechikByteVector* roundKeysKuznechik) {
	cuda::std::array <kuznechikByteVector, 32> constTable = getConstTableKuz();
	uint8_t lo[16];
	uint8_t hi[16];
	size_t numberKey = 0;
	for (int i = 0; i < 16; ++i) {
		lo[i] = mainKey.bytes[i];
		hi[i] = mainKey.bytes[i + 16];
	}
	kuznechikByteVector leftPart(lo);
	kuznechikByteVector rightPart(hi);
	roundKeysKuznechik[0] = rightPart;
	roundKeysKuznechik[1] = leftPart;
	numberKey += 2;
	for (size_t i = 1; i < 5; i++) {
		int iter = 0;
		for (size_t j = 0; j < 8; j++) {
			kuznechikByteVector tmp2 = leftPart;
			leftPart = rightPart;
			kuznechikByteVector tmp = transformationF(rightPart, constTable[(8 * (i - 1) + j)]);
			rightPart = XOR(tmp, tmp2);
			iter++;
		}
		roundKeysKuznechik[numberKey] = rightPart;
		numberKey++;
		roundKeysKuznechik[numberKey] = leftPart;
		numberKey++;
	}
}

__device__ kuznechikByteVector revTransformationS(const kuznechikByteVector src) {
	kuznechikByteVector tmp{};
	for (size_t i = 0; i < 16; i++) {
		tmp.bytes[i] = revSTable[src.bytes[i]];
	}
	return tmp;
}

__device__ kuznechikByteVector revTransformationR(const kuznechikByteVector src)
{
	uint8_t a0 = src.bytes[15];
	kuznechikByteVector internal = { 0, 0 };
	for (int i = 1; i < 16; i++)
	{
		internal.bytes[i] = src.bytes[i - 1];
		a0 ^= multiplicationGalua(internal.bytes[i], lVector[i]);
	}
	internal.bytes[0] = a0;
	return internal;
}

__device__ kuznechikByteVector revTransformationL(const kuznechikByteVector& inData)
{
	kuznechikByteVector tmp = inData;
	for (int i = 0; i < 16; i++) {
		tmp = revTransformationR(tmp);
	}
	return tmp;
}


__global__ void encryptOneBlock(kuznechikByteVector* src, const kuznechikByteVector * roundKeysKuznechik, const size_t count) {
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	auto tcnt = gridDim.x * blockDim.x;
	for (auto i = tid; i < count; i += tcnt) {
		kuznechikByteVector temp = src[i];
		for (int j = 0; j < 9; ++j) {
			temp = XOR(temp, roundKeysKuznechik[j]);
			temp = transformationS(temp);
			temp = transformaionL(temp);
		}
		src[i] = XOR(temp, roundKeysKuznechik[9]);
	}
}

__global__ void decryptOneBlock(kuznechikByteVector* src, const kuznechikByteVector* roundKeysKuznechik, const size_t count) {
	auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	auto tcnt = gridDim.x * blockDim.x;
	for (auto i = tid; i < count; i += tcnt) {
		kuznechikByteVector temp = XOR(src[i], roundKeysKuznechik[9]);
		for (int j = 8; j >= 0; --j) {
			temp = revTransformationL(temp);
			temp = revTransformationS(temp);
			temp = XOR(temp, roundKeysKuznechik[j]);
		}
		src[i] = temp;
	}
}

void kuznechik::processData(kuznechikByteVector* src, kuznechikByteVector* dest, const size_t countBlocks, bool enc) const {
	
	cuda_ptr<kuznechikByteVector> dev_keys = cuda_alloc<kuznechikByteVector>(10);
	cuda_ptr<kuznechikByteVector[]> dev_blocks = cuda_alloc<kuznechikByteVector[]>(countBlocks);
	cudaError_t cudaStatus;

	cudaCheck(cudaMemcpy(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));

	cudaCheck(cudaMemcpy(dev_blocks.get(), src, countBlocks * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));

	//cudaCheck(cudaGetLastError());

	if (enc) {
		encryptOneBlock <<<blockSize, gridSize>>>(dev_blocks.get(), dev_keys.get(), countBlocks);
	}
	else {
		decryptOneBlock <<<blockSize, gridSize>>> (dev_blocks.get(), dev_keys.get(), countBlocks);
	}

	//cudaCheck(cudaGetLastError());

	cudaCheck(cudaDeviceSynchronize());

	cudaCheck(cudaMemcpy(dest, dev_blocks.get(), countBlocks * sizeof(kuznechikByteVector), cudaMemcpyDeviceToHost));
}

void kuznechik::checkEcnAndDec() {
	uint8_t testString[16] = {
		0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
	};
	uint8_t encryptValidString[16] = {
		0xcd, 0xed, 0xd4, 0xb9, 0x42, 0x8d, 0x46, 0x5a, 0x30, 0x24, 0xbc, 0xbe, 0x90, 0x9d, 0x67, 0x7f
	};
	uint8_t keys[32] = {
		 0xef, 0xcd, 0xab, 0x89, 
		 0x67, 0x45, 0x23, 0x01, 
		 0x10, 0x32, 0x54, 0x76, 
		 0x98, 0xba, 0xdc, 0xfe, 
		 0x77, 0x66, 0x55, 0x44, 
		 0x33, 0x22, 0x11, 0x00, 
		 0xff, 0xee, 0xdd, 0xcc, 
		 0xbb, 0xaa, 0x99, 0x88
	};
	kuznechikByteVector testBlock(testString);
	kuznechikByteVector resultBlock(testString);
	kuznechikByteVector validBlock(encryptValidString);
	kuznechikKeys testKeys(keys);

	kuznechik testAlgorithm(testKeys, 16, 8, 8);
	testAlgorithm.processData(&testBlock, &resultBlock, 1, true);

	std::cout << "Test string: " << testBlock << std::endl;
	std::cout << "Result encryption test string: " << resultBlock << std::endl;
	std::cout << "Valide encryption string: " << validBlock << std::endl;
	if (validBlock.ull == resultBlock.ull)
		std::cout << "Encryption algoritm valid!" << std::endl;
	else
		std::cout << "Encryption algoritm unvalid!" << std::endl;

	testAlgorithm.processData(&resultBlock, &resultBlock, 1, false);
	std::cout << "Result decryption test string: " << resultBlock << std::endl;
	if (testBlock.ull == resultBlock.ull)
		std::cout << "Decryption algoritm valid!" << std::endl;
	else
		std::cout << "Decryption algoritm unvalid!" << std::endl;
}

double kuznechik::testSpeedUnequalBytes() {
	size_t size = buffSize;
	std::vector<kuznechikByteVector> data(size);
	uint32_t i = 0;
	for (auto& b : data) b.ull = ++i << 32 | ++i;
	processData(data.data(), data.data(), 10, true);
	using duration = std::chrono::duration<double, std::milli>;
	auto start = std::chrono::high_resolution_clock::now();

	processData(data.data(), data.data(), data.size(), true);
	duration time = std::chrono::high_resolution_clock::now() - start;
	double speed = (size * sizeof(magmaBlockT) / 1024.0 / 1024 / 1024) / time.count() * 1000;
	std::cout << "SIZE: " << size << "\tTIME: " << time.count() << "ms\t SPEED: " << speed << " GB/s" << std::endl;
	return speed;
}

void kuznechik::searchBestBlockAndGridSize() {
	cudaDeviceProp prop;
	cudaError_t(cudaGetDeviceProperties(&prop, 0));
	size_t size = 1024 * 1024 * 1024 * 2;
	double max_speed = 0, max_speed_avg = 0;
	size_t gridSize = 8, bestGridSize;
	size_t blockSize = 8, bestBlockSize;
	for (size_t i = blockSize; i <= prop.maxThreadsDim[0]; i *= 2) {
		for (size_t j = gridSize; j <= prop.maxThreadsDim[1]; j *= 2) {
			setBlockSize(i);
			setGridSize(j);
			for (size_t k = 0; k < 10; ++k) {
				max_speed_avg += testSpeedUnequalBytes();
			}
			max_speed_avg = max_speed_avg / 10.0;
			if (max_speed < max_speed_avg) {
				max_speed = max_speed_avg;
				bestBlockSize = i;
				bestGridSize = j;
			}
			std::cout << "\nAVG SPEED: " << max_speed_avg << " BLOCK SIZE: " << i << " GRID SIZE: " << j << "\n" << std::endl;
			max_speed_avg = 0;
		}
		
	}
	std::cout << "Max speed: " << max_speed << "\nBest block size: " << bestBlockSize << "\nBest grid size: " << bestGridSize << std::endl;
}

void kuznechik::setGridSize(const size_t newGridSize) {
	gridSize = newGridSize;
}

void kuznechik::setBlockSize(const size_t newBlockSize) {
	blockSize = newBlockSize;
}

kuznechik::kuznechik(const kuznechikKeys& mainKey, const size_t buffSize, const size_t blockSize, const size_t gridSize) {
	this->buffSize = buffSize;
	this->blockSize = blockSize;
	this->gridSize = gridSize;

	cuda_ptr<kuznechikKeys> dev_keys = cuda_alloc<kuznechikKeys>();
	cuda_ptr<kuznechikByteVector> dev_res_keys = cuda_alloc<kuznechikByteVector>(10);

	cudaCheck(cudaMemcpy(dev_keys.get(), mainKey.bytes, sizeof(kuznechikKeys), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dev_res_keys.get(), roundKeysKuznechik, sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));

	getRoundKeys <<<blockSize, gridSize>>> (*dev_keys, dev_res_keys.get());

	cudaCheck(cudaGetLastError());
	cudaCheck(cudaDeviceSynchronize());

	cudaCheck(cudaMemcpy(this->roundKeysKuznechik, dev_res_keys.get(), 10 * sizeof(kuznechikByteVector), cudaMemcpyDeviceToHost));
	//cudaFree(dev_keys.get());
	//cudaCheck(cudaGetLastError());
}

std::vector<float> kuznechik::testDefault(std::vector<kuznechikByteVector>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
	cudaCheck(cudaGetLastError());
	std::vector<float> time{ 0, 0 };

	const size_t countBlocks = data.size();
	std::cout << "countBlock: " << countBlocks << std::endl;
	const size_t dataSize = countBlocks * sizeof(kuznechikByteVector);

	cuda_ptr<kuznechikByteVector> dev_keys = cuda_alloc<kuznechikByteVector>(10);
	cuda_ptr<kuznechikByteVector[]> dev_blocks = cuda_alloc<kuznechikByteVector[]>(countBlocks);

	cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
	cudaCheck(cudaEventCreate(&startEnc));
	cudaCheck(cudaEventCreate(&stopEnc));
	cudaCheck(cudaEventCreate(&startCopyAndEnc));
	cudaCheck(cudaEventCreate(&stopCopyAndEnc));

	if (encryptStatus) {

		//std::cout << "Data before encrypt: " << data.data() << std::endl;

		cudaCheck(cudaEventRecord(startCopyAndEnc));
		//_Thrd_sleep_for(10000);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(startEnc));

		cudaCheck(cudaGetLastError());

		encryptOneBlock <<< blockSize, gridSize >>> (dev_blocks.get(), dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));

		cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopCopyAndEnc));

		cudaCheck(cudaGetLastError());

		//std::cout << "Data after encrypt: " << data.data() << std::endl;
	}
	else {
		//std::cout << "Data before decrypt: " << data.data() << std::endl;

		cudaCheck(cudaEventRecord(startCopyAndEnc));

		cudaCheck(cudaMemcpyAsync(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(startEnc));

		decryptOneBlock << <blockSize, gridSize >> > (dev_blocks.get(), dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopCopyAndEnc));

		cudaCheck(cudaGetLastError());

		//std::cout << "Data after decrypt: " << data.data() << std::endl;
	}

	cudaCheck(cudaEventSynchronize(stopCopyAndEnc));
	cudaCheck(cudaGetLastError());
	cudaEventElapsedTime(&time[0], startCopyAndEnc, stopCopyAndEnc);
	cudaEventElapsedTime(&time[1], startEnc, stopEnc);

	cudaCheck(cudaEventDestroy(startCopyAndEnc));
	cudaCheck(cudaEventDestroy(stopCopyAndEnc));
	cudaCheck(cudaEventDestroy(startEnc));
	cudaCheck(cudaEventDestroy(stopEnc));

	return time;
}

std::vector<float> kuznechik::testPinned(std::vector<kuznechikByteVector>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
	std::vector<float> time{ 0, 0 };

	const size_t countBlocks = data.size();
	const size_t dataSize = countBlocks * sizeof(kuznechikByteVector);

	cuda_ptr<kuznechikByteVector> dev_keys = cuda_alloc<kuznechikByteVector>(10);
	cuda_ptr<kuznechikByteVector[]> dev_blocks = cuda_alloc<kuznechikByteVector[]>(countBlocks);

	cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
	cudaCheck(cudaEventCreate(&startEnc));
	cudaCheck(cudaEventCreate(&stopEnc));
	cudaCheck(cudaEventCreate(&startCopyAndEnc));
	cudaCheck(cudaEventCreate(&stopCopyAndEnc));

	if (encryptStatus) {

		//std::cout << "Data before encrypt: " << data.data() << std::endl;

		cudaCheck(cudaEventRecord(startCopyAndEnc));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));

		cudaCheck(cudaGetLastError());
		cudaCheck(cudaHostRegister((void*)data.data(), dataSize, cudaHostRegisterDefault));

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(startEnc));

		encryptOneBlock <<< blockSize, gridSize >>> (dev_blocks.get(), dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));

		cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));

		cudaCheck(cudaStreamSynchronize(0));

		//if (blocks != out_blocks)
		cudaCheck(cudaHostUnregister((void*)data.data()));

		cudaCheck(cudaEventRecord(stopCopyAndEnc));

		//std::cout << "Data after encrypt: " << data.data() << std::endl;
	}
	else {
		//std::cout << "Data before decrypt: " << data.data() << std::endl;

		cudaCheck(cudaEventRecord(startCopyAndEnc));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));
		cudaCheck(cudaHostRegister((void*)data.data(), dataSize, cudaHostRegisterDefault));

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(startEnc));

		decryptOneBlock <<< blockSize, gridSize >>> (dev_blocks.get(), dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));

		cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaHostUnregister((void*)data.data()));
		cudaCheck(cudaGetLastError());

		cudaCheck(cudaStreamSynchronize(0));

		cudaCheck(cudaEventRecord(stopCopyAndEnc));

		//std::cout << "Data after decrypt: " << data.data() << std::endl;
	}

	cudaCheck(cudaEventSynchronize(stopCopyAndEnc));
	cudaEventElapsedTime(&time[0], startCopyAndEnc, stopCopyAndEnc);
	cudaEventElapsedTime(&time[1], startEnc, stopEnc);

	cudaCheck(cudaEventDestroy(startCopyAndEnc));
	cudaCheck(cudaEventDestroy(stopCopyAndEnc));
	cudaCheck(cudaEventDestroy(startEnc));
	cudaCheck(cudaEventDestroy(stopEnc));

	//cudaCheck(cudaHostUnregister((void*)out_blocks));

	return time;
}

std::vector<float> kuznechik::testManaged(std::vector<kuznechikByteVector>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
	std::vector<float> time{ 0, 0 };
	kuznechikByteVector* buffer;

	const size_t countBlocks = data.size();

	cuda_ptr<kuznechikByteVector> dev_keys = cuda_alloc<kuznechikByteVector>(10);

	size_t dataSize = countBlocks * sizeof(kuznechikByteVector);

	cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
	cudaCheck(cudaEventCreate(&startEnc));
	cudaCheck(cudaEventCreate(&stopEnc));
	cudaCheck(cudaEventCreate(&startCopyAndEnc));
	cudaCheck(cudaEventCreate(&stopCopyAndEnc));

	cudaCheck(cudaMemcpyAsync(dev_keys.get(), roundKeysKuznechik, 10 * sizeof(kuznechikByteVector), cudaMemcpyHostToDevice));

	cudaCheck(cudaMallocManaged(&buffer, dataSize));

	cudaCheck(cudaEventRecord(startCopyAndEnc));
	cudaCheck(cudaMemcpyAsync(buffer, data.data(), dataSize, cudaMemcpyHostToHost));

	cudaCheck(cudaGetLastError());

	//std::cout << "Buffer: " << buffer << std::endl;

	if (encryptStatus) {
		cudaCheck(cudaEventRecord(startEnc));

		encryptOneBlock <<< blockSize, gridSize >>> (buffer, dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));
	}
	else {

		cudaCheck(cudaEventRecord(startEnc));

		decryptOneBlock <<< blockSize, gridSize >>> (buffer, dev_keys.get(), countBlocks);

		cudaCheck(cudaGetLastError());

		cudaCheck(cudaEventRecord(stopEnc));
	}

	//std::cout << "After Decrypt. Buffer: " << buffer->data()<< std::endl;

	cudaCheck(cudaMemcpyAsync(data.data(), buffer, dataSize, cudaMemcpyHostToHost));

	cudaCheck(cudaEventRecord(stopCopyAndEnc));

	cudaCheck(cudaEventSynchronize(stopCopyAndEnc));
	cudaEventElapsedTime(&time[0], startCopyAndEnc, stopCopyAndEnc);
	cudaEventElapsedTime(&time[1], startEnc, stopEnc);

	cudaCheck(cudaEventDestroy(startCopyAndEnc));
	cudaCheck(cudaEventDestroy(stopCopyAndEnc));
	cudaCheck(cudaEventDestroy(startEnc));
	cudaCheck(cudaEventDestroy(stopEnc));

	cudaCheck(cudaFree(buffer));

	return time;
}

