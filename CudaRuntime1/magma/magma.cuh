#pragma once
#ifndef H_MAGMA
#define H_MAGMA
#include "../structures/structures.cuh"

class magma {
private:
	magmaKeySet keys;
	size_t buffSize;
	unsigned int gridSize;
	unsigned int blockSize;

	void copyKeys(const unsigned char inputKeys[32]) {
		std::copy(inputKeys, inputKeys + 32, keys.keys->bytes);
	}

public:
	magma();
	magma(const unsigned char keys[32], const size_t buffSize, const unsigned int gridSize, const unsigned int blockSize);

	void changeKey(const unsigned char* key);
	magmaKeySet getKeys();
	
	void checkEcnAndDec();
	double testSpeedUnequalBytes();

	void setGridSize(const size_t newGridSize);
	void setBlockSize(const size_t newBlockSize);

	void encryptCuda(const uint8_t* block, uint8_t* out_block, const magmaKeySet inputKeys, const size_t dataSize);
	void decryptCuda(const uint8_t* block, uint8_t* out_block, const magmaKeySet inputKeys, const size_t dataSize);
	void searchBestBlockAndGridSize();
	~magma() {
	}
};

#endif
