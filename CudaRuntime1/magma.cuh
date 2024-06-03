#ifndef H_MAGMA
#define H_MAGMA
#include "structures.cuh"

class magma {
private:
	key_set keys;
	size_t buffSize;
	unsigned int blockSize;
	unsigned int gridSize;

	void copyKesys(const unsigned char inputKeys[32]) {
		std::copy(inputKeys, inputKeys + 32, keys.keys->bytes);
	}

public:
	magma() {};
	magma(const unsigned char keys[32], const size_t buffSize, const unsigned int blockSize, const unsigned int gridSize);
	void checkEcnAndDec();

	void encryptCuda(const uint8_t* block, uint8_t* out_block, const size_t dataSize, const unsigned int blockSize, const unsigned int gridSize);
	void decryptCuda(const uint8_t* block, uint8_t* out_block, const size_t dataSize, const unsigned int blockSize, const unsigned int gridSize);
	~magma() {
	}
};

#endif