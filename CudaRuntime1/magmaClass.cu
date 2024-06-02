#include "magmaClass.cuh"

magma::magma(const unsigned char keys[32], const size_t buffSize, const unsigned int blockSize, const unsigned int gridSize) {
	std::copy(keys, keys + 32, this->keys.bytes);
	this->buffSize = buffSize;
	this->blockSize = blockSize;
	this->gridSize = gridSize;
}