#include "structures.cuh"

class magma {
private:
	key_set keys;
	size_t buffSize;
	unsigned int blockSize;
	unsigned int gridSize;

public:
	magma(const unsigned char keys[32], const size_t buffSize, const unsigned int blockSize, const unsigned int gridSize);

	~magma() {
	}
};