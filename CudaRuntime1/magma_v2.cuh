//#pragma once
//#ifndef H_MAGMA_V2
//#define H_MAGMA_V2
//#include "structures.cuh"
//
//class magma_v2 {
//private:
//	magmaKeySet keys;
//	size_t buffSize;
//	size_t buffSizeOneBlock;
//	int streamsCount;
//	std::vector<cudaStream_t> streams; 
//	// завести доп переменную size_t stream_buffsize
//	//cuda_ptr<magmaBlockT[]> memmory;
//	std::vector<cuda_ptr<magmaBlockT[]>> ptrs; // std::vector<magmaBlockT*> ptr;
//
//	void copyKeys(const unsigned char inputKeys[32]) {
//		std::copy(inputKeys, inputKeys + 32, keys.keys->bytes);
//	}
//
//public:
//	//magma() {};
//	magma_v2(const unsigned char keys[32], const size_t buffSize); // cuda ptr buffer
//
//	// array streams
//	// cudaPtr all data
//
//	void checkEcnAndDec();
//	double testSpeedUnequalBytes();
//
//	void encryptCuda(const cuda_ptr<magmaBlockT[]> blocks, cuda_ptr<magmaBlockT[]> out_block, const size_t countBlocks); // заменить на magmaBlockT входные данные и выходные тоже
//	// заменить cuda_ptr на указатели на данные из RAM (обычные указатели)
//	void decryptCuda(const uint8_t* block, uint8_t* out_block, const magmaKeySet inputKeys, const size_t dataSize);
//	void searchBestBlockAndGridSize();
//	~magma_v2();
//};
//
//#endif