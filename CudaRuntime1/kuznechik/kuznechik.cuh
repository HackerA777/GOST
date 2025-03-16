#pragma once
#ifndef H_KUZNECHIK
#define H_KUZNECHIK
#include "../structures/structures.cuh"
#include <vector>
#include <array>

class kuznechik {
private:
    kuznechikKeys keys;
    kuznechikByteVector roundKeysKuznechik[10];
    size_t blockSize, gridSize, buffSize;
public:
    kuznechik();
    kuznechik(const kuznechikKeys& key, const size_t buffSize, const size_t blockSize, const size_t gridSize);
    void processData(kuznechikByteVector* src, kuznechikByteVector* dest, const size_t countBlocks, bool enc)  const;

    void changeKey(const unsigned char* key);
    kuznechikKeys getKeys();

    void setBlockSize(const size_t newBlockSize);
    void setGridSize(const size_t newGridSize);

    void checkEcnAndDec();
    double testSpeedUnequalBytes();
    void searchBestBlockAndGridSize();
};

#endif
