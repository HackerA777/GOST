#pragma once
#ifndef H_KUZNECHIK
#define H_KUZNECHIK
#include "../structures/structures.cuh"
#include "../table.cuh"
#include <vector>
#include <array>

class kuznechik {
private:
    kuznechikByteVector roundKeysKuznechik[10];
    size_t blockSize, gridSize, buffSize;
public:
    kuznechik(const kuznechikKeys& key, const size_t buffSize, const size_t blockSize, const size_t gridSize);
    void processData(kuznechikByteVector* src, kuznechikByteVector* dest, const size_t countBlocks, bool enc)  const;

    void processData2(kuznechikByteVector* src, kuznechikByteVector* dest, const size_t countBlocks, bool enc) const;

    std::vector<float> testDefault(std::vector<kuznechikByteVector>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus);
    std::vector<float> testPinned(std::vector<kuznechikByteVector>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus);
    std::vector<float> testManaged(std::vector<kuznechikByteVector>& data, const size_t blokSize, const size_t gridSize, const bool encryptStatus);

    void setBlockSize(const size_t newBlockSize);
    void setGridSize(const size_t newGridSize);

    void checkEcnAndDec();
    double testSpeedUnequalBytes();
    void searchBestBlockAndGridSize();


};

#endif