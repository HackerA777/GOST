#include "./magma/magma.cuh"
#include "./kuznechik/kuznechik.cuh"
#include "./testSpeed/testSpeed.h"
#include "./tests/tests.cuh"
#include <Windows.h>
#include <fstream>

template<typename typeVector>
std::string printVector(const typeVector& block) {
    std::string result;
    for (int i = sizeof(typeVector) - 1; i >= 0; --i) {
        uint8_t high, low;
        high = block.bytes[i] >> 4;
        low = block.bytes[i] & 0xf;
        result.push_back((char)(high < 10 ? ('0' + high) : ('A' + high - 10)));
        result.push_back((char)(low < 10 ? ('0' + low) : ('A' + low - 10)));
    }
    return result;
}

int main()
{
    SetConsoleOutputCP(65001);

    std::vector<std::vector<double>> results;
    std::cout << "Results for algorithm 'Magma': " << std::endl;

    std::vector<size_t> blockSizeVector = { 32, 64, 128, 192, 256, 384, 512, 768, 1024 };
    size_t bestBlockSizeDefault, bestBlockSizePinned, bestBlockSizeManaged, bestBlockSizeOneStream, bestBlockSizeFourStreams;
    size_t bestSpeedDefault = 0, bestSpeedPinned = 0, bestSpeedManaged = 0, bestSpeedOneStream = 0, bestSpeedFourStreams = 0;

    for (size_t blockSize : blockSizeVector) {
        results = testSpeedMagma(".\\", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, blockSize, 16 * 1024);
        if (results[0][0] > bestSpeedDefault) {
            bestSpeedDefault = results[0][0];
            bestBlockSizeDefault = blockSize;
        }
        if (results[0][1] > bestSpeedPinned) {
            bestSpeedPinned = results[0][1];
            bestBlockSizePinned = blockSize;
        }
        if (results[0][2] > bestSpeedManaged) {
            bestSpeedManaged = results[0][2];
            bestBlockSizeManaged = blockSize;
        }
        if (results[0][3] > bestSpeedOneStream) {
            bestSpeedOneStream = results[0][3];
            bestBlockSizeOneStream = blockSize;
        }
        if (results[0][4] > bestSpeedFourStreams) {
            bestSpeedFourStreams = results[0][4];
            bestBlockSizeFourStreams = blockSize;
        }
    }
    
    std::cout << "Encryption speed for mode copy 'Default': " << results[0][0] << " Gb/s\nEncryption speed with copy: " << results[1][0] << " Gb/s" << "; Best block size: " << bestBlockSizeDefault << std::endl;
    std::cout << "Encryption speed for mode copy 'Pinned': " << results[0][1] << " Gb/s\nEncryption speed with copy: " << results[1][1] << " Gb/s" << "; Best block size: " << bestBlockSizePinned << std::endl;
    std::cout << "Encryption speed for mode copy 'Managed': " << results[0][2] << " Gb/s\nEncryption speed with copy: " << results[1][2] << " Gb/s" << "; Best block size: " << bestBlockSizeManaged << std::endl;
    std::cout << "Encryption speed with use one stream command: " << results[0][3] << " Gb/s" << "; Best block size: " << bestBlockSizeOneStream << std::endl;
    std::cout << "Encryption speed with use four streams command: " << results[0][4] << " Gb/s" << "; Best block size: " << bestBlockSizeFourStreams << std::endl;

    bestSpeedDefault = bestSpeedPinned = bestSpeedManaged = bestSpeedOneStream = bestSpeedFourStreams = 0;
    bestBlockSizeDefault = bestBlockSizePinned = bestBlockSizeManaged = bestBlockSizeOneStream = bestBlockSizeFourStreams = 0;

    std::cout << "\nResults for algorithm 'Kuznechik' release - 1: " << std::endl;
    for (size_t blockSize : blockSizeVector) {
        results = testSpeedKuznechik(".\\", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, blockSize, 16 * 1024, 1);
        if (results[0][0] > bestSpeedDefault) {
            bestSpeedDefault = results[0][0];
            bestBlockSizeDefault = blockSize;
        }
        if (results[0][1] > bestSpeedPinned) {
            bestSpeedPinned = results[0][1];
            bestBlockSizePinned = blockSize;
        }
        if (results[0][2] > bestSpeedManaged) {
            bestSpeedManaged = results[0][2];
            bestBlockSizeManaged = blockSize;
        }
        if (results[0][3] > bestSpeedOneStream) {
            bestSpeedOneStream = results[0][3];
            bestBlockSizeOneStream = blockSize;
        }
        if (results[0][4] > bestSpeedFourStreams) {
            bestSpeedFourStreams = results[0][4];
            bestBlockSizeFourStreams = blockSize;
        }
    }
    
    std::cout << "Encryption speed for mode copy 'Default': " << results[0][0] << " Gb/s\nEncryption speed with copy: " << results[1][0] << " Gb/s" << "; Best block size: " << bestBlockSizeDefault << std::endl;
    std::cout << "Encryption speed for mode copy 'Pinned': " << results[0][1] << " Gb/s\nEncryption speed with copy: " << results[1][1] << " Gb/s" << "; Best block size: " << bestBlockSizePinned << std::endl;
    std::cout << "Encryption speed for mode copy 'Managed': " << results[0][2] << " Gb/s\nEncryption speed with copy: " << results[1][2] << " Gb/s" << "; Best block size: " << bestBlockSizeManaged << std::endl;
    std::cout << "Encryption speed with use one stream command: " << results[0][3] << " Gb/s" << "; Best block size: " << bestBlockSizeOneStream << std::endl;
    std::cout << "Encryption speed with use four streams command: " << results[0][4] << " Gb/s" << "; Best block size: " << bestBlockSizeFourStreams << std::endl;
    
    bestSpeedDefault = bestSpeedPinned = bestSpeedManaged = bestSpeedOneStream = bestSpeedFourStreams = 0;
    bestBlockSizeDefault = bestBlockSizePinned = bestBlockSizeManaged = bestBlockSizeOneStream = bestBlockSizeFourStreams = 0;

    std::cout << "\nResults for algorithm 'Kuznechik' release - 2: " << std::endl;
    for (size_t blockSize : blockSizeVector) {
        results = testSpeedKuznechik(".\\", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, blockSize, 16 * 1024, 2);
        if (results[0][0] > bestSpeedDefault) {
            bestSpeedDefault = results[0][0];
            bestBlockSizeDefault = blockSize;
        }
        if (results[0][1] > bestSpeedPinned) {
            bestSpeedPinned = results[0][1];
            bestBlockSizePinned = blockSize;
        }
        if (results[0][2] > bestSpeedManaged) {
            bestSpeedManaged = results[0][2];
            bestBlockSizeManaged = blockSize;
        }
        if (results[0][3] > bestSpeedOneStream) {
            bestSpeedOneStream = results[0][3];
            bestBlockSizeOneStream = blockSize;
        }
        if (results[0][4] > bestSpeedFourStreams) {
            bestSpeedFourStreams = results[0][4];
            bestBlockSizeFourStreams = blockSize;
        }
    }

    std::cout << "Encryption speed for mode copy 'Default': " << results[0][0] << " Gb/s\nEncryption speed with copy: " << results[1][0] << " Gb/s" << "; Best block size: " << bestBlockSizeDefault << std::endl;
    std::cout << "Encryption speed for mode copy 'Pinned': " << results[0][1] << " Gb/s\nEncryption speed with copy: " << results[1][1] << " Gb/s" << "; Best block size: " << bestBlockSizePinned << std::endl;
    std::cout << "Encryption speed for mode copy 'Managed': " << results[0][2] << " Gb/s\nEncryption speed with copy: " << results[1][2] << " Gb/s" << "; Best block size: " << bestBlockSizeManaged << std::endl;
    std::cout << "Encryption speed with use one stream command: " << results[0][3] << " Gb/s" << "; Best block size: " << bestBlockSizeOneStream << std::endl;
    std::cout << "Encryption speed with use four streams command: " << results[0][4] << " Gb/s" << "; Best block size: " << bestBlockSizeFourStreams << std::endl;

    return 0;
}