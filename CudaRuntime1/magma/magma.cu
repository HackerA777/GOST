#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "magma.cuh"
#include <vector>
#include <random>
#include <chrono>

static __device__ constexpr uint8_t table[8][16] = {
    {12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1},
    {6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15},
    {11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0},
    {12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11},
    {7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12},
    {5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0},
    {8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7},
    {1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2}
};

static __device__ constexpr uint8_t table4x256[4][256] = {
{ 108, 100, 102, 98, 106, 101, 107, 105, 110, 104, 109, 103, 96, 99, 111, 97, 140, 132, 134, 130, 138, 133, 139, 137, 142, 136, 141, 135, 128, 131, 143, 129, 44, 36, 38, 34, 42, 37, 43, 41, 46, 40, 45, 39, 32, 35, 47, 33, 60, 52, 54, 50, 58, 53, 59, 57, 62, 56, 61, 55, 48, 51, 63, 49, 156, 148, 150, 146, 154, 149, 155, 153, 158, 152, 157, 151, 144, 147, 159, 145, 172, 164, 166, 162, 170, 165, 171, 169, 174, 168, 173, 167, 160, 163, 175, 161, 92, 84, 86, 82, 90, 85, 91, 89, 94, 88, 93, 87, 80, 83, 95, 81, 204, 196, 198, 194, 202, 197, 203, 201, 206, 200, 205, 199, 192, 195, 207, 193, 28, 20, 22, 18, 26, 21, 27, 25, 30, 24, 29, 23, 16, 19, 31, 17, 236, 228, 230, 226, 234, 229, 235, 233, 238, 232, 237, 231, 224, 227, 239, 225, 76, 68, 70, 66, 74, 69, 75, 73, 78, 72, 77, 71, 64, 67, 79, 65, 124, 116, 118, 114, 122, 117, 123, 121, 126, 120, 125, 119, 112, 115, 127, 113, 188, 180, 182, 178, 186, 181, 187, 185, 190, 184, 189, 183, 176, 179, 191, 177, 220, 212, 214, 210, 218, 213, 219, 217, 222, 216, 221, 215, 208, 211, 223, 209, 12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1, 252, 244, 246, 242, 250, 245, 251, 249, 254, 248, 253, 247, 240, 243, 255, 241 },
{ 203, 195, 197, 200, 194, 207, 202, 205, 206, 193, 199, 196, 204, 201, 198, 192, 139, 131, 133, 136, 130, 143, 138, 141, 142, 129, 135, 132, 140, 137, 134, 128, 43, 35, 37, 40, 34, 47, 42, 45, 46, 33, 39, 36, 44, 41, 38, 32, 27, 19, 21, 24, 18, 31, 26, 29, 30, 17, 23, 20, 28, 25, 22, 16, 219, 211, 213, 216, 210, 223, 218, 221, 222, 209, 215, 212, 220, 217, 214, 208, 75, 67, 69, 72, 66, 79, 74, 77, 78, 65, 71, 68, 76, 73, 70, 64, 251, 243, 245, 248, 242, 255, 250, 253, 254, 241, 247, 244, 252, 249, 246, 240, 107, 99, 101, 104, 98, 111, 106, 109, 110, 97, 103, 100, 108, 105, 102, 96, 123, 115, 117, 120, 114, 127, 122, 125, 126, 113, 119, 116, 124, 121, 118, 112, 11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0, 171, 163, 165, 168, 162, 175, 170, 173, 174, 161, 167, 164, 172, 169, 166, 160, 91, 83, 85, 88, 82, 95, 90, 93, 94, 81, 87, 84, 92, 89, 86, 80, 59, 51, 53, 56, 50, 63, 58, 61, 62, 49, 55, 52, 60, 57, 54, 48, 235, 227, 229, 232, 226, 239, 234, 237, 238, 225, 231, 228, 236, 233, 230, 224, 155, 147, 149, 152, 146, 159, 154, 157, 158, 145, 151, 148, 156, 153, 150, 144, 187, 179, 181, 184, 178, 191, 186, 189, 190, 177, 183, 180, 188, 185, 182, 176 },
{ 87, 95, 85, 90, 88, 81, 86, 93, 80, 89, 83, 94, 91, 84, 82, 92, 215, 223, 213, 218, 216, 209, 214, 221, 208, 217, 211, 222, 219, 212, 210, 220, 247, 255, 245, 250, 248, 241, 246, 253, 240, 249, 243, 254, 251, 244, 242, 252, 103, 111, 101, 106, 104, 97, 102, 109, 96, 105, 99, 110, 107, 100, 98, 108, 151, 159, 149, 154, 152, 145, 150, 157, 144, 153, 147, 158, 155, 148, 146, 156, 39, 47, 37, 42, 40, 33, 38, 45, 32, 41, 35, 46, 43, 36, 34, 44, 199, 207, 197, 202, 200, 193, 198, 205, 192, 201, 195, 206, 203, 196, 194, 204, 167, 175, 165, 170, 168, 161, 166, 173, 160, 169, 163, 174, 171, 164, 162, 172, 183, 191, 181, 186, 184, 177, 182, 189, 176, 185, 179, 190, 187, 180, 178, 188, 119, 127, 117, 122, 120, 113, 118, 125, 112, 121, 115, 126, 123, 116, 114, 124, 135, 143, 133, 138, 136, 129, 134, 141, 128, 137, 131, 142, 139, 132, 130, 140, 23, 31, 21, 26, 24, 17, 22, 29, 16, 25, 19, 30, 27, 20, 18, 28, 71, 79, 69, 74, 72, 65, 70, 77, 64, 73, 67, 78, 75, 68, 66, 76, 55, 63, 53, 58, 56, 49, 54, 61, 48, 57, 51, 62, 59, 52, 50, 60, 231, 239, 229, 234, 232, 225, 230, 237, 224, 233, 227, 238, 235, 228, 226, 236, 7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12 },
{ 24, 30, 18, 21, 22, 25, 17, 28, 31, 20, 27, 16, 29, 26, 19, 23, 120, 126, 114, 117, 118, 121, 113, 124, 127, 116, 123, 112, 125, 122, 115, 119, 232, 238, 226, 229, 230, 233, 225, 236, 239, 228, 235, 224, 237, 234, 227, 231, 216, 222, 210, 213, 214, 217, 209, 220, 223, 212, 219, 208, 221, 218, 211, 215, 8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7, 88, 94, 82, 85, 86, 89, 81, 92, 95, 84, 91, 80, 93, 90, 83, 87, 136, 142, 130, 133, 134, 137, 129, 140, 143, 132, 139, 128, 141, 138, 131, 135, 56, 62, 50, 53, 54, 57, 49, 60, 63, 52, 59, 48, 61, 58, 51, 55, 72, 78, 66, 69, 70, 73, 65, 76, 79, 68, 75, 64, 77, 74, 67, 71, 248, 254, 242, 245, 246, 249, 241, 252, 255, 244, 251, 240, 253, 250, 243, 247, 168, 174, 162, 165, 166, 169, 161, 172, 175, 164, 171, 160, 173, 170, 163, 167, 104, 110, 98, 101, 102, 105, 97, 108, 111, 100, 107, 96, 109, 106, 99, 103, 152, 158, 146, 149, 150, 153, 145, 156, 159, 148, 155, 144, 157, 154, 147, 151, 200, 206, 194, 197, 198, 201, 193, 204, 207, 196, 203, 192, 205, 202, 195, 199, 184, 190, 178, 181, 182, 185, 177, 188, 191, 180, 187, 176, 189, 186, 179, 183, 40, 46, 34, 37, 38, 41, 33, 44, 47, 36, 43, 32, 45, 42, 35, 39 },
};

struct Tables {
    uint8_t s_table4x256[4][256];
};

static __device__ magmaHalfBlockT invert(const magmaHalfBlockT block) {
    magmaHalfBlockT result;
    for (int i = 0; i < 4; ++i) {
        result.bytes[i] = block.bytes[3 - i];
    }
    return result;
}

static __device__ uint32_t Add(const magmaHalfBlockT a, const magmaHalfBlockT b) {
    return a.uint ^ b.uint;
}

static __device__ uint32_t Add32(const magmaHalfBlockT a, const magmaHalfBlockT b) {
    return a.uint + b.uint;
}

static __device__ magmaHalfBlockT magma_T(const magmaHalfBlockT data, const Tables& t) {
    magmaHalfBlockT result;
    for (int i = 0; i < 4; ++i) {
        result.bytes[i] = t.s_table4x256[i][data.bytes[i]];
    }
    return result;
}

static __device__ magmaHalfBlockT magma_g(const magmaKeyT round_key, const magmaHalfBlockT block, const Tables& t) {
    magmaHalfBlockT internal = block;
    internal.uint = Add32(block, round_key);
    internal = magma_T(internal, t);
    internal.uint = (internal.uint << 11) | (internal.uint >> 21);

    return internal;
}

static __device__ magmaBlockT magma_G(const magmaKeyT round_key, const magmaBlockT block, const Tables& t) {

    magmaBlockT result = block;
    result.lo = magma_g(round_key, result.lo, t);
    result.lo.uint = Add(result.lo, block.hi);

    result.hi = block.lo;
    return result;
}

static __device__ magmaBlockT magma_G_last(const magmaKeyT round_key, const magmaBlockT block, const Tables& t)
{
    magmaBlockT result = block;
    result.lo = magma_g(round_key, result.lo,t);
    result.lo.uint = Add(result.lo, block.hi);

    result.hi = result.lo;
    result.lo = block.lo;

    return result;
}

static __global__ void encryptMgm(const magmaKeySet& round_key, magmaBlockT* blocks, const size_t count)
{
    __shared__ magmaKeySet s_keys;

    for (auto j = threadIdx.x; j < 32; j += blockDim.x)
        s_keys.keys[j] = round_key.keys[j];
    
    __shared__ Tables t;
    
    for (auto k = threadIdx.x;k < 4*256; k += blockDim.x){
        int i = k / 256;
        int j = k % 256;
        t.s_table4x256[i][j] = table4x256[i][j];
        
    }
    __syncthreads();

    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    tid = blockDim.x * blockIdx.x + threadIdx.x;
    auto tcnt = gridDim.x * blockDim.x;

    for (auto i = tid; i < count; i+=tcnt) {

        magmaBlockT tempBlock = blocks[i];
        // Первое преобразование G
        tempBlock = magma_G(s_keys.keys[0], tempBlock, t);
        // Последующие (со второго по тридцать первое) преобразования G
        for (int j = 1; j < 24; ++j)
            tempBlock = magma_G(s_keys.keys[j % 8], tempBlock, t);
        for (int j = 0; j < 7; ++j)
            tempBlock = magma_G(s_keys.keys[7 - j % 8], tempBlock, t);
        // Последнее (тридцать второе) преобразование G
        blocks[i] = magma_G_last(s_keys.keys[0], tempBlock, t);
    }
}

static __global__ void decryptMgm(const magmaKeySet& round_key, magmaBlockT* blocks, const size_t count)
{
    __shared__ magmaKeySet s_keys;

    for (auto j = threadIdx.x; j < 32; j += blockDim.x)
        s_keys.keys[j] = round_key.keys[j];

    __shared__ Tables t;

    for (auto k = threadIdx.x; k < 4 * 256; k += blockDim.x) {
        int i = k / 256;
        int j = k % 256;
        t.s_table4x256[i][j] = table4x256[i][j];

    }
    __syncthreads();

    auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    auto tcnt = gridDim.x * blockDim.x;
    for (auto i = tid; i < count; i += tcnt) {
        magmaBlockT tempBlock = blocks[i];
        // Первое преобразование G
        tempBlock = magma_G(s_keys.keys[0], tempBlock, t);
        // Последующие (со второго по тридцать первое) преобразования G
        for (int j = 1; j < 8; ++j)
            tempBlock = magma_G(s_keys.keys[j % 8], tempBlock, t);
        for (int j = 0; j < 23; ++j)
            tempBlock = magma_G(s_keys.keys[7 - j % 8], tempBlock, t);
        // Последнее (тридцать второе) преобразование G
        blocks[i] = magma_G_last(s_keys.keys[0], tempBlock, t);
    }
}

void magma::encryptCuda(const uint8_t* blocks, uint8_t* out_blocks, const size_t dataSize) {
    const size_t countBlocks = dataSize / 8;

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();
    cuda_ptr<magmaBlockT[]> dev_blocks = cuda_alloc_async<magmaBlockT[]>(countBlocks);

    cudaError_t cudaStatus;

    cudaCheck(cudaMemcpyAsync(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));
    cudaCheck(cudaHostRegister((void*)blocks, dataSize, cudaHostRegisterDefault));


    cudaCheck(cudaMemcpyAsync(dev_blocks.get(), blocks, dataSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaGetLastError());

    encryptMgm <<< gridSize, blockSize >>> (*dev_keys, dev_blocks.get(), countBlocks);

    cudaCheck(cudaGetLastError());
    if(blocks != out_blocks)
        cudaCheck(cudaHostRegister((void*)out_blocks, countBlocks * sizeof(magmaBlockT), cudaHostRegisterDefault));

    cudaCheck(cudaMemcpyAsync(out_blocks, dev_blocks.get(), countBlocks * sizeof(magmaBlockT), cudaMemcpyDeviceToHost));

    if (blocks != out_blocks)
        cudaCheck(cudaHostUnregister((void*)blocks));

    cudaCheck(cudaStreamSynchronize(0));
    
    cudaCheck(cudaHostUnregister((void*)out_blocks));
}

void magma::decryptCuda(const uint8_t* blocks, uint8_t* out_blocks, const size_t dataSize) {
    const size_t countBlocks = dataSize / 8;

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();
    cuda_ptr<magmaBlockT[]> dev_blocks = cuda_alloc<magmaBlockT[]>(countBlocks);

    cudaError_t cudaStatus;

    //cudaCheck(cudaMemcpy(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));

    //cudaCheck(cudaMemcpy(dev_blocks.get(), blocks, countBlocks * sizeof(magmaBlockT), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpyAsync(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));
    cudaCheck(cudaHostRegister((void*)blocks, countBlocks * sizeof(magmaBlockT), cudaHostRegisterDefault));

    cudaCheck(cudaMemcpyAsync(dev_blocks.get(), blocks, countBlocks * sizeof(magmaBlockT), cudaMemcpyHostToDevice));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaGetLastError());

    decryptMgm <<< gridSize, blockSize >> > (*dev_keys, dev_blocks.get(), countBlocks);

    cudaCheck(cudaGetLastError());
    if (blocks != out_blocks)
        cudaCheck(cudaHostRegister((void*)out_blocks, countBlocks * sizeof(magmaBlockT), cudaHostRegisterDefault));

    cudaCheck(cudaMemcpyAsync(out_blocks, dev_blocks.get(), countBlocks * sizeof(magmaBlockT), cudaMemcpyDeviceToHost));

    if (blocks != out_blocks)
        cudaCheck(cudaHostUnregister((void*)blocks));

    cudaCheck(cudaStreamSynchronize(0));


    cudaCheck(cudaHostUnregister((void*)out_blocks));
}

void magma::checkEcnAndDec() {
    cudaError_t cudaStatus;
    const unsigned char testString[8] = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    };
    const unsigned char encryptValidString[8] = {
        0x3d, 0xca, 0xd8, 0xc2, 0xe5, 0x01, 0xe9, 0x4e
    };
    const unsigned char keys[32] = {
        0xcc, 0xdd, 0xee, 0xff,
        0x88, 0x99, 0xaa, 0xbb,
        0x44, 0x55, 0x66, 0x77,
        0x00, 0x11, 0x22, 0x33,
        0xf3, 0xf2, 0xf1, 0xf0,
        0xf7, 0xf6, 0xf5, 0xf4,
        0xfb, 0xfa, 0xf9, 0xf8,
        0xff, 0xfe, 0xfd, 0xfc
    };

    magmaBlockT testBlock, resultBlock, validBlock; 
    magmaKeySet testKeys;

    magmaBlockT* testBlockPtr, *testResultPtr;
    magmaKeySet* testKeysPtr;

    cudaCheck(cudaMallocManaged(&testBlockPtr, sizeof(magmaBlockT)));
    cudaCheck(cudaMallocManaged(&testResultPtr, sizeof(magmaBlockT)));
    cudaCheck(cudaMallocManaged(&testKeysPtr, sizeof(magmaKeySet)));

    std::copy(testString, testString + 8, testBlock.bytes);
    std::copy(testString, testString + 8, resultBlock.bytes);

    std::copy(testString, testString + 8, testBlockPtr->bytes);
    std::copy(keys, keys + 32, testKeysPtr->keys->bytes);

    std::copy(encryptValidString, encryptValidString + 8, validBlock.bytes);
    std::copy(keys, keys + 32, testKeys.keys->bytes);

    std::cout << "Before Test block: " << *testBlockPtr << std::endl;

    //encryptDefault(testBlock.bytes, resultBlock.bytes, 1);
    cudaCheck(cudaGetLastError());
    encryptMgm <<< this->blockSize, this->gridSize >>> (*testKeysPtr, testBlockPtr, 1);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaDeviceSynchronize());

    std::cout << "Test block: " << *testBlockPtr << std::endl;

    std::copy(testBlockPtr->bytes, testBlockPtr->bytes + 8, testResultPtr->bytes);

    std::cout << "result block: " << *testResultPtr << std::endl;
    //std::cout << "Result encryption test string: " << resultBlock << std::endl;
    std::cout << "Valide encryption string: " << validBlock << std::endl;
    //if (validBlock.ull == resultBlock.ull)
    //    std::cout << "Encryption algoritm valid!" << std::endl;
    //else
    //    std::cout << "Encryption algoritm unvalid!" << std::endl;

    if (validBlock.ull == testBlockPtr->ull)
        std::cout << "Encryption algoritm valid!" << std::endl;
    else
        std::cout << "Encryption algoritm unvalid!" << std::endl;

    //decryptCuda(resultBlock.bytes, resultBlock.bytes, 1);
    decryptMgm <<< this->blockSize, this->gridSize >>> (*testKeysPtr, testResultPtr, 1);

    cudaCheck(cudaDeviceSynchronize());

    std::cout << "Result decryption test string: " << resultBlock << std::endl;
    if (testBlock.ull == testResultPtr->ull)
        std::cout << "Decryption algoritm valid!" << std::endl;
    else
        std::cout << "Decryption algoritm unvalid!" << std::endl;

    //if (testBlock.ull == resultBlock.ull)
    //    std::cout << "Decryption algoritm valid!" << std::endl;
    //else
    //    std::cout << "Decryption algoritm unvalid!" << std::endl;

    cudaCheck(cudaFree(testBlockPtr));
    cudaCheck(cudaFree(testResultPtr));
    cudaCheck(cudaFree(testKeysPtr));
}

double magma::testSpeedUnequalBytes() {
    size_t size = buffSize;
    std::vector<magmaBlockT> data(size);
    uint32_t i = 0;
    for (auto& b : data) b.ull = ++i << 32 | ++i;
    encryptCuda((uint8_t*)data.data(), (uint8_t*)data.data(), 16);
    using duration = std::chrono::duration<double, std::milli>;
    auto start = std::chrono::high_resolution_clock::now();

    encryptCuda((uint8_t*)data.data(), (uint8_t*)data.data(), data.size());
    duration time = std::chrono::high_resolution_clock::now() - start;
    double speed = (size * sizeof(magmaBlockT) / 1024.0 / 1024 / 1024) / time.count() * 1000;
    std::cout << "SIZE: " << size << "\tTIME: " << time.count() << "ms\t SPEED: " << speed  << " GB/s" << " BLOCK SIZE: " << blockSize << " GRID SIZE: " << gridSize << std::endl;
    return speed;
}

void magma::searchBestBlockAndGridSize() {
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

void magma::setGridSize(const size_t newGridSize) {
    gridSize = newGridSize;
}

void magma::setBlockSize(const size_t newBlockSize) {
    blockSize = newBlockSize;
}

magma::magma(const unsigned char keys[32], const size_t buffSize, const unsigned int gridSize, const unsigned int blockSize ) {
    copyKeys(keys);
    this->buffSize = buffSize;
    this->blockSize = blockSize;
    this->gridSize = gridSize;
}

std::vector<float> magma::testDefault(std::vector<magmaBlockT>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
    cudaCheck(cudaGetLastError());
    std::vector<float> time{ 0, 0 };

    const size_t countBlocks = data.size();
    const size_t dataSize = countBlocks * sizeof(magmaBlockT);

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();
    cuda_ptr<magmaBlockT[]> dev_blocks = cuda_alloc<magmaBlockT[]>(countBlocks);

    cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
    cudaCheck(cudaEventCreate(&startEnc));
    cudaCheck(cudaEventCreate(&stopEnc));
    cudaCheck(cudaEventCreate(&startCopyAndEnc));
    cudaCheck(cudaEventCreate(&stopCopyAndEnc));

    cudaCheck(cudaEventRecord(startCopyAndEnc));

    cudaCheck(cudaGetLastError());
        
    cudaCheck(cudaMemcpyAsync(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(startEnc));

    cudaCheck(cudaGetLastError());

    if (encryptStatus) {
        encryptMgm << < gridSize, blockSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
    }
    else {
        decryptMgm << < gridSize, blockSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
    }

    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(stopEnc));

    cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));

    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(stopCopyAndEnc));
        
    cudaCheck(cudaGetLastError());

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

std::vector<float> magma::testPinned(std::vector<magmaBlockT>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
    std::vector<float> time{ 0, 0 };

    const size_t countBlocks = data.size();
    const size_t dataSize = countBlocks * 8;

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();
    cuda_ptr<magmaBlockT[]> dev_blocks = cuda_alloc<magmaBlockT[]>(countBlocks);

    cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
    cudaCheck(cudaEventCreate(&startEnc));
    cudaCheck(cudaEventCreate(&stopEnc));
    cudaCheck(cudaEventCreate(&startCopyAndEnc));
    cudaCheck(cudaEventCreate(&stopCopyAndEnc)); 
        
    cudaCheck(cudaEventRecord(startCopyAndEnc));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpyAsync(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaHostRegister((void*)data.data(), dataSize, cudaHostRegisterDefault));

    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(startEnc));

    if (encryptStatus) {
        encryptMgm << < gridSize, blockSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
    }
    else {
        decryptMgm << < gridSize, blockSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
    }

    cudaCheck(cudaGetLastError());

    cudaCheck(cudaEventRecord(stopEnc));

    cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));

    cudaCheck(cudaStreamSynchronize(0));

    cudaCheck(cudaHostUnregister((void*)data.data()));

    cudaCheck(cudaEventRecord(stopCopyAndEnc));

    cudaCheck(cudaEventSynchronize(stopCopyAndEnc));
    cudaEventElapsedTime(&time[0], startCopyAndEnc, stopCopyAndEnc);
    cudaEventElapsedTime(&time[1], startEnc, stopEnc);

    cudaCheck(cudaEventDestroy(startCopyAndEnc));
    cudaCheck(cudaEventDestroy(stopCopyAndEnc));
    cudaCheck(cudaEventDestroy(startEnc));
    cudaCheck(cudaEventDestroy(stopEnc));

    return time;
}

std::vector<float> magma::testManaged(std::vector<magmaBlockT>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
    std::vector<float> time { 0, 0 };
    magmaBlockT* buffer;

    const size_t countBlocks = data.size();

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();

    size_t dataSize = countBlocks * sizeof(magmaBlockT);

    cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
    cudaCheck(cudaEventCreate(&startEnc));
    cudaCheck(cudaEventCreate(&stopEnc));
    cudaCheck(cudaEventCreate(&startCopyAndEnc));
    cudaCheck(cudaEventCreate(&stopCopyAndEnc));

    cudaCheck(cudaMemcpyAsync(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));
    
    cudaCheck(cudaMallocManaged(&buffer, dataSize));

    cudaCheck(cudaEventRecord(startCopyAndEnc));
    cudaCheck(cudaMemcpyAsync(buffer, data.data(), dataSize, cudaMemcpyHostToHost));

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(startEnc));

    if (encryptStatus) {
        encryptMgm <<< gridSize, blockSize >>> (*dev_keys, buffer, countBlocks);
    }
    else {
        decryptMgm <<< gridSize, blockSize >>> (*dev_keys, buffer, countBlocks);
    }

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventRecord(stopEnc));

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

double magma::testStreams(std::vector<magmaBlockT>& data, const size_t blockSize, const size_t gridSize, const size_t countStreams, const size_t blocksPerStream, const bool encryptStatus) {

    std::vector<cudaStream_t> streams;
    streams.resize(countStreams);
    std::vector<cudaEvent_t> startEvents;
    startEvents.resize(countStreams);
    std::vector<cudaEvent_t> stopEvents;
    stopEvents.resize(countStreams);
    std::vector<bool> flags;
    flags.resize(countStreams);
    for (int i = 0; i < countStreams; ++i) {
        flags[i] = false;
    }

    const size_t countBlocks = data.size();
    const size_t dataSize = countBlocks * 8;

    size_t bufferSize = dataSize;

    if (countBlocks > blocksPerStream) {
        bufferSize = blocksPerStream * 8;
    }

    const size_t newCountBlocks = bufferSize / 8;

    cuda_ptr<magmaKeySet> dev_keys = cuda_alloc<magmaKeySet>();
    cuda_ptr<magmaBlockT[]> dev_blocks = cuda_alloc<magmaBlockT[]>(newCountBlocks * countStreams);

    cudaCheck(cudaGetLastError());
    
    cudaCheck(cudaMemcpy(dev_keys.get(), this->keys.keys, sizeof(magmaKeySet), cudaMemcpyHostToDevice));
    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaHostRegister((void*)data.data(), dataSize, cudaHostRegisterDefault));
    
    cudaCheck(cudaGetLastError());
    
    for (int i = 0; i < countStreams; ++i) {
        cudaCheck(cudaStreamCreate(&streams[i]));
        cudaCheck(cudaEventCreate(&startEvents[i]));
        cudaCheck(cudaEventCreate(&stopEvents[i]));
    }

    int streamId = 0;
    for (int i = 0; i < countBlocks; i += blocksPerStream) {

        if (!flags[streamId]) {
            cudaCheck(cudaEventRecord(startEvents[streamId], streams[streamId]));
            flags[streamId] = true;
        }

        cudaCheck(cudaMemcpyAsync(dev_blocks.get() + blocksPerStream * streamId, data.data() + blocksPerStream * (i / blocksPerStream), bufferSize, cudaMemcpyHostToDevice, streams[streamId]));
        cudaCheck(cudaGetLastError());
    
        if (encryptStatus) {
            encryptMgm <<< gridSize, blockSize, 0, streams[streamId] >> > (*dev_keys, dev_blocks.get() + newCountBlocks * streamId, blocksPerStream);
        }
        else {
            decryptMgm <<< gridSize, blockSize, 0, streams[streamId] >> > (*dev_keys, dev_blocks.get() + newCountBlocks * streamId, blocksPerStream);
        }
    
        cudaCheck(cudaGetLastError());
    
        cudaCheck(cudaMemcpyAsync(data.data() + blocksPerStream * (i / blocksPerStream), dev_blocks.get() + blocksPerStream * streamId, bufferSize, cudaMemcpyDeviceToHost, streams[streamId]));

        streamId = (streamId + 1) % countStreams;
    }
    
    for (int i = 0; i < countStreams; ++i) {
        cudaCheck(cudaEventRecord(stopEvents[i], streams[i]));
    }
    
    for (int i = 0; i < countStreams; ++i) {
        cudaCheck(cudaEventSynchronize(stopEvents[i]));
    }
    
    cudaCheck(cudaHostUnregister((void*)data.data()));
    
    float maxTime = 0;
    float currentTime = 0;
    for (int i = 0; i < countStreams; ++i) {
        for (int j = 0; j < countStreams; ++j) {
            if (flags[i] && flags[j])
            cudaCheck(cudaEventElapsedTime(&currentTime, startEvents[i], stopEvents[j]));
            if (currentTime > maxTime) {
                maxTime = currentTime;
            }
        }
    }
    
    for (int i = 0; i < countStreams; ++i) {
        cudaCheck(cudaStreamDestroy(streams[i]));
        cudaCheck(cudaEventDestroy(startEvents[i]));
        cudaCheck(cudaEventDestroy(stopEvents[i]));
    }

    return maxTime;
}
