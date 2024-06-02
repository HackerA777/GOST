#include "structures.cuh"
#include "magmaFunctions.cuh"
#include "magmaClass.cuh"
//#include "magmaFunctions.cu"

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>
#include <Windows.h>
#include <iostream>
#include <cooperative_groups.h>
#include <vector>
#include <random>
#include <chrono>


#define cudaCheck(e) cudaCheck__((e), __FILE__, __LINE__)

void cudaCheck__(cudaError_t e, const char* file, int line) {
    if (e == cudaSuccess)
        return;
    std::cerr << " File: " << file << "( line " << line << " ): " << cudaGetErrorString(e) << std::endl;
    exit(e);
}

__device__ const uint8_t table[8][16] = {
    {12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1},
    {6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15},
    {11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0},
    {12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11},
    {7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12},
    {5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0},
    {8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7},
    {1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2}
};

const key_set keys = {
        key_t {0xcc, 0xdd, 0xee, 0xff},
        key_t {0x88, 0x99, 0xaa, 0xbb},
        key_t {0x44, 0x55, 0x66, 0x77},
        key_t {0x00, 0x11, 0x22, 0x33},
        key_t {0xf3, 0xf2, 0xf1, 0xf0},
        key_t {0xf7, 0xf6, 0xf5, 0xf4},
        key_t {0xfb, 0xfa, 0xf9, 0xf8},
        key_t {0xff, 0xfe, 0xfd, 0xfc}
};

template<class T>
struct CudaDeleter {
    void operator()(std::remove_all_extents_t<T>* ptr) { cudaFree(ptr); };
};

template<class T>
using cuda_ptr = std::unique_ptr < T, CudaDeleter<T> >;

template<class T>
cuda_ptr<T> cuda_alloc() {
    T* ptr;
    cudaCheck(cudaMalloc((void**)&ptr, sizeof(T)));
    return cuda_ptr<T>{ptr};
}

template<class T>
cuda_ptr<T> cuda_alloc(const size_t n) {
    T* ptr;
    cudaCheck(cudaMalloc((void**)&ptr, sizeof(T) * n));
    return cuda_ptr<T>{ptr};
}

void encryptCuda(const uint8_t* block, uint8_t* out_block, const size_t dataSize, const key_set& keys, const unsigned int blockSize, const unsigned int gridSize) {
    cuda_ptr<key_set> dev_keys = cuda_alloc<key_set>();
    cuda_ptr<block_t> dev_block = cuda_alloc<block_t>(dataSize);
    cuda_ptr<block_t> dev_out_block = cuda_alloc<block_t>(dataSize);

    cudaError_t cudaStatus;

    cudaCheck(cudaMemcpy(dev_keys.get(), keys.keys, sizeof(key_set), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dev_block.get(), block, dataSize, cudaMemcpyHostToDevice));

    // cudaCheck(cudaGetLastError());

    Encrypt <<< blockSize, gridSize >> > (*dev_keys, *dev_block, *dev_out_block);

    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(out_block, dev_out_block.get(), dataSize, cudaMemcpyDeviceToHost));
}


void decryptCuda(const uint8_t* block, uint8_t* out_block, const size_t dataSize, const key_set& keys, const unsigned int blockSize, const unsigned int gridSize) {
    cuda_ptr<key_set> dev_keys = cuda_alloc<key_set>();
    cuda_ptr<block_t> dev_block = cuda_alloc<block_t>(dataSize);
    cuda_ptr<block_t> dev_out_block = cuda_alloc<block_t>(dataSize);

    cudaError_t cudaStatus;

    cudaCheck(cudaMemcpy(dev_keys.get(), keys.keys, sizeof(key_set), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dev_block.get(), block, dataSize, cudaMemcpyHostToDevice));

    // cudaCheck(cudaGetLastError());

    Decrypt <<< blockSize, gridSize >> > (*dev_keys, *dev_block, *dev_out_block);

    // cuddaGetLastError
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(out_block, dev_out_block.get(), dataSize, cudaMemcpyDeviceToHost));
}

void test() {
    constexpr size_t size = 1024 * 1024;
    std::vector<block_t> data(size / sizeof(block_t));
    uint32_t i = 0;
    for (auto& b : data) b.ull = ++i << 32 | ++i;

    using duration = std::chrono::duration<double, std::milli>;
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << data.data() << std::endl;
    encryptCuda((uint8_t*)data.data(), (uint8_t*)data.data(), size, keys, 1, 1);
    duration time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "SIZE: " << size << "\tTIME: " << time.count() << "ms\t SPEED: " << (size / 1024.0 / 1024) / time.count() * 1000 << " GB/s" << std::endl;
}


int main()
{
    // SetConsoleOutputCP(1251);
    // SetConsoleCP(1251);
    SetConsoleOutputCP(65001);
    block_t en_blk, enc_e_t;
    const unsigned char encrypt_test_string[8] = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    };
    memccpy(en_blk.bytes, encrypt_test_string, 8, 8);
    const unsigned char encrypt_exaple_text[8] = {
        0x3d, 0xca, 0xd8, 0xc2, 0xe5, 0x01, 0xe9, 0x4e
    };
    memccpy(enc_e_t.bytes, encrypt_exaple_text, 8, 8);

    block_t resultEncrypt{};

    // uint8_t* iter_keys[32];
    // ExpandKey(keys);
    std::cout << "encrypt_test_string: " << en_blk << std::endl;

    encryptCuda(encrypt_test_string, resultEncrypt.bytes, 8, keys, 1, 1);

    //printf("Encryption text: {%s}\n", result);
    std::cout << "Encryption text: " << resultEncrypt << std::endl;
    std::cout << "Encr examp text: " << enc_e_t << std::endl;

    block_t resultDecrypt{};

    decryptCuda(resultEncrypt.bytes, resultDecrypt.bytes, 8, keys, 1, 1);

    //printf("Encryption text: {%s}\n", result);
    std::cout << "Decryption text: " << resultDecrypt << std::endl;

    test();

    return 0;
}
