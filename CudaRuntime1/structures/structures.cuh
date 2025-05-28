#pragma once
#ifndef H_STRUCTURES
#define H_STRUCTURES
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <span>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

union magmaHalfBlockT {
    uint8_t bytes[4];
    uint32_t uint;
};

using magmaKeyT = magmaHalfBlockT;

struct magmaKeySet {
    magmaKeyT keys[8];
};

struct magmaRoundKeySet {
    magmaKeyT keys[8];
};

union magmaBlockT {
    uint64_t ull;
    struct {
        magmaHalfBlockT lo, hi;
    };
    uint8_t bytes[sizeof(magmaHalfBlockT) * 2];
    //magmaBlockT() = default;
    //explicit magmaBlockT(uint8_t* data);
    //magmaBlockT(uint8_t);
};

inline std::ostream& operator << (std::ostream& s, const magmaBlockT& block) {
    for (int i = 0; i < sizeof(magmaBlockT); ++i) {
        uint8_t high, low;
        high = block.bytes[i] >> 4;
        low = block.bytes[i] & 0xf;
        s << (high < 10 ? ('0' + high) : ('A' + high));
        s << (low < 10 ? ('0' + low) : ('A' + low));
    }
    return s;
}

struct kuznechikKeys {
    uint8_t bytes[32];
    __device__ __host__ explicit kuznechikKeys(uint8_t* data) {
        std::copy_n(data, 32, bytes);
    };
    __device__ kuznechikKeys() = default;
};

union kuznechikHalfVector {
    uint8_t bytes[sizeof(uint64_t)];
    uint64_t halfVector;
    __device__ __host__ kuznechikHalfVector() = default;
    __device__ __host__ kuznechikHalfVector(const uint64_t src) : halfVector(src) {};
};

union kuznechikByteVector {
    struct halfs
    {
        kuznechikHalfVector lo, hi;
    } halfsData;
    uint64_t ull;
    uint8_t bytes[sizeof(uint64_t) * 2];
    __device__ __host__ kuznechikByteVector(const kuznechikHalfVector& lo, const kuznechikHalfVector& hi) : halfsData{ lo, hi } {};
    __device__ __host__ explicit kuznechikByteVector(uint8_t* src){
        //std::copy_n(src, 16, bytes);
        for (int i = 0; i < 16; ++i) {
            bytes[i] = src[i];
        }
    };
    __device__ __host__ kuznechikByteVector(uint8_t byte){
        for (size_t i = 0; i < 16; ++i)
        {
            this->bytes[i] = byte;
        }
    };
    __device__ __host__ kuznechikByteVector() = default;
};

inline std::ostream& operator << (std::ostream& s, const kuznechikByteVector& block) {
    for (int i = 0; i < sizeof(kuznechikByteVector); ++i) {
        uint8_t high, low;
        high = block.bytes[i] >> 4;
        low = block.bytes[i] & 0xf;
        s << (high < 10 ? ('0' + high) : ('A' + high));
        s << (low < 10 ? ('0' + low) : ('A' + low));
    }
    return s;
}

struct timeRes {
    std::string testName;
    std::string path;
    size_t size;
    bool encrypt;
    std::vector<float> time{ 0, 0 };
};

struct timeResStream {
    size_t size;
    double countStream;
    double tileSize;
    double blockSize;
    double gridSize;
    bool encrypt;
    double time;
};

// structures for CUDA
#define cudaCheck(e) cudaCheck__((e), __FILE__, __LINE__)

__inline__ void cudaCheck__(cudaError_t e, const char* file, int line) {
    if (e == cudaSuccess)
        return;
    std::cerr << " File: " << file << "( line " << line << " ): " << cudaGetErrorString(e) << std::endl;
    exit(e);
}

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
    std::remove_extent_t<T>* ptr;
    cudaCheck(cudaMalloc((void**)&ptr, sizeof(std::remove_extent_t<T>) * n));
    return cuda_ptr<T>{ptr};
}

template<class T>
cuda_ptr<T> cuda_alloc_async(const size_t n) {
    std::remove_extent_t<T>* ptr;
    cudaCheck(cudaMallocAsync((void**)&ptr, sizeof(std::remove_extent_t<T>) * n, 0));
    return cuda_ptr<T>{ptr};
}
#endif