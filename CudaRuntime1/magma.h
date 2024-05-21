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

#include "cuda_runtime.h"

class magma {
private:
    union half_block_t {
        uint8_t bytes[4];
        uint32_t uint;
    };

    using key_t = half_block_t;

    struct key_set {
        key_t keys[8];
    };

    struct round_key_set {
        key_t keys[8];
    };

    union block_t {
        uint64_t ull;
        struct {
            half_block_t lo, hi;
        };
        uint8_t bytes[sizeof(half_block_t) * 2];
    };

    cudaError_t cuda_error;

    // functions
    half_block_t invert(const half_block_t block);
    uint32_t Add(const half_block_t a, const half_block_t b);
    uint32_t Add32(const half_block_t a, const half_block_t b);
    half_block_t magma_T(const half_block_t data);
    half_block_t magma_g(const key_t round_key, const half_block_t block);
    block_t magma_G(const key_t round_key, const block_t block);
    block_t magma_G_last(const key_t round_key, const block_t block);
    void Encrypt(const key_set& round_key, const block_t& block, block_t& result_block);
    void Decrypt(const key_set& round_key, const block_t& block, block_t& result_block);

public:
    
    friend std::ostream& operator << (std::ostream& s, const block_t& block) {
        for (int i = 0; i < sizeof(block_t); ++i) {
            uint8_t high, low;
            high = block.bytes[i] >> 4;
            low = block.bytes[i] & 0xf;
            s << (high < 10 ? ('0' + high) : ('A' + high));
            s << (low < 10 ? ('0' + low) : ('A' + low));
        }
        return s;
    }

    void cudaCheck__(cudaError_t e, const char* file, int line) {
        if (e == cudaSuccess)
            return;
        std::cerr << " File: " << file << "( line " << line << " ): " << cudaGetErrorString(e) << std::endl;
        exit(e);
    }

    magma();
    
};