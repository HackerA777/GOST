#ifndef H_STRUCTURES
#define H_STRUCTURES

#include <iostream>

union half_block_t {
    uint8_t bytes[4];
    uint32_t uint;
};

using key_t = half_block_t;

struct key_set {
    key_t keys[8];
    //uint8_t bytes[sizeof(half_block_t) * 8];
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

std::ostream& operator << (std::ostream& s, const block_t& block) {
    for (int i = 0; i < sizeof(block_t); ++i) {
        uint8_t high, low;
        high = block.bytes[i] >> 4;
        low = block.bytes[i] & 0xf;
        s << (high < 10 ? ('0' + high) : ('A' + high));
        s << (low < 10 ? ('0' + low) : ('A' + low));
    }
    return s;
}

#endif