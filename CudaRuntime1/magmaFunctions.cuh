#include "structures.cuh"

half_block_t invert(const half_block_t block);

uint32_t Add(const half_block_t a, const half_block_t b);

uint32_t Add32(const half_block_t a, const half_block_t b);

half_block_t magma_T(const half_block_t data);

half_block_t magma_g(const key_t round_key, const half_block_t block);

block_t magma_G(const key_t round_key, const block_t block);

block_t magma_G_last(const key_t round_key, const block_t block);

__global__ void Encrypt(const key_set& round_key, const block_t& block, block_t& result_block);
__global__ void Decrypt(const key_set& round_key, const block_t& block, block_t& result_block);