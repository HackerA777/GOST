
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <cstring>
#include <string.h>
#include <Windows.h>
#include <iostream>
#include <cooperative_groups.h>



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

void cudaCheck__(cudaError_t e, const char* file, int line) {
    if (e == cudaSuccess)
        return;
    std::cerr << " File: " << file << "( line " << line << " ): " << cudaGetErrorString(e) << std::endl;
    exit(e);
}

#define cudaCheck(e) cudaCheck__((e), __FILE__, __LINE__)

__device__ const unsigned char table[8][16] = {
    {12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1},
    {6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15},
    {11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0},
    {12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11},
    {7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12},
    {5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0},
    {8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7},
    {1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2}
};


__device__ uint32_t Add(const half_block_t a, const half_block_t b) {
    return a.uint ^ b.uint;
}

__device__ uint32_t Add32(const half_block_t a, const half_block_t b) {
    return a.uint + b.uint;
}

//__host__ void ExpandKey(const uint8_t* key)
//{
//    memcpy(iter_keys[0], key, 4);
//    memcpy(iter_keys[1], key + 4, 4);
//    memcpy(iter_keys[2], key + 8, 4);
//    memcpy(iter_keys[3], key + 12, 4);
//    memcpy(iter_keys[4], key + 16, 4);
//    memcpy(iter_keys[5], key + 20, 4);
//    memcpy(iter_keys[6], key + 24, 4);
//    memcpy(iter_keys[7], key + 28, 4);
//
//    memcpy(iter_keys[8], key, 4);
//    memcpy(iter_keys[9], key + 4, 4);
//    memcpy(iter_keys[10], key + 8, 4);
//    memcpy(iter_keys[11], key + 12, 4);
//    memcpy(iter_keys[12], key + 16, 4);
//    memcpy(iter_keys[13], key + 20, 4);
//    memcpy(iter_keys[14], key + 24, 4);
//    memcpy(iter_keys[15], key + 28, 4);
//
//    memcpy(iter_keys[16], key, 4);
//    memcpy(iter_keys[17], key + 4, 4);
//    memcpy(iter_keys[18], key + 8, 4);
//    memcpy(iter_keys[19], key + 12, 4);
//    memcpy(iter_keys[20], key + 16, 4);
//    memcpy(iter_keys[21], key + 20, 4);
//    memcpy(iter_keys[22], key + 24, 4);
//    memcpy(iter_keys[23], key + 28, 4);
//
//    memcpy(iter_keys[24], key + 28, 4);
//    memcpy(iter_keys[25], key + 24, 4);
//    memcpy(iter_keys[26], key + 20, 4);
//    memcpy(iter_keys[27], key + 16, 4);
//    memcpy(iter_keys[28], key + 12, 4);
//    memcpy(iter_keys[29], key + 8, 4);
//    memcpy(iter_keys[30], key + 4, 4);
//    memcpy(iter_keys[31], key, 4);
//}
__device__ half_block_t magma_T(const half_block_t data) {
    half_block_t result;
    uint8_t first_part_byte, sec_part_byte;
    for (int i = 0; i < 4; ++i) {
        first_part_byte = data.bytes[i] >> 4;
        sec_part_byte = (data.bytes[i] & 0x0f);
        first_part_byte = table[i * 2][first_part_byte];
        sec_part_byte = table[i * 2 + 1][sec_part_byte];
        result.bytes[i] = (first_part_byte << 4) | sec_part_byte;
    }
    return result;
}

__device__ half_block_t magma_g(const key_t round_key, const half_block_t block) {
    half_block_t internal = block;
    internal.uint = Add32(internal, round_key);
    internal = magma_T(internal);
    internal.uint = (internal.uint << 11) | (internal.uint >> 21);

    return internal;
}

__device__ block_t magma_G(const key_t round_key, const block_t block) {

    block_t result = block;
    result.lo = magma_g(round_key, result.lo);
    result.lo.uint = Add(result.lo, block.hi);

    result.hi = block.lo;
    return result;
}

__device__ block_t magma_G_last(const key_t round_key, const block_t block)
{
    block_t result = block;
    result.lo = magma_g(round_key, result.lo);
    result.lo.uint = Add(result.lo, block.hi);

    result.hi = result.lo;
    result.lo = block.lo;

    return result;
}

__global__ void Encrypt(const key_set& round_key, const block_t& block, block_t& result_block)
{

    // Первое преобразование G
    block_t result;
    result = magma_G(round_key.keys[0], block);
    // Последующие (со второго по тридцать первое) преобразования G
    for (int i = 1; i < 24; ++i)
        result = magma_G(round_key.keys[i % 8], result);
    for (int i = 0; i < 7; ++i)
        result = magma_G(round_key.keys[7 - i % 8], result);
    // Последнее (тридцать второе) преобразование G
    result = magma_G_last(round_key.keys[0], result);
    result_block = result;
}


__global__ void Decrypt(const key_set& round_key, const block_t& block, block_t& result_block)
{
    block_t result;
    result = magma_G(round_key.keys[0], block);
    for (int i = 1; i < 8; ++i)
        result = magma_G(round_key.keys[i % 8], result);
    for (int i = 0; i < 23; ++i)
        result = magma_G(round_key.keys[7 - i % 8], result);
    // Последнее (тридцать второе) преобразование G
    result = magma_G_last(round_key.keys[0], result);
    result_block = result;
}


void encryptCuda(const uint8_t* block, uint8_t* out_block, const key_set& keys);
void decryptCuda(const uint8_t* block, uint8_t* out_block, const key_set& keys);

/** @brief Конвертирует массив байт в int32
 *
 * @param[in] input массив из 4 байт
 * @return int32 число
 */
static unsigned int uint8ToUint32(unsigned char* input)
{
    unsigned int r = ((input[3]) | (input[2] << 8) | (input[1] << 16) | (input[0] << 24));
    return r;
}

/** @brief Конвертирует int32 в массив байт
 *
 * @param[in] input int32 число
 * @param[out] output массив из 4 байт
 */
static void uint32ToUint8(unsigned int input, unsigned char* output)
{
    int i;
    for (i = 0; i < 4; ++i)
    {
        output[3 - i] = ((input >> (8 * i)) & 0x000000ff);
    }
}

int main()
{
    // SetConsoleOutputCP(1251);
    // SetConsoleCP(1251);
    SetConsoleOutputCP(65001);
    block_t en_blk, dec_blk;
    const unsigned char encrypt_test_string[8] = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    };
    memccpy(en_blk.bytes, encrypt_test_string, 8, 8);

    std::cout << en_blk.lo.uint << std::endl;
    en_blk.hi.uint = uint8ToUint32(en_blk.hi.bytes);
    en_blk.lo.uint = uint8ToUint32(en_blk.lo.bytes);
    uint32ToUint8(en_blk.hi.uint, en_blk.hi.bytes);
    uint32ToUint8(en_blk.lo.uint, en_blk.lo.bytes);
    std::cout << en_blk.hi.uint << std::endl;
    std::cout << en_blk.lo.uint << std::endl;


    const unsigned char decrypt_test_string[8] = {
        0x3d, 0xca, 0xd8, 0xc2, 0xe5, 0x01, 0xe9, 0x4e
    };
    memccpy(dec_blk.bytes, decrypt_test_string, 8, 8);

    const key_set keys = {
        key_t {0xff, 0xfe, 0xfd, 0xfc},
        key_t {0xfb, 0xfa, 0xf9, 0xf8},
        key_t {0xf7, 0xf6, 0xf5, 0xf4},
        key_t {0xf3, 0xf2, 0xf1, 0xf0},
        key_t {0x00, 0x11, 0x22, 0x33},
        key_t {0x44, 0x55, 0x66, 0x77},
        key_t {0x88, 0x99, 0xaa, 0xbb},
        key_t {0xcc, 0xdd, 0xee, 0xff}
    };

    cudaDeviceProp prop;
    int count;
    cudaError_t(cudaGetDeviceCount(&count));
    /*for (int i = 0; i < count; i++) {
        cudaError_t(cudaGetDeviceProperties(&prop, i));
        printf("--- Общая информация об устройстве %d ---\n", i);
        printf("Имя: %s\n", prop.name);
        printf("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
        printf("Тактовая частота: %d\n", prop.clockRate);
        printf("Перекрытие копирования: ");
        if (prop.deviceOverlap)
            printf("Разрешено\n");
        else
            printf("Запрещено\n");
        printf("Тайм-аут выполнения ядра: ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Включён\n");
        else
            printf("Выключен\n");
        printf("--- Информация о памяти для устройства %d ---\n", i);
        printf("Всего глобальной памяти: %ld\n", prop.totalGlobalMem);
        printf("Всего константной памяти: %ld\n", prop.totalConstMem);
        printf("Максимальный шаг: %ld\n", prop.memPitch);
        printf("Выравнивание текстур: %ld\n", prop.textureAlignment);

        printf("--- Информация о мультипроцессорах для устройства %d ---\n", i);
        printf("Количество мультипроцессоров: %d\n", prop.multiProcessorCount);
        printf("Разделяемая память на один МП: %ld\n", prop.sharedMemPerBlock);
        printf("Регистров на один МП: %d\n", prop.regsPerBlock);
        printf("Нитей в варпе: %d\n", prop.warpSize);
        printf("Макс. количество ниетй в блоке: %d\n", prop.maxThreadsPerBlock);
        printf("Макс. количество нитей по измерениям: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Максимальные размеры сетки: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }*/

    block_t resultEncrypt{};

    // uint8_t* iter_keys[32];
    // ExpandKey(keys);
    std::cout << "encrypt_test_string: " << en_blk << std::endl;

    encryptCuda(encrypt_test_string, resultEncrypt.bytes, keys);


    //printf("Encryption text: {%s}\n", result);
    std::cout << "Encryption text: " << resultEncrypt << std::endl;
    std::cout << "encrypt_test_string: " << dec_blk << std::endl;

    block_t resultDecrypt{};

    decryptCuda(resultEncrypt.bytes, resultDecrypt.bytes, keys);

    //printf("Encryption text: {%s}\n", result);
    std::cout << "Decryption text: " << resultDecrypt << std::endl;

    cudaCheck(cudaDeviceReset());
    return 0;
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
cuda_ptr<T[]> cuda_alloc(size_t n) {
    T* ptr;
    cudaCheck(cudaMalloc((void**)&ptr, sizeof(T) * n));
    return cuda_ptr<T[]>{ptr};
}

void encryptCuda(const uint8_t* block, uint8_t* out_block, const key_set& keys) {
    cuda_ptr<key_set> dev_keys = cuda_alloc<key_set>();
    cuda_ptr<block_t> dev_block = cuda_alloc<block_t>();
    cuda_ptr<block_t> dev_out_block = cuda_alloc<block_t>();

    cudaError_t cudaStatus;

    cudaCheck(cudaSetDevice(0));


    cudaCheck(cudaMemcpy(dev_keys.get(), keys.keys, sizeof(key_set), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dev_block.get(), block, sizeof(block_t), cudaMemcpyHostToDevice));

    // cudaCheck(cudaGetLastError());

    Encrypt << <1, 1 >> > (*dev_keys, *dev_block, *dev_out_block);

    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(out_block, dev_out_block.get(), sizeof(block_t), cudaMemcpyDeviceToHost));



    // return cudaStatus;
}


void decryptCuda(const uint8_t* block, uint8_t* out_block, const key_set& keys) {
    cuda_ptr<key_set> dev_keys = cuda_alloc<key_set>();
    cuda_ptr<block_t> dev_block = cuda_alloc<block_t>();
    cuda_ptr<block_t> dev_out_block = cuda_alloc<block_t>();

    cudaError_t cudaStatus;

    cudaCheck(cudaSetDevice(0));

    cudaCheck(cudaMemcpy(dev_keys.get(), keys.keys, sizeof(key_set), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(dev_block.get(), block, sizeof(block_t), cudaMemcpyHostToDevice));

    // cudaCheck(cudaGetLastError());

    Decrypt << <1, 1 >> > (*dev_keys, *dev_block, *dev_out_block);

    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(out_block, dev_out_block.get(), sizeof(block_t), cudaMemcpyDeviceToHost));

Error:
    dev_block.get_deleter();
    dev_out_block.get_deleter();
    dev_keys.get_deleter();


    // return cudaStatus;
}



// text: fedcba9876543210
// encr_text: 4ee901e5c2d8ca3d
// dec_text: fedcba9876543210