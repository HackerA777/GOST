#include "magma.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>

int main()
{
    // SetConsoleOutputCP(1251);
    // SetConsoleCP(1251);
    SetConsoleOutputCP(65001);
    /*cudaDeviceProp prop;
    int count;
    cudaError_t(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        cudaError_t(cudaGetDeviceProperties(&prop, i));
        std::cout << "--- Информация о мультипроцессорах для устройства " << i << " ---" << std::endl;
        std::cout << "Количество мультипроцессоров: " << prop.multiProcessorCount << std::endl;
        std::cout << "Разделяемая память на один МП: " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Регистров на один МП: " << prop.regsPerBlock << std::endl;
        std::cout << "Нитей в варпе: " << prop.warpSize << std::endl;
        std::cout << "Макс. количество ниетй в блоке: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Макс. количество нитей по измерениям: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Максимальные размеры сетки: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    }*/
    cudaDeviceProp prop;
    cudaError_t(cudaGetDeviceProperties(&prop, 0));
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
    //std::cout << "MAX VALUE SIZE_T: " << MAXSIZE_T << "\nsize: " << 1024.0*1024*1024 << std::endl;
    magma magmaElement(keys, 1024*1024*1024*3.0/sizeof(block_t), 512, 512);
    // magmaElement.checkEcnAndDec();
    double max_speed = 0, max_speed_avg = 0;
    /* size_t size = 1024 * 1024 * 1024 * 3;
    size_t gridSize = 8, bestGridSize;
    size_t blockSize = 8, bestBlockSize;
    for (size_t i = blockSize; i < prop.maxThreadsDim[0]; i *= 2) {
        for (size_t j = gridSize; j < prop.maxThreadsDim[1]; j *= 2) {
            magmaElement.setBlockSize(i);
            magmaElement.setGridSize(j);
            for (size_t k = 0; k < 10; ++k) {
                max_speed_avg += magmaElement.testSpeedRandomBytes();
            }
            max_speed_avg = max_speed_avg / 10;
            if (max_speed < max_speed_avg) {
                max_speed = max_speed_avg;
                bestBlockSize = i;
                bestGridSize = j;
            }
        }
    }
    std::cout << "Max speed: " << max_speed << "\nBest block size: " << bestBlockSize << "\nBest grid size: " << bestGridSize << std::endl;*/
    /*for (int i = 0; i < 15; ++i)
        max_speed_avg += magmaElement.testSpeedRandomBytes();
    std::cout << "Avg speed: " << max_speed_avg / 15.0 << std::endl;*/
    magmaElement.testSpeedRandomBytes();
    return 0;
}