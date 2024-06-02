#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

void getInfoDevice() {
    cudaDeviceProp prop;
    int count;
    cudaError_t(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        cudaError_t(cudaGetDeviceProperties(&prop, i));
        std::cout << "--- Общая информация об устройстве " << i << " ---" << std::endl;
        std::cout << "Имя: " << prop.name << std::endl;
        std::cout << "Вычислительные возможности: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Тактовая частота: " << prop.clockRate << std::endl;
        std::cout << "Перекрытие копирования: ";
        if (prop.deviceOverlap)
            std::cout << "Разрешено" << std::endl;
        else
            std::cout << "Запрещено" << std::endl;
        std::cout << "Тайм-аут выполнения ядра: ";
        if (prop.kernelExecTimeoutEnabled)
            std::cout << "Включён" << std::endl;
        else
            std::cout << "Выключен" << std::endl;
        std::cout << "--- Информация о памяти для устройства" << i << " ---" << std::endl;
        std::cout << "Всего глобальной памяти: " << prop.totalGlobalMem << std::endl;
        std::cout << "Всего константной памяти: " << prop.totalConstMem << std::endl;
        std::cout << "Максимальный шаг: " << prop.memPitch << std::endl;
        std::cout << "Выравнивание текстур: " << prop.textureAlignment << std::endl;

        std::cout << "--- Информация о мультипроцессорах для устройства " << i << " ---" << std::endl;
        std::cout << "Количество мультипроцессоров: " << prop.multiProcessorCount << std::endl;
        std::cout << "Разделяемая память на один МП: " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Регистров на один МП: " << prop.regsPerBlock << std::endl;
        std::cout << "Нитей в варпе: " << prop.warpSize << std::endl;
        std::cout << "Макс. количество ниетй в блоке: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Макс. количество нитей по измерениям: " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Максимальные размеры сетки: (%d, %d, %d)\n" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    }
}
