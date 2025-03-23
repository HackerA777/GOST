#include "./magma/magma.cuh"
#include "./kuznechik/kuznechik.cuh"
#include <Windows.h>

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
    magma magmaElement(keys, 1024 * 1024 * 1024 * 2.0 / sizeof(magmaBlockT), 512, 1024);
    magmaElement.checkEcnAndDec();
    //magmaElement.testSpeedUnequalBytes();
    //magmaElement.searchBestBlockAndGridSize();

    uint8_t testKeyBytesKuz[] = { 
        0xef, 0xcd, 0xab, 0x89, 
        0x67, 0x45, 0x23, 0x01, 
        0x10, 0x32, 0x54, 0x76, 
        0x98, 0xba, 0xdc, 0xfe, 
        0x77, 0x66, 0x55, 0x44, 
        0x33, 0x22, 0x11, 0x00, 
        0xff, 0xee, 0xdd, 0xcc, 
        0xbb, 0xaa, 0x99, 0x88 };

    kuznechikKeys testKeyKuz(testKeyBytesKuz);

    std::cout << 1024 * 1024 * 1024 * 0.5 / sizeof(kuznechikByteVector) << "  " << 1024 * 1024 * 1024 * 0.5 << std::endl;

    kuznechik kuznechikElement(testKeyKuz, 1024*1024*1024*0.5/sizeof(kuznechikByteVector), 512, 1024);

    kuznechikElement.checkEcnAndDec();
    //kuznechikElement.testSpeedUnequalBytes();
    //kuznechikElement.searchBestBlockAndGridSize();
    return 0;
}