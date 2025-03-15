#include <iostream>
#include <cuda_runtime.h>

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << deviceCount << std::endl;

    return 0;
}