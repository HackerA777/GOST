#include "./magma/magma.cuh"
#include "./kuznechik/kuznechik.cuh"
#include "./testSpeed/testSpeed.h"
#include "./tests/tests.cuh"
#include <Windows.h>
#include <fstream>

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
    SetConsoleOutputCP(65001);

    std::cout << "Results for alg 'Magma': " << std::endl;
    testSpeedMagma("C:\\Users\\artio\\Documents\\testFilesForGOST", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, 192, 16 * 1024);

    std::cout << "Results for alg 'Kuznechik - 1': " << std::endl;
    testSpeedKuznechik("C:\\Users\\artio\\Documents\\testFilesForGOST", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, 1024, 16 * 1024, 1);

    std::cout << "Results for alg 'Kuznechik - 2': " << std::endl;
    testSpeedKuznechik("C:\\Users\\artio\\Documents\\testFilesForGOST", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, 1024, 16 * 1024, 2);

    return 0;
}