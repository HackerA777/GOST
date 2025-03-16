#include "generateFile.h"

int main(){
    const std::string pathToFile = "/home/user/Documents/GOST/CudaRuntime1/testFiles/input/test1024";
    const size_t sizeFile = 1024;
    const size_t sizeBlock = 8;

    generateFile generatorFile(sizeFile, sizeBlock);

    if (!generatorFile.generate(pathToFile)){
        std::cout << "Error!" << std::endl;
    }
    return 0;
}
