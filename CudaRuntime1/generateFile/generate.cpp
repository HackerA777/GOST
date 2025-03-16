#include "generateFile.h"

int main(){
    const std::string pathToFile = "./generatedFiles/file";
    const size_t sizeFile = 32;
    const size_t sizeBlock = 8;

    generateFile generatorFile(sizeFile, sizeBlock);

    if (!generatorFile.generate(pathToFile)){
        std::cout << "Error!" << std::endl;
    }
    return 0;
}