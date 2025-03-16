#include "generateFile.h"

generateFile::generateFile() {
    this->sizeFile = 16.0 * 1024 * 1024;
    this->sizeBlock = 8;
}

generateFile::generateFile(const size_t sizeFile, const size_t sizeBlock) { 
    this->sizeFile = sizeFile * 1024 * 1024;
    this->sizeBlock = sizeBlock;
}

bool generateFile::generate(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file){
        std::cerr << "Error creating file: " << path << std::endl;
        return false;
    }

    std::random_device random;
    std::mt19937 generator(random());
    std::uniform_int_distribution<unsigned char> distribution(0, 255);

    const size_t bufferSize = 4096;
    char buffer[bufferSize];

    for (size_t i = 0; i < this->sizeFile; i += bufferSize){
        size_t bytesToWrite = std::min(bufferSize, this->sizeFile - i);
        for (size_t j = 0; j < bytesToWrite; ++j){
            buffer[j] = distribution(generator);
        }
        file.write(buffer, bytesToWrite);
    }
    
    file.close();
    std::cout << "File generated: " << path << " (" << this->sizeFile << " bytes)" << std::endl;
    return true;
}