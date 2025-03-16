#pragma once

#include <iostream>
#include <fstream>
#include <random> 
#include <string>

class generateFile{
private:
    size_t sizeFile;
    size_t sizeBlock;

public:
    generateFile();
    generateFile(const size_t sizeFile, const size_t sizeBlock = 8);
    bool generate(const std::string& path);
    ~generateFile() {};
};