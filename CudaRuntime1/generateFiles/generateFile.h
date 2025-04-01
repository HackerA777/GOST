#pragma once

#include <iostream>
#include <fstream>
#include <random> 
#include <string>

class generateFile{
private:
    std::vector<size_t> range;
    size_t sizeBlock;

public:
    generateFile();
    generateFile(const std::vector<size_t> range = {1, 32}, const size_t sizeBlock = 8);
    bool generate(const std::string& path);
    ~generateFile() {};
};