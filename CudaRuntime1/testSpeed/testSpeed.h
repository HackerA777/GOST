#pragma one
#include "../structures/structures.cuh"
#include "../magma/magma.cuh"
#include "../kuznechik/kuznechik.cuh"
#include "../generateFiles/generateFile.h"

std::vector<uint8_t> readFileMagma(const std::string path);
void testSpeedMagma(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize);
void testSpeedKuznechik(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize);