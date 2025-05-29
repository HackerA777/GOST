#include "generateFile.h"

generateFile::generateFile() {
    this->range = {1, 32};
    this->sizeBlock = 8;
}

generateFile::generateFile(const std::vector<size_t> range, const size_t sizeBlock) {
    this->range = range;
    this->sizeBlock = sizeBlock;
}

bool generateFile::generate(const std::string& path) {
    if (range.size() > 2) {
        std::cout << "Invalid value range!" << std::endl;
        return false;
    }
    std::string newPath;
    for (size_t s = range[0]; s <= range[1]; s = s * 2) {
        newPath.append(path.data());

        std::ifstream inputDirectory(path, std::ios::binary);
        if (!inputDirectory) {
            std::error_code ec;
            bool created = std::filesystem::create_directory(path, ec);

            if (ec) { // Ошибки файловой системы (нет прав, неверный путь и т.д.)
                throw std::filesystem::filesystem_error("Ошибка создания директории", path, ec);
            }
        }

        newPath.append("\\");
        newPath.append(std::to_string(s));
        newPath.append("bytes");

        std::ifstream inputFile(newPath, std::ios::binary);
        if (!inputFile) {
            std::cerr << "Error opening file: " << path << std::endl;
        }
        inputFile.seekg(0, std::ios::end);        // Переходим в конец файла
        size_t inputFileSize = inputFile.tellg();      // Получаем размер файла
        inputFile.seekg(0, std::ios::beg);

        if (inputFileSize == s) {
            inputFile.close();
            newPath.clear();
            continue;
        }
        else {
            inputFile.close();
        }

        std::ofstream file(newPath, std::ios::binary);
        if (!file) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return false;
        }

        std::random_device random;
        std::mt19937 generator(random());
        std::uniform_int_distribution<unsigned int> distribution(0, 255);

        const size_t bufferSize = 4096;
        char buffer[bufferSize];

        for (size_t i = 0; i < s; i += bufferSize) {
            size_t bytesToWrite = std::min(bufferSize, s - i);
            for (size_t j = 0; j < bytesToWrite; ++j) {
                buffer[j] = distribution(generator);
            }
            file.write(buffer, bytesToWrite);
        }

        file.close();
        newPath.clear();
        std::cout << "File generated: " << newPath << " (" << s << " bytes)" << std::endl;
    }
    return true;
}