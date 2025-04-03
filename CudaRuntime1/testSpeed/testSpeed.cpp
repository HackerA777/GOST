#include "./testSpeed.h"

void readFile(const std::string path, std::vector<magmaBlockT>& result) {
    std::cout << "Opening file: " << path << std::endl;
    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile) {
        std::cerr << "Error opening file: " << path << std::endl;
    }

    inputFile.seekg(0, std::ios::end);        // Переходим в конец файла
    size_t inputFileSize = inputFile.tellg();      // Получаем размер файла
    inputFile.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer{};
    buffer.resize(inputFileSize);

    inputFile.read(reinterpret_cast<char*>(buffer.data()), inputFileSize);

    if (inputFile.gcount() != inputFileSize) {
        std::cerr << "Error read input file!" << std::endl;
    }

    inputFile.close();
    size_t count = 0;
    //std::vector<magmaBlockT> result(buffer.size() / 8);
    result.resize(buffer.size() / 8);
    for (auto &elem : result) {
        std::copy(buffer.begin() + count, buffer.begin() + count + 8, elem.bytes);
        count += 8;
    }

    buffer.~vector();
    //return result;
}

void replaceTimeRes(timeRes &timeRes_, const std::string newPath, const std::string nameTest, const bool encrypt, const size_t size) {
    timeRes_.path = newPath;
    timeRes_.testName = nameTest;
    timeRes_.encrypt = encrypt;
    timeRes_.size = size;
}

void testSpeed(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize) {
    //std::vector<size_t> range{ 1*1024*1024, 1024 * 1024 * 1024};
    //std::vector<size_t> range{ 16*1024, 32*1024 };
    std::vector<magmaBlockT> buffer;
    std::string newPath;

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
    magma testMagma(keys, 1024, blockSize, gridSize);

	generateFile generateFileForTestSpeed(range, 8);
    if (generateFileForTestSpeed.generate(path)) {
        std::cout << "create files successful" << std::endl;
    }
    else {
        std::cout << "create files error" << std::endl;
    }

    //std::vector<float> time{ 0, 0 };
    std::vector<timeRes> timeVector;

    for (size_t i = range[0]; i <= range[1]; i = i * 2) {
        timeRes tempTimeRes;
        newPath.append(path.data());
        newPath.append("\\");
        newPath.append(std::to_string(i));
        newPath.append("bytes");
        readFile(newPath, buffer);

        replaceTimeRes(tempTimeRes, newPath, "testDefault", true, i);

        tempTimeRes.time = testMagma.testDefault(buffer, buffer.size(), 16, 16, true);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testPinned", true, i);

        tempTimeRes.time = testMagma.testPinned(buffer, buffer.size(), 16, 16, true);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testManaged", true, i);

        tempTimeRes.time = testMagma.testManaged(buffer, buffer.size(), 16, 16, true);

        timeVector.push_back(tempTimeRes);

        newPath.append("Enc");

        std::ofstream file(newPath, std::ios::binary);
        if (!file) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        file.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        file.close();

        replaceTimeRes(tempTimeRes, newPath, "testManaged", false, i);

        tempTimeRes.time = testMagma.testManaged(buffer, buffer.size(), 16, 16, false);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testPinned", false, i);

        tempTimeRes.time = testMagma.testPinned(buffer, buffer.size(), 16, 16, false);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testDefault", false, i);

        tempTimeRes.time = testMagma.testDefault(buffer, buffer.size(), 16, 16, false);

        timeVector.push_back(tempTimeRes);

        newPath.append("Dec");

        std::ofstream fileDec(newPath, std::ios::binary);
        if (!fileDec) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        fileDec.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        fileDec.close();

        newPath.clear();
        
        buffer.clear();
    }

    for (auto elem : timeVector) {
        std::cout << elem.testName << ": size: " << elem.size / 1024 /1024.0 << "MB path: " << elem.path << " enc: " << elem.encrypt << std::endl;
        std::cout << "Time copyAndEnc: " << elem.time[0] / 1000 << "s; Time enc: " << elem.time[1] / 1000 << "s" << std::endl;
    }
}