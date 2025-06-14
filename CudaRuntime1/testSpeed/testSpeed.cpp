#include "./testSpeed.h"

void readFileMagma(const std::string path, std::vector<magmaBlockT>& result) {
    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile) {
        throw "Error read input file!";
    }

    inputFile.seekg(0, std::ios::end);
    size_t inputFileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer{};
    buffer.resize(inputFileSize);

    inputFile.read(reinterpret_cast<char*>(buffer.data()), inputFileSize);

    if (inputFile.gcount() != inputFileSize) {
        throw "Error read input file!";
    }

    inputFile.close();
    size_t count = 0;
    result.resize(buffer.size() / 8);
    for (auto &elem : result) {
        std::copy(buffer.begin() + count, buffer.begin() + count + 8, elem.bytes);
        count += 8;
    }
}

void readFileKuznechik(const std::string path, std::vector<kuznechikByteVector>& result) {
    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile) {
        throw "Error read input file!";
    }

    inputFile.seekg(0, std::ios::end);        // Переходим в конец файла
    size_t inputFileSize = inputFile.tellg();      // Получаем размер файла
    inputFile.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer{};
    buffer.resize(inputFileSize);

    inputFile.read(reinterpret_cast<char*>(buffer.data()), inputFileSize);

    if (inputFile.gcount() != inputFileSize) {
        throw "Error read input file!";
    }

    inputFile.close();
    size_t count = 0;
    result.resize(buffer.size() / 8);
    for (auto& elem : result) {
        std::copy(buffer.begin() + count, buffer.begin() + count + 8, elem.bytes);
        count += 8;
    }

    //buffer.~vector();
    //return result;
}

void replaceTimeRes(timeRes &timeRes_, const std::string newPath, const std::string nameTest, const bool encrypt, const size_t size) {
    timeRes_.path = newPath;
    timeRes_.testName = nameTest;
    timeRes_.encrypt = encrypt;
    timeRes_.size = size;
}

void testSpeedMagma(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize) {
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
    if (!generateFileForTestSpeed.generate(path)) {
        throw "Сreate files error!";
    }

    std::vector<timeRes> timeVector;
    std::vector<timeResStream> timeResStreamVector;

    // std::vector<std::vector<double>> testParametrs;

    std::vector<double> timeEnc{}; // [0] - testDefaul, [1] - testPinned, [2] - testManaged, [3] - testStreams with one stream, [4] - testStreams with four streams
    std::vector<double> timeEncAndCopy{}; // [0] - testDefaul, [1] - testPinned, [2] - testManaged

    timeEnc.resize(5);
    timeEncAndCopy.resize(3);

    //for (int i = 2; i < 5; ++i) {
    //    for (int j = 1; j < 4; ++j) {
    //        //testParametrs.push_back( { double(i), 1024 * std::pow(2, j), 128 * std::pow(2, j), 128 * std::pow(2, j) });
    //        testParametrs.push_back({ double(i), 1024 * std::pow(2, j), 256, 1024*16 });
    //    }
    //}

    for (size_t i = range[0]; i <= range[1]; i = i * 2) {
        timeRes tempTimeRes;
        newPath.append(path.data());
        newPath.append("\\");
        newPath.append(std::to_string(i));
        newPath.append("bytes");

        for (size_t j = 0; j < 10; ++j) {
            readFileMagma(newPath, buffer);

            replaceTimeRes(tempTimeRes, newPath, "testDefault", true, i);

            tempTimeRes.time = testMagma.testDefault(buffer, blockSize, gridSize, true);

            timeVector.push_back(tempTimeRes);

            replaceTimeRes(tempTimeRes, newPath, "testPinned", true, i);

            tempTimeRes.time = testMagma.testPinned(buffer, blockSize, gridSize, true);

            timeVector.push_back(tempTimeRes);

            replaceTimeRes(tempTimeRes, newPath, "testManaged", true, i);

            tempTimeRes.time = testMagma.testManaged(buffer, blockSize, gridSize, true);

            timeVector.push_back(tempTimeRes);

            timeEnc[3] += testMagma.testStreams(buffer, blockSize, gridSize, 1, 8 * 1024 * 1024, true);

            timeEnc[4] += testMagma.testStreams(buffer, blockSize, gridSize, 4, 8 * 1024 * 1024, true);

            buffer.clear();
        }

        newPath.append("Enc");

        newPath.clear();
    }
    
    for (auto elem : timeVector) {
        if (elem.testName == "testDefault") {
            timeEnc[0] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[0] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
        if (elem.testName == "testPinned") {
            timeEnc[1] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[1] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
        if (elem.testName == "testManaged") {
            timeEnc[2] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[2] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
    }

    /*for (auto elem : timeResStreamVector) {
        timeEnc[3] += (elem.size / 1024 / 1024.0 / 1024) / (elem.time / 1000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
    }*/

    timeEnc[0] = timeEnc[0] / 10.0;
    timeEnc[1] = timeEnc[1] / 10.0;
    timeEnc[2] = timeEnc[2] / 10.0;
    timeEnc[3] = 1.0 / (timeEnc[3] / 10000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
    timeEnc[4] = 1.0 / (timeEnc[4] / 10000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;

    timeEncAndCopy[0] = timeEncAndCopy[0] / 10.0;
    timeEncAndCopy[1] = timeEncAndCopy[1] / 10.0;
    timeEncAndCopy[2] = timeEncAndCopy[2] / 10.0;

    std::cout << "Encryption speed for mode copy 'Default': " << timeEnc[0] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[0] << " Gb/s" << std::endl;
    std::cout << "Encryption speed for mode copy 'Pinned': " << timeEnc[1] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[1] << " Gb/s" << std::endl;
    std::cout << "Encryption speed for mode copy 'Managed': " << timeEnc[2] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[2] << " Gb/s" << std::endl;
    std::cout << "Encryption speed with use one stream command: " << timeEnc[3] << " Gb/s" << std::endl;
    std::cout << "Encryption speed with use four streams command: " << timeEnc[4] << " Gb/s" << std::endl;
}

void testSpeedKuznechik(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize, const int realeseVersion) {
    std::vector<kuznechikByteVector> buffer;
    std::string newPath;

    uint8_t keysKuz[32] = {
         0xef, 0xcd, 0xab, 0x89,
         0x67, 0x45, 0x23, 0x01,
         0x10, 0x32, 0x54, 0x76,
         0x98, 0xba, 0xdc, 0xfe,
         0x77, 0x66, 0x55, 0x44,
         0x33, 0x22, 0x11, 0x00,
         0xff, 0xee, 0xdd, 0xcc,
         0xbb, 0xaa, 0x99, 0x88
    };
    kuznechikKeys testKeys(keysKuz);

    kuznechik testKuznechik(testKeys, 32, 32, 32);

    generateFile generateFileForTestSpeed(range, 8);
    if (!generateFileForTestSpeed.generate(path)) {
        throw "Сreate files error!";
    }

    //std::vector<float> time{ 0, 0 };
    std::vector<timeRes> timeVector;
    std::vector<double> timeStreams;

    std::vector<double> timeEnc{}; // [0] - testDefaul, [1] - testPinned, [2] - testManaged, [3] - testStreams with one stream, [4] - testStreams with four streams
    std::vector<double> timeEncAndCopy{}; // [0] - testDefaul, [1] - testPinned, [2] - testManaged

    timeEnc.resize(5);
    timeEncAndCopy.resize(3);

    for (size_t i = range[0]; i <= range[1]; i = i * 2) {
        timeRes tempTimeRes;
        newPath.append(path.data());
        newPath.append("\\");
        newPath.append(std::to_string(i));
        newPath.append("bytes");

        for (size_t j = 0; j < 10; ++j) {
            readFileKuznechik(newPath, buffer);

            replaceTimeRes(tempTimeRes, newPath, "testDefault", true, i);

            tempTimeRes.time = testKuznechik.testDefault(buffer, blockSize, gridSize, true, realeseVersion);

            timeVector.push_back(tempTimeRes);

            replaceTimeRes(tempTimeRes, newPath, "testPinned", true, i);

            tempTimeRes.time = testKuznechik.testPinned(buffer, blockSize, gridSize, true, realeseVersion);

            timeVector.push_back(tempTimeRes);

            replaceTimeRes(tempTimeRes, newPath, "testManaged", true, i);

            tempTimeRes.time = testKuznechik.testManaged(buffer, blockSize, gridSize, true, realeseVersion);

            timeVector.push_back(tempTimeRes);

            timeEnc[3] += testKuznechik.testStreams(buffer, blockSize, gridSize, 1, 8 * 1024 * 1024, true, realeseVersion);

            timeEnc[4] += testKuznechik.testStreams(buffer, blockSize, gridSize, 4, 8 * 1024 * 1024, true, realeseVersion);

            buffer.clear();
        }

        newPath.append("Enc");

        newPath.clear();
    }

    for (auto elem : timeVector) {
        if (elem.testName == "testDefault") {
            timeEnc[0] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[0] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
        if (elem.testName == "testPinned") {
            timeEnc[1] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[1] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
        if (elem.testName == "testManaged") {
            timeEnc[2] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
            timeEncAndCopy[2] += ((elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000)) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
        }
    }

    /*for (auto elem : timeResStreamVector) {
        timeEnc[3] += (elem.size / 1024 / 1024.0 / 1024) / (elem.time / 1000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
    }*/

    timeEnc[0] = timeEnc[0] / 10.0;
    timeEnc[1] = timeEnc[1] / 10.0;
    timeEnc[2] = timeEnc[2] / 10.0;
    timeEnc[3] = 1.0 / (timeEnc[3] / 10000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;
    timeEnc[4] = 1.0 / (timeEnc[4] / 10000) * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8;

    timeEncAndCopy[0] = timeEncAndCopy[0] / 10.0;
    timeEncAndCopy[1] = timeEncAndCopy[1] / 10.0;
    timeEncAndCopy[2] = timeEncAndCopy[2] / 10.0;

    std::cout << "Encryption speed for mode copy 'Default': " << timeEnc[0] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[0] << " Gb/s" << std::endl;
    std::cout << "Encryption speed for mode copy 'Pinned': " << timeEnc[1] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[1] << " Gb/s" << std::endl;
    std::cout << "Encryption speed for mode copy 'Managed': " << timeEnc[2] << " Gb/s\nEncryption speed with copy: " << timeEncAndCopy[2] << " Gb/s" << std::endl;
    std::cout << "Encryption speed with use one stream command: " << timeEnc[3] << " Gb/s" << std::endl;
    std::cout << "Encryption speed with use four streams command: " << timeEnc[4] << " Gb/s" << std::endl;

}