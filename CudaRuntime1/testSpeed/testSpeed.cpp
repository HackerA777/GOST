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
        std::cout << "create files error" << std::endl;
    }

    std::vector<timeRes> timeVector;
    std::vector<timeResStream> timeResStreamVector;

    std::vector<std::vector<double>> testParametrs;

    for (int i = 2; i < 5; ++i) {
        for (int j = 0; j < 4; ++j) {
            testParametrs.push_back( { double(i), 1024 * std::pow(2, j), 128 * std::pow(2, j), 128 * std::pow(2, j) });
        }
    }

    for (size_t i = range[0]; i <= range[1]; i = i * 2) {
        timeRes tempTimeRes;
        newPath.append(path.data());
        newPath.append("\\");
        newPath.append(std::to_string(i));
        newPath.append("bytes");
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

        newPath.append("Enc");
        
        for (auto parametrs : testParametrs) {
            std::cout << parametrs[0] << " : " << parametrs[1] << " : " << parametrs[2] << " : " << parametrs[3] << " : " << i / 1024.0 / 1024 << "MB" << std::endl;
            timeResStream timeResTemp;
            timeResTemp.size = i;
            timeResTemp.encrypt = true;
            timeResTemp.countStream = parametrs[0];
            timeResTemp.tileSize = parametrs[1];
            timeResTemp.blockSize = parametrs[2];
            timeResTemp.gridSize = parametrs[3];

            timeResTemp.time = testMagma.testStreams(buffer, parametrs[2], parametrs[3], parametrs[0], parametrs[1], true);

            timeResStreamVector.push_back(timeResTemp);
        }

        /*std::ofstream file(newPath, std::ios::binary);
        if (!file) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        file.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        file.close();*/

        /*replaceTimeRes(tempTimeRes, newPath, "testManaged", false, i);

        tempTimeRes.time = testMagma.testManaged(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testPinned", false, i);

        tempTimeRes.time = testMagma.testPinned(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);
        

        replaceTimeRes(tempTimeRes, newPath, "testDefault", false, i);

        tempTimeRes.time = testMagma.testDefault(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);

        newPath.append("Dec");

        std::ofstream fileDec(newPath, std::ios::binary);
        if (!fileDec) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        fileDec.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        fileDec.close();
        */

        newPath.clear();
        
        buffer.clear();
    }
    std::cout << "Name test;size;enc;gridSize;blockSize;speedCopyAndEnc;speedEnc" << std::endl;
    for (auto elem : timeVector) {
        //std::cout << elem.testName << ": size: " << elem.size / 1024 / 1024.0 << "MB path: " << elem.path << " enc: " << elem.encrypt << std::endl;
       //std::cout << "blockSize: " << blockSize << "; gridSize: " << gridSize << std::endl;
       //std::cout << "Time copyAndEnc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << "GB/s; Time enc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << "GB/s" << std::endl;

        std::cout << elem.testName << ";" << elem.size / 1024 / 1024.0 << ";" << elem.encrypt << ";" << gridSize << ";" << blockSize << ";" <<
            (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << ";" << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << std::endl;
    }

    std::cout << "Test streams:\nsize;enc;countStream;tileSize;gridSize;blockSize;speed" << std::endl;
    for (auto elem : timeResStreamVector) {
        std::cout << elem.size / 1024 / 1024.0 << ";" << elem.encrypt << ";" << elem.countStream << ";" << elem.gridSize << ";" << elem.blockSize << ";" <<
            (elem.size / 1024 / 1024.0 / 1024) / (elem.time / 1000) << std::endl;
    }
}

void testSpeedKuznechik(const std::string& path, const std::vector<size_t> range, const size_t blockSize, const size_t gridSize) {
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
        readFileKuznechik(newPath, buffer);

        replaceTimeRes(tempTimeRes, newPath, "testDefault", true, i);

        tempTimeRes.time = testKuznechik.testDefault(buffer, blockSize, gridSize, true);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testPinned", true, i);

        tempTimeRes.time = testKuznechik.testPinned(buffer, blockSize, gridSize, true);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testManaged", true, i);

        tempTimeRes.time = testKuznechik.testManaged(buffer, blockSize, gridSize, true);

        timeVector.push_back(tempTimeRes);

        newPath.append("Enc");

        std::ofstream file(newPath, std::ios::binary);
        if (!file) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        file.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        file.close();

        /*replaceTimeRes(tempTimeRes, newPath, "testManaged", false, i);

        tempTimeRes.time = testMagma.testManaged(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);

        replaceTimeRes(tempTimeRes, newPath, "testPinned", false, i);

        tempTimeRes.time = testMagma.testPinned(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);


        replaceTimeRes(tempTimeRes, newPath, "testDefault", false, i);

        tempTimeRes.time = testMagma.testDefault(buffer, blockSize, gridSize, false);

        timeVector.push_back(tempTimeRes);

        newPath.append("Dec");

        std::ofstream fileDec(newPath, std::ios::binary);
        if (!fileDec) {
            std::cerr << "Error creating file: " << newPath << std::endl;
            return;
        }
        fileDec.write((char*)buffer.data()->bytes, sizeof(magmaBlockT) * buffer.size());
        fileDec.close();
        */

        newPath.clear();

        buffer.clear();
    }

    //std::cout << "Name test;size;enc;gridSize;blockSize;timeCopyAndEnc;timeEnc" << std::endl;
    //for (auto elem : timeVector) {
    //    //std::cout << elem.testName << ": size: " << elem.size / 1024 / 1024.0 << "MB path: " << elem.path << " enc: " << elem.encrypt << std::endl;
    //    //std::cout << "blockSize: " << blockSize << "; gridSize: " << gridSize << std::endl;
    //    //std::cout << "Time copyAndEnc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << "GB/s; Time enc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << "GB/s" << std::endl;
    //    
    //    std::cout << elem.testName << ";" << elem.size / 1024 / 1024.0 << ";" << elem.encrypt << ";" << gridSize << ";" << blockSize << ";" <<
    //        (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << ";" << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << std::endl;
    //}
    std::cout << "Name test;size;enc;gridSize;blockSize;speedCopyAndEnc;speedEnc" << std::endl;
    for (auto elem : timeVector) {
        //std::cout << elem.testName << ": size: " << elem.size / 1024 / 1024.0 << "MB path: " << elem.path << " enc: " << elem.encrypt << std::endl;
       //std::cout << "blockSize: " << blockSize << "; gridSize: " << gridSize << std::endl;
       //std::cout << "Time copyAndEnc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << "GB/s; Time enc: " << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << "GB/s" << std::endl;

        std::cout << elem.testName << ";" << elem.size / 1024 / 1024.0 << ";" << elem.encrypt << ";" << gridSize << ";" << blockSize << ";" <<
            (elem.size / 1024 / 1024.0 / 1024) / (elem.time[0] / 1000) << ";" << (elem.size / 1024 / 1024.0 / 1024) / (elem.time[1] / 1000) << std::endl;
    }
}