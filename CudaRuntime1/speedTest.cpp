#include "./ctxFactory/ctxFactory.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

std::vector<uint8_t> readFile(const std::string path){
    std::cout << "Opening file: " << path << std::endl;
    std::ifstream inputFile(path, std::ios::binary);
    if (!inputFile){
        std::cerr << "Error opening file: " << path << std::endl;
    }

    inputFile.seekg(0, std::ios::end);        // Переходим в конец файла
    size_t inputFileSize = inputFile.tellg();      // Получаем размер файла
    inputFile.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer {};
    buffer.resize(inputFileSize);

    inputFile.read(reinterpret_cast<char*>(buffer.data()), inputFileSize);

    if (inputFile.gcount() != inputFileSize) {
        std::cerr << "Error read input file!" << std::endl;
    }

    inputFile.close();
    return buffer;
}

int main(){
    //MAGMA
    CtxFactory F = CtxFactory("gost_provider", "magma");
    std::vector<uint8_t> buffer = readFile("./testFiles/input/test1024");
    std::vector<uint8_t> result;

    unsigned char key[32] = {
        0xcc, 0xdd, 0xee, 0xff,
        0x88, 0x99, 0xaa, 0xbb,
        0x44, 0x55, 0x66, 0x77,
        0x00, 0x11, 0x22, 0x33,
        0xf3, 0xf2, 0xf1, 0xf0,
        0xf7, 0xf6, 0xf5, 0xf4,
        0xfb, 0xfa, 0xf9, 0xf8,
        0xff, 0xfe, 0xfd, 0xfc
    };

    std::string outputPath = "./testFiles/outputEnc/testSPM1024.txt.enc";
    /*
    result.resize(buffer.size());
    OsslCtx c = F.encryptInit(key);

    size_t step = 1024 * 1024; // MBytes

    c.encrypt((unsigned char *)buffer.data(), (unsigned char *)result.data(), buffer.size());

    std::ofstream outputFile(outputPath, std::ios::binary);
    if (!outputFile){
        std::cerr << "Error opening file: " << outputPath << std::endl;
        return -1;
    }

    outputFile.write((char*)result.data(), result.size());
    outputFile.close();

    CtxFactory D = CtxFactory("gost_provider", "magma");

    buffer = readFile("./testFiles/outputEnc/testSPM1024.txt.enc");
    OsslCtx d = D.decryptInit(key);

    d.decrypt((unsigned char *)buffer.data(), (unsigned char *)result.data(), buffer.size());

    std::string outputPathDec = "./testFiles/outputDec/testSPM1024.txt";
    std::ofstream outputFileDec(outputPathDec, std::ios::binary);
    if (!outputFileDec){
        std::cerr << "Error opening file: " << outputPathDec << std::endl;
        return -1;
    }

    outputFileDec.write((char*)result.data(), result.size());
    outputFileDec.close();
    */

    //KUZNECHIK
    ///*
    CtxFactory KZ = CtxFactory("gost_provider", "kuznechik");
    buffer = readFile("./testFiles/input/test1024");

    result.resize(buffer.size());
    OsslCtx kz = KZ.encryptInit(key);

    kz.encrypt((unsigned char *)buffer.data(), (unsigned char *)result.data(), buffer.size());

    outputPath = "./testFiles/outputEnc/testSPK1024.txt.enc";
    std::ofstream outputFileEncK(outputPath, std::ios::binary);
    if (!outputFileEncK){
        std::cerr << "Error opening file: " << outputPath << std::endl;
        return -1;
    }

    outputFileEncK.write((char*)result.data(), result.size());
    outputFileEncK.close();

    CtxFactory K = CtxFactory("gost_provider", "kuznechik");

    buffer = readFile("./testFiles/outputEnc/testSPK1024.txt.enc");
    OsslCtx k = K.decryptInit(key);

    k.decrypt((unsigned char *)buffer.data(), (unsigned char *)result.data(), buffer.size());

    std::string outputPathDecK = "./testFiles/outputDec/testSPK1024.txt";
    std::ofstream outputFileDecK(outputPathDecK, std::ios::binary);
    if (!outputFileDecK){
        std::cerr << "Error opening file: " << outputPathDecK << std::endl;
        return -1;
    }

    outputFileDecK.write((char*)result.data(), result.size());
    outputFileDecK.close();
    //*/
}