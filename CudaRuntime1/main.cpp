#include "./magma/magma.cuh"
#include "./kuznechik/kuznechik.cuh"
#include "./testSpeed/testSpeed.h"
#include "./tests/tests.cuh"
#include <Windows.h>
#include <fstream>

template<typename typeVector>
std::string printVector(const typeVector& block) {
    std::string result;
    for (int i = sizeof(typeVector) - 1; i >= 0; --i) {
        uint8_t high, low;
        high = block.bytes[i] >> 4;
        low = block.bytes[i] & 0xf;
        result.push_back((char)(high < 10 ? ('0' + high) : ('A' + high - 10)));
        result.push_back((char)(low < 10 ? ('0' + low) : ('A' + low - 10)));
    }
    return result;
}

int main()
{
    SetConsoleOutputCP(65001);

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
    magma magmaElement(keys, 1024 * 1024 * 1024 * 2.0 / sizeof(magmaBlockT), 512, 1024);
    //magmaElement.checkEcnAndDec();

    const unsigned char testString[8] = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    };

    magmaBlockT t;
    std::vector<magmaBlockT> tVector;
    
    std::copy(testString, testString + 8, t.bytes);
    //tVector.push_back(t);
    for (int i = 0; i < 2048; ++i) {
        tVector.push_back(t);
    }
    
    /*std::cout << "Test span: " << std::endl;

    for (auto i : testBlock) {
        std::cout << i;
    }
    std::cout << std::endl;*/

    //std::cout << "test copy: " << *tPtr << std::endl;

    // std::cout << std::endl << "Test Default" << tVector.size() << std::endl;
    //magmaKeySet magmaKeys;
    //std::copy(keys, keys + 32, magmaKeys.keys->bytes);
    //std::vector<float> timeDefault;
    //std::vector<float> time;
    //const size_t countStreams = 2;
    //time.resize(countStreams);
    //
    //std::cout << tVector[0] << " : " << tVector[1023] << " : " << tVector[2047] << std::endl;
    //
    //std::cout << "testStreams: " << magmaElement.testStreams(tVector, 128, 128, countStreams, 1024, true) << std::endl;    

    // std::cout << "\ntime dafault: ";
    // for (auto time : timeDefault) {
    //  std::cout << time << "; ";
    // }
    // std::cout << "milisecond" << std::endl;

    //std::cout << tVector[0] << " : " << tVector[1023] << " : " << tVector[2047] << std::endl;
    //
    //std::cout << "testStreams: " << magmaElement.testStreams(tVector, 128, 128, countStreams, 1024, false) << std::endl;
    //
    //std::cout << tVector[0] << " : " << tVector[1023] << " : " << tVector[2047] << std::endl;

    // timeDefault = magmaElement.testDefault(tVector, 128, 128, false);
    // timeDefault = testDefaultTemplate<magmaBlockT, magmaKeySet>(tVector, magmaKeys, 32, 32, true);

    // std::cout << "\ntime default: ";
    // for (auto time : timeDefault) {
    //     std::cout << time << "; ";
    // }
    // std::cout << "milisecond" << std::endl;

    /*std::cout << tVector[0] << " : " << tVector[1] << std::endl;

    std::cout << std::endl << "Test Managed" << std::endl;

    std::vector<float> timeManaged;

    timeManaged = magmaElement.testManaged(tVector, tVector.size(), 16, 16, true);

    std::cout << "\ntime managed: ";
    for (auto time : timeManaged) {
        std::cout << time << "; ";
    }
    std::cout << "milisecond" << std::endl;

    std::cout << tVector[0] << " : " << tVector[1] << std::endl;

    timeManaged = magmaElement.testManaged(tVector, tVector.size(), 16, 16, false);

    std::cout << "\ntime managed: ";
    for (auto time : timeManaged) {
        std::cout << time << "; ";
    }
    std::cout << "milisecond" << std::endl;

    std::cout << tVector[0] << " : " << tVector[1] << std::endl;

    std::cout << std::endl << "Test Pinned" << std::endl;

    std::vector<float> timePinned;

    timePinned = magmaElement.testPinned(tVector, tVector.size(), 16, 16, true);

    std::cout << "\ntime pinned: ";
    for (auto time : timePinned) {
        std::cout << time << "; ";
    }
    std::cout << "milisecond" << std::endl;

    std::cout << tVector[0] << " : " << tVector[1] << std::endl;

    timePinned = magmaElement.testPinned(tVector, tVector.size(), 16, 16, false);

    std::cout << "\ntime pinned: ";
    for (auto time : timePinned) {
        std::cout << time << "; ";
    }
    std::cout << "milisecond" << std::endl;

    std::cout << tVector[0] << " : " << tVector[1] << std::endl;
    */

    /*std::vector<uint8_t> buffer = readFile("C:\\Users\\artio\\Documents\\testFilesForGOST\\1bytes");
    std::cout << "\nbuffer\n" << buffer.data() << std::endl;
    std::vector<uint8_t> result;
    result.resize(buffer.size());

    magmaElement.encryptCuda((unsigned char*)buffer.data(), (unsigned char*)result.data(), buffer.size() / 8);

    std::cout << "\resultEnc\n" << result.data() << std::endl;

    magmaElement.decryptCuda((unsigned char*)result.data(), (unsigned char*)result.data(), buffer.size() / 8);

    std::cout << "\resultDec\n" << result.data() << std::endl;*/

    //magmaElement.testSpeedUnequalBytes();
    //magmaElement.searchBestBlockAndGridSize();

    uint8_t testStringKuz[16] = {
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
    };

    kuznechikByteVector k;
    std::vector<kuznechikByteVector> kVector;

    std::copy(testStringKuz, testStringKuz + 16, k.bytes);
    //tVector.push_back(t);
    for (int i = 0; i < 2048; ++i) {
        kVector.push_back(k);
    }

    uint8_t testKeyBytesKuz[] = { 
        0xef, 0xcd, 0xab, 0x89, 
        0x67, 0x45, 0x23, 0x01, 
        0x10, 0x32, 0x54, 0x76, 
        0x98, 0xba, 0xdc, 0xfe, 
        0x77, 0x66, 0x55, 0x44, 
        0x33, 0x22, 0x11, 0x00, 
        0xff, 0xee, 0xdd, 0xcc, 
        0xbb, 0xaa, 0x99, 0x88 };

    kuznechikKeys testKeyKuz(testKeyBytesKuz);

    // std::cout << 1024 * 1024 * 1024 * 0.5 / sizeof(kuznechikByteVector) << "  " << 1024 * 1024 * 1024 * 0.5 << std::endl;

    kuznechik kuznechikElement(testKeyKuz, 1024*1024*1024*0.5/sizeof(kuznechikByteVector), 512, 1024);

    kuznechikElement.checkEcnAndDec();

    std::cout << kVector[0] << " : " << kVector[1023] << " : " << kVector[2047] << std::endl;

    std::cout << kuznechikElement.testStreams(kVector, 256, 16 * 1024, 2, 512, true) << std::endl;;

    std::cout << kVector[0] << " : " << kVector[1023] << " : " << kVector[2047] << std::endl;

    //kuznechikElement.testSpeedUnequalBytes();
    //kuznechikElement.searchBestBlockAndGridSize();Ну

    /*for (size_t i = 32; i <= 1024; i *= 2) {
        for (size_t j = 32; j <= 1024; j *= 2) {
            testSpeed("C:\\Users\\artio\\Documents\\testFilesForGOST", { 128 * 1024 * 1024, 1024 * 1024 * 1024 }, i, j);
        }
    }*/
    //testSpeedMagma("C:\\Users\\artio\\Documents\\testFilesForGOST", { 8 * 1024 * 1024, 8 * 1024 * 1024 }, 1024, 1024);
    std::cout << "Name test;size;enc;gridSize;blockSize;speedCopyAndEnc;speedEnc" << std::endl;
    //for (size_t j = 128; j < 2048; j = j * 2) {
        for (size_t i = 0; i < 1; ++i) {
            testSpeedKuznechik("C:\\Users\\artio\\Documents\\testFilesForGOST", { 1024 * 1024 * 1024, 1024 * 1024 * 1024 }, 256, 16*1024);
        }
    //}

    return 0;
}