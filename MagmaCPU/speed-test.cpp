#include <Windows.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <string>
#include "./Magma.hpp"
#include "testData.hpp"
#include <chrono>
#include <string_view>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <cstdlib> 
//#include "./TimingAttackTests/FunctionsForTests/functionsForTests.hpp"

#define GIGABYTE 1024*1024*1024

using duration_t = std::chrono::duration<double>;

void writeVectorToFile(std::vector<double>& vec, const std::string& filename)
{
	std::ofstream file(filename, std::ios::app);

	if (!file.is_open()) {
		throw std::runtime_error("Не удалось открыть файл для записи");
	}

	int i = 0;
	for (const auto& num : vec) {
		file << num << '\n';
	}

	file.close();
}


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

template<typename typeVector>
void generateRandomByteArray(std::vector<typeVector>& dst) {
	std::srand(static_cast<unsigned int>(std::time(0)));
	for (size_t i = 0; i < dst.size(); ++i)
	{
		for (size_t j = 0; j < sizeof(typeVector); ++j)
		{
			dst[i].bytes[j] = static_cast<unsigned char>(std::rand() % 256);
		}
	}
}

template<typename algorithm>
void magmaSpeedTests(const algorithm& alg, size_t testVectorSize, const std::string& filename)
{
	byteVectorMagma testBlockMagma(testDataBytesMagma);
	byteVectorMagma expectedBlockMagma(expectedBlockBytesMagma);

	std::vector<byteVectorMagma> testSrcMagma(testVectorSize, testBlockMagma);
	std::vector<byteVectorMagma> testDestMagma(testVectorSize);

	std::cout << "Тестовые данные: " << printVector(testBlockMagma) << std::endl;
	std::cout << "Ожидаемый результат: " << printVector(expectedBlockMagma) << std::endl;

	alg.processData(testSrcMagma, testDestMagma);

	std::cout << "Реальный результат: " << printVector(testDestMagma[0]) << std::endl;

	std::cout << "----------------------------------------------" << std::endl;

	std::cout << "Запуск теста скорости..." << std::endl;

	std::vector<double> times;

	std::vector<byteVectorMagma> vectorForMagma(GIGABYTE / 8);
	generateRandomByteArray(vectorForMagma);
	std::span<byteVectorMagma, GIGABYTE / 8> tmpMagma(vectorForMagma);

	std::cout << "Сгенирован 1 ГБ данных..." << std::endl;

	std::vector<byteVectorMagma> b(GIGABYTE / 8);
	std::span<byteVectorMagma, GIGABYTE / 8> dest(b);

	auto begin = std::chrono::steady_clock::now();
	alg.processData(tmpMagma, dest);
	auto end = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<duration_t>(end - begin);
	times.push_back(time.count());

	std::cout << "Скорость шифрования: " << 1.0 / times[0] * (1024 * 1024 * 1024) / (1000 * 1000 * 1000) * 8 << " Gb/s" << std::endl;

	writeVectorToFile(times, filename);

	std::cout << "----------------------------------------------" << std::endl;
}

template<typename algorithm>
void kuznechikSpeedTests(const algorithm& alg, size_t testVectorSize, const std::string& filename)
{
	byteVectorKuznechik testBlockKuz(testDataBytesKuz);
	byteVectorKuznechik expectedBlockKuz(expectedBlockBytesKuz);


	std::vector<byteVectorKuznechik> testSrcKuz(testVectorSize, testBlockKuz);
	std::vector<byteVectorKuznechik> testDestKuz(testVectorSize);

	std::cout << "Тестовые данные: " << printVector(testBlockKuz) << std::endl;
	std::cout << "Ожидаемый результат: " << printVector(expectedBlockKuz) << std::endl;

	alg.processData(testSrcKuz, testDestKuz);

	std::cout << "Реальный результат: " << printVector(testDestKuz[0]) << std::endl;

	std::cout << "----------------------------------------------" << std::endl;

	std::cout << "Запуск теста скорости..." << std::endl;
	std::vector<double> timesKuz;

	std::vector<byteVectorKuznechik> a(GIGABYTE / 16);
	generateRandomByteArray(a);
	std::span<byteVectorKuznechik, GIGABYTE / 16> aa2(a);

	std::cout << "Сгенирован 1 ГБ данных..." << std::endl;

	std::vector<byteVectorKuznechik> b(GIGABYTE / 16);
	std::span<byteVectorKuznechik, GIGABYTE / 16> dest(b);


	auto begin = std::chrono::steady_clock::now();
	alg.processData(aa2, dest);
	auto end = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<duration_t>(end - begin);
	timesKuz.push_back(time.count());


	writeVectorToFile(timesKuz, filename);

	std::cout << "----------------------------------------------" << std::endl;
}




int main(int argc, char* argv[])
{
	SetConsoleOutputCP(1251);

	key testKeyMagma(testKeyBytesMagma);
	key testKeyKuz(testKeyBytesKuz);

	Magma M(testKeyMagma);
	magmaSpeedTests(M, 1, "./speed-test.m");
}