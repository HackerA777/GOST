#pragma once
#pragma once
#include <stdint.h>
#include <inttypes.h>
#include <span>
#include "Structures.hpp"
#include <vector>
#include <atomic>
#include <omp.h>

class Magma {
private:
	halfVectorMagma roundKeys[32];
public:
	Magma();
	Magma(const key& key);
	void processData(std::span<const byteVectorMagma> src, std::span<byteVectorMagma> dest) const;
	void changeKey(const key& key);
	static void prefetch_table();
	void timingAttack(std::span<const byteVectorMagma> src, std::span<byteVectorMagma> dest, std::vector<double>& times) const;
};