#pragma once
#include <iostream>
#include <vector>
#include "../structures/structures.cuh"
#include "../magma/magma.cuh"
#include "../kuznechik/kuznechik.cuh"

//template<typename T, typename K> std::vector<float> testDefaultTemplate(std::vector<T>& data, const K keys, const size_t blockSize, const size_t gridSize, const bool encryptStatus);
//template<typename T> std::vector<float> testPinned(std::vector<T>& data, const size_t blockSize, const size_t gridSize, const bool encryptStatus);
//template<typename T> std::vector<float> testManaged(std::vector<T>& data, const size_t blokSize, const size_t gridSize, const bool encryptStatus);
//template<typename T, typename K> std::vector<float> testDefaultTemplate(std::vector<T>& data, const K keys, const size_t blockSize, const size_t gridSize, const bool encryptStatus) {
//    cudaCheck(cudaGetLastError());
//    std::vector<float> time{ 0, 0 };
//
//    const size_t countBlocks = data.size();
//    const size_t dataSize = countBlocks * sizeof(T);
//
//    cuda_ptr<K> dev_keys;
//
//    if (std::is_same<K, kuznechikByteVector>::value) {
//        cuda_ptr<K> dev_keys = cuda_alloc<K>(10);
//    }
//    else {
//        cuda_ptr<K> dev_keys = cuda_alloc<K>();
//    }
//    cuda_ptr<T[]> dev_blocks = cuda_alloc<T[]>(countBlocks);
//
//    cudaEvent_t startEnc, stopEnc, startCopyAndEnc, stopCopyAndEnc;
//    cudaCheck(cudaEventCreate(&startEnc));
//    cudaCheck(cudaEventCreate(&stopEnc));
//    cudaCheck(cudaEventCreate(&startCopyAndEnc));
//    cudaCheck(cudaEventCreate(&stopCopyAndEnc));
//
//    if (encryptStatus) {
//
//        //std::cout << "Data before encrypt: " << data.data() << std::endl;
//
//        cudaCheck(cudaEventRecord(startCopyAndEnc));
//        //_Thrd_sleep_for(10000);
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaMemcpyAsync(dev_keys.get(), keys.keys, sizeof(K), cudaMemcpyHostToDevice));
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(startEnc));
//
//        cudaCheck(cudaGetLastError());
//
//        if (std::is_same<T, magmaBlockT>::value) {
//            encrypt << < blockSize, gridSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
//        }
//        else {
//            encryptKuz << < blockSize, gridSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
//        }
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(stopEnc));
//
//        cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(stopCopyAndEnc));
//
//        cudaCheck(cudaGetLastError());
//
//        //std::cout << "Data after encrypt: " << data.data() << std::endl;
//    }
//    else {
//        //std::cout << "Data before decrypt: " << data.data() << std::endl;
//
//        cudaCheck(cudaEventRecord(startCopyAndEnc));
//
//        cudaCheck(cudaMemcpyAsync(dev_keys.get(), keys.keys, sizeof(K), cudaMemcpyHostToDevice));
//
//        cudaCheck(cudaMemcpyAsync(dev_blocks.get(), data.data(), dataSize, cudaMemcpyHostToDevice));
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(startEnc));
//
//        if (std::is_same<T, magmaBlockT>::value) {
//            decrypt << < blockSize, gridSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
//        }
//        else {
//            decryptKuz << < blockSize, gridSize >> > (*dev_keys, dev_blocks.get(), countBlocks);
//        }
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(stopEnc));
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaMemcpyAsync(data.data(), dev_blocks.get(), dataSize, cudaMemcpyDeviceToHost));
//
//        cudaCheck(cudaGetLastError());
//
//        cudaCheck(cudaEventRecord(stopCopyAndEnc));
//
//        cudaCheck(cudaGetLastError());
//
//        //std::cout << "Data after decrypt: " << data.data() << std::endl;
//    }
//
//    cudaCheck(cudaEventSynchronize(stopCopyAndEnc));
//    cudaCheck(cudaGetLastError());
//    cudaEventElapsedTime(&time[0], startCopyAndEnc, stopCopyAndEnc);
//    cudaEventElapsedTime(&time[1], startEnc, stopEnc);
//
//    cudaCheck(cudaEventDestroy(startCopyAndEnc));
//    cudaCheck(cudaEventDestroy(stopCopyAndEnc));
//    cudaCheck(cudaEventDestroy(startEnc));
//    cudaCheck(cudaEventDestroy(stopEnc));
//
//    return time;
//}
