#pragma once

#include "../gParams.h"

extern "C" {
#include "../libprov/include/prov/num.h"
#include "../libprov/include/prov/err.h"
}

#include <openssl/core.h>
#include <openssl/core_dispatch.h>
#include <openssl/params.h>

#include "../magma/magma.cuh"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstddef>

#define DEFAULT_KEYLENGTH 32    // bytes
#define BLOCKSIZE 8             // bytes


static OSSL_FUNC_provider_query_operation_fn magmaProvOperation; // сам провайдер
static OSSL_FUNC_provider_get_params_fn magmaProvGetParams;
static OSSL_FUNC_provider_get_reason_strings_fn magmaProvGetReasonStrings;

OSSL_FUNC_cipher_newctx_fn magmaNewCtx;
OSSL_FUNC_cipher_freectx_fn magmaFreeCtx;
OSSL_FUNC_cipher_encrypt_init_fn magmaEncryptInit;
OSSL_FUNC_cipher_decrypt_init_fn magmaDecryptInit;
OSSL_FUNC_cipher_update_fn magmaUpdate;
OSSL_FUNC_cipher_final_fn magmaFinal;
OSSL_FUNC_cipher_get_params_fn magmaGetParams;
OSSL_FUNC_cipher_gettable_params_fn magmaGetTableParams;
OSSL_FUNC_cipher_set_ctx_params_fn magmaSetCtxParams;
OSSL_FUNC_cipher_get_ctx_params_fn magmaGetCtxParams;
OSSL_FUNC_cipher_settable_ctx_params_fn magmaSetTableCtxParams;
OSSL_FUNC_cipher_gettable_ctx_params_fn magmaGetTableCtxParams;

struct providerCtxSt;

struct magmaCtxSt { // контекст самого алгоритма
	struct providerCtxSt* provCtx;

	size_t keyL = DEFAULT_KEYLENGTH; // размер ключа шифрования
	size_t bufferSize; // размер буффера данных 
	unsigned char* key; // ключ шифрования
	unsigned char* buffer; // данные для шифрования/дешифрования
	size_t blockSize; // размер одного шифруемого блока
	//uint32_t bufferUint; // данные представленные в виде uint32
	std::vector<magmaBlockT> buffer2;    // Внутренний буффер контекста алгоритма, содержащий данные для операций.
	size_t addedBytesForBuffer; // количество дополненных байт до кратности блока
	size_t partialBlockLen; // число, описывающее фактический размер последнего необработанного блкоа данных
	magma mgm;	// класс МАГМА (конструктор по умолчанию)
	bool enc; // 0 = decrypt, 1 = encrypt
	size_t last;    // число, описывающее количество обработанных данных (в байтах).
	size_t gridSize;
};

void* magmaNewCtx(void* provCtx);

void magmaFreeCtx(void* magmaCtx);

//static int magmaOperationInit(void* magmaCtx, const unsigned char* key, size_t keyL, 
//								const unsigned char* buffer, size_t bufferSize, const OSSL_PARAM* params[]);

int magmaEncryptInit(void* magmaCtx, const unsigned char* key,
	size_t keyLen, const unsigned char* iv,
	size_t ivLen, const OSSL_PARAM params[]);
int magmaDecryptInit(void* magmaCtx, const unsigned char* key,
	size_t keyLen, const unsigned char* iv,
	size_t ivLen, const OSSL_PARAM params[]);

int magmaUpdate(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize, const unsigned char* in, size_t inL);

int magmaFinal(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize);

int magmaGetParams(OSSL_PARAM params[]);

int magmaGetCtxParams(void* magmaCtx, OSSL_PARAM params[]);

int magmaSetCtxParams(void* magmaCtx, const OSSL_PARAM params[]);

const OSSL_PARAM* magmaGetTableCtxParams(void* magmaCtx, void* provCtx);

const OSSL_PARAM* magmaSetTableCtxParams(void* magmaCtx, void* provCtx);

extern const OSSL_DISPATCH magmaFunctions[];