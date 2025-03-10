#pragma once
#include <openssl/core.h>
#include <openssl/core_dispatch.h>

#include "magma.cuh"
#include <iostream>
#include <vector>

#include "gParams.h"
#include "./num.h"
#include "./err.h"

#define DEFAULT_KEYLENGTH 32    // bytes
#define BLOCKSIZE 8             // bytes


static OSSL_FUNC_provider_query_operation_fn magmaProvOperation; // сам провайдер
static OSSL_FUNC_provider_get_params_fn magmaProvGetParams;
static OSSL_FUNC_provider_get_reason_strings_fn magmaProvGetReasonStrings;

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

static void* magmaNewCtx(void* provCtx);

static void magmaFreeCtx(void* magmaCtx);

//static int magmaOperationInit(void* magmaCtx, const unsigned char* key, size_t keyL, 
//								const unsigned char* buffer, size_t bufferSize, const OSSL_PARAM* params[]);
static int magmaEncryptInit(void* magmaCtx, const unsigned char* key,
	size_t keyLen, const unsigned char* iv,
	size_t ivLen, const OSSL_PARAM params[]);
static int magmaDecryptInit(void* magmaCtx, const unsigned char* key,
	size_t keyLen, const unsigned char* iv,
	size_t ivLen, const OSSL_PARAM params[]);

static int magmaUpdate(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize, const unsigned char* in, size_t inL);

static int magmaFinal(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize);

static int magmaGetParams(OSSL_PARAM params[]);

static int magmaGetCtxParams(void* magmaCtx, OSSL_PARAM params[]);

static int magmaSetCtxParams(void* magmaCtx, const OSSL_PARAM params[]);

static const OSSL_PARAM* magmaGetTableCtxParams(void* magmaCtx, void* provCtx);

static const OSSL_PARAM* magmaSetTableCtxParams(void* magmaCtx, void* provCtx);

static OSSL_FUNC_cipher_newctx_fn magmaNewCtx;
static OSSL_FUNC_cipher_freectx_fn magmaFreeCtx;
static OSSL_FUNC_cipher_encrypt_init_fn magmaEncryptInit;
static OSSL_FUNC_cipher_decrypt_init_fn magmaDecryptInit;
static OSSL_FUNC_cipher_update_fn magmaUpdate;
static OSSL_FUNC_cipher_final_fn magmaFinal;
static OSSL_FUNC_cipher_get_params_fn magmaGetParams;
static OSSL_FUNC_cipher_gettable_params_fn magmaGetTableParams;
static OSSL_FUNC_cipher_set_ctx_params_fn magmaSetCtxParams;
static OSSL_FUNC_cipher_get_ctx_params_fn magmaGetCtxParams;
static OSSL_FUNC_cipher_settable_ctx_params_fn magmaSetTableCtxParams;
static OSSL_FUNC_cipher_gettable_ctx_params_fn magmaGetTableCtxParams;

extern const OSSL_DISPATCH magmaFunctions[] = {
	{ OSSL_FUNC_CIPHER_NEWCTX, (void (*)(void))magmaNewCtx },
	{ OSSL_FUNC_CIPHER_FREECTX, (void (*)(void))magmaFreeCtx },
	{ OSSL_FUNC_CIPHER_ENCRYPT_INIT, (void (*)(void))magmaEncryptInit },
	{ OSSL_FUNC_CIPHER_DECRYPT_INIT, (void (*)(void))magmaDecryptInit },
	{ OSSL_FUNC_CIPHER_UPDATE, (void (*)(void))magmaUpdate },
	{ OSSL_FUNC_CIPHER_FINAL, (void (*)(void))magmaFinal },
	{ OSSL_FUNC_CIPHER_GET_PARAMS, (void (*)(void))magmaGetParams },
	{ OSSL_FUNC_CIPHER_GETTABLE_PARAMS, (void (*)(void))magmaGetTableParams },
	{ OSSL_FUNC_CIPHER_GET_CTX_PARAMS, (void (*)(void))magmaGetCtxParams },
	{ OSSL_FUNC_CIPHER_GETTABLE_CTX_PARAMS, (void (*)(void))magmaGetTableCtxParams },
	{ OSSL_FUNC_CIPHER_SET_CTX_PARAMS, (void (*)(void))magmaSetCtxParams },
	{ OSSL_FUNC_CIPHER_SETTABLE_CTX_PARAMS, (void (*)(void))magmaSetTableCtxParams },
	{ 0, NULL }
};