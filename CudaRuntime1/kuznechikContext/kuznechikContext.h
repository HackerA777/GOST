#pragma once
#include <openssl/core.h>
#include <openssl/core_dispatch.h>
#include <openssl/params.h>

#include "../kuznechik/kuznechik.cuh"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstddef>


#include "../gParams.h"
#include "../libprov/include/prov/num.h"
#include "../libprov/include/prov/err.h"

#define DEFAULT_KEYLENGTH 32
#define BLOCKSIZE 8

static OSSL_FUNC_provider_query_operation_fn kuznechikProvOperation; // сам провайдер
static OSSL_FUNC_provider_get_params_fn kuznechikProvGetParams;
static OSSL_FUNC_provider_get_reason_strings_fn kuznechikProvGetReasonStrings;

OSSL_FUNC_cipher_newctx_fn kuznechikNewCtx;
OSSL_FUNC_cipher_freectx_fn kuznechikFreeCtx;
OSSL_FUNC_cipher_encrypt_init_fn kuznechikEncryptInit;
OSSL_FUNC_cipher_decrypt_init_fn kuznechikDecryptInit;
OSSL_FUNC_cipher_update_fn kuznechikUpdate;
OSSL_FUNC_cipher_final_fn kuznechikFinal;
OSSL_FUNC_cipher_get_params_fn kuznechikGetParams;
OSSL_FUNC_cipher_gettable_params_fn kuznechikGetTableParams;
OSSL_FUNC_cipher_set_ctx_params_fn kuznechikSetCtxParams;
OSSL_FUNC_cipher_get_ctx_params_fn kuznechikGetCtxParams;
OSSL_FUNC_cipher_settable_ctx_params_fn kuznechikSetTableCtxParams;
OSSL_FUNC_cipher_gettable_ctx_params_fn kuznechikGetTableCtxParams;

struct providerCtxSt;

struct kuznechikCtxSt {
    struct providerCtxSt* provCtx;

    size_t keyLen;
    size_t bufferSize;

    unsigned char* key;
    unsigned char* buffer;
    size_t blockSize;

    std::vector<kuznechikByteVector> buffer2;
    size_t addedBytesForBuffer;
    size_t partialBlockLen;
    kuznechik kzk;
    bool enc;
    size_t last; 
    size_t cudaGridSize;
    size_t cudaBlockSize;
};

void* kuznechikNewCtx(void* provCtx);

void kuznechikFreeCtx(void* kuznechikCtx);

int kuznechikEncryptInit(void* kuznechikCtx, const unsigned char* key, size_t keyLen, 
            const unsigned char* iv, size_t ivLen, 
            const OSSL_PARAM params[]);
int kuznechikDecryptInit(void* kuznechikCtx, const unsigned char* key, size_t keyLen, 
            const unsigned char* iv, size_t ivLen, 
            const OSSL_PARAM params[]);

int kuznechikUpdate(void* kuznechikCtx, unsigned char* out, size_t* outLen, size_t outSize, 
            const unsigned char* in, size_t inLen);

int kuznechikFinal(void* kuznechikCtx, unsigned char* out, size_t* outLen, size_t outSize);

int kuznechikGetParams(OSSL_PARAM params[]);

int kuznechikGetCtxParams(void* kuznechikCtx, OSSL_PARAM params[]);

int kuznechikSetCtxParams(void* kuznechikCtx, const OSSL_PARAM params[]);

const OSSL_PARAM* kuznechikGetTableCtxParams(void* kuznechikCtx, void* provCtx);

const OSSL_PARAM* kuznechikSetTableCtxParams(void* kuznechikCtx, void* provCtx);

extern const OSSL_DISPATCH kuznechikFunctions[];
