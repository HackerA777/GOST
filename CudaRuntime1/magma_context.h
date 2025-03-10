#pragma once
#include <openssl/core.h>
#include <openssl/core_dispatch.h>

#include "magma.cuh"
static OSSL_FUNC_provider_query_operation_fn magma_prov_operation; // сам провайдер
static OSSL_FUNC_provider_get_params_fn magma_prov_get_params;
static OSSL_FUNC_provider_get_reason_strings_fn magma_prov_get_reason_strings;

#define DEFAULT_MAGMA_KEYLENGTH 32    /* bytes */
#define MAGMA_BLOCKSIZE 8             /* bytes*/

struct provider_ctx_st;

struct magma_ctx_st { // контекст самого алгоритма
	struct provider_ctx_st* prov_ctx;

	size_t keyl = DEFAULT_MAGMA_KEYLENGTH;
	size_t buffer_size;
	unsigned char* key;
	unsigned char* buffer;
	size_t added_bytes_for_buffer;
	magma mgm;
	// buffer; keys; кол-во дополненных байт (возможно); добавить классы магма (конструктор по умолчанию)

	size_t keysize; // const 
	size_t keypos; // нафиг не нужен
	bool enc; /* 0 = decrypt, 1 = encrypt */
	int ongoing; // можно убрать
};

static magma_context_set* magma_new_ctx(provider* vprovctx);