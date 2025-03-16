#pragma once

extern "C" {
#include "./libprov/include/prov/err.h"
#include "./libprov/include/prov/num.h"
}

#include <iostream>
#include <cassert>

#include <openssl/core.h>
#include <openssl/core_dispatch.h>
#include <openssl/params.h>
#include <openssl/opensslconf.h>
#include <openssl/core_names.h>

#include "./magmaContext/magmaContext.h"
#include "./magma/magma.cuh"

#include "./kuznechikContext/kuznechikContext.h"
#include "./kuznechik/kuznechik.cuh"

#include "gParams.h"

extern const OSSL_DISPATCH magmaFunctions[];
extern const OSSL_DISPATCH kuznechikFunctions[];

const OSSL_ALGORITHM ghostChipersTable[] = {
	{ "magma", "provider=gost_provider", magmaFunctions, NULL },
	{ "kuznechik", "provider=gost_provider", kuznechikFunctions, NULL },
	{ NULL, NULL, NULL, NULL }
};

struct providerCtxSt {
	OSSL_CORE_HANDLE* handle;
	// struct proverr_functions_st* proverr_handle;
};

// Освобождение ресурсов провайдера и создание нового контекста провайдера
void providerCtxFree(struct providerCtxSt* ctx);
struct providerCtxSt* providerCtxNew(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in);

//int OSSL_provider_init(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in, const OSSL_DISPATCH** out, void** vprovctx);

void gostProvTeardown(void* providerCtx);
const OSSL_ALGORITHM* gostProvOperation(void* providerCtx, int operationId, int* noCache);

// OSSL_DISPATH для провайдера
const OSSL_DISPATCH providerFunctions[] =
{
	{ OSSL_FUNC_PROVIDER_TEARDOWN, (void (*)(void)) gostProvTeardown },
	{ OSSL_FUNC_PROVIDER_QUERY_OPERATION, (void (*)(void))gostProvOperation },
	{ 0, NULL }
};