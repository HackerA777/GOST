#pragma once
#include <iostream>
#include <cassert>

#include <openssl/core.h>
#include <openssl/core_dispatch.h>
#include <openssl/params.h>
#include <openssl/opensslconf.h>
#include <openssl/core_names.h>

#include "./err.h"
#include "./num.h"
#include "./magmaContext.h"
#include "magma.cuh"

extern const OSSL_DISPATCH magmaFunctions[];
extern const OSSL_DISPATCH kuznechikFunctions[];

static const OSSL_ALGORITHM gostChipersTable[] = {
	{ "magma", "provider=gost", magmaFunctions, NULL },
	{ "kuznechik", "provider=gost", kuznechikFunctions, NULL },
	{ NULL, NULL, NULL, NULL }
};

struct providerCtxSt {
	OSSL_CORE_HANDLE* coreHandle;
	// struct proverr_functions_st* proverr_handle;
};

// Освобождение ресурсов провайдера и создание нового контекста провайдера
static void* providerCtxFree(struct providerCtxSt* ctx);
static struct providerCtxSt* providerCtxNew(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in);

int OSSL_provider_init(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in, const OSSL_DISPATCH** out, void** vprovctx);

static void gostProvTeardown(void* provider_ctx);
static const OSSL_ALGORITHM* gostProvOperation(void* providerCtx, int operationId, int* noCache);

// OSSL_DISPATH для провайдера
static const OSSL_DISPATCH providerFunctions[] =
{
	{ OSSL_FUNC_PROVIDER_TEARDOWN, (void (*)(void)) gostProvTeardown },
	{ OSSL_FUNC_PROVIDER_QUERY_OPERATION, (void (*)(void))gostProvOperation },
	{ 0, NULL }
};