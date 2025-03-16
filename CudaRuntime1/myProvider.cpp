//#include "v_params.h"

#define assertm(exp, msg) assert((void(msg), exp))

// Errors

#define MAGMA_NO_KEYLEN_SET 1
#define MAGMA_ONGOING_OPERATION 2
#define MAGMA_INCORRECT_KEYLEN 3

#include "myProvider.h"

// Provider context

void providerCtxFree(struct providerCtxSt* ctx) {
	/*if (ctx == nullptr)
		proverr_ctx_free(ctx->core_handle)*/
	delete ctx;
} 

struct providerCtxSt* providerCtxNew(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in) {
	struct providerCtxSt* ctx;
	ctx = new providerCtxSt;

	if (ctx != nullptr)
		ctx->handle = (OSSL_CORE_HANDLE *)core;
	else {
		providerCtxFree(ctx);
		ctx = nullptr;
	}

	return ctx;
}

int OSSL_provider_init(const OSSL_CORE_HANDLE* core, const OSSL_DISPATCH* in, const OSSL_DISPATCH** out, void** vprovctx) {
	if ((*vprovctx = providerCtxNew(core, in)) == nullptr)
		return 0;
	*out = providerFunctions;
	return 1;
}

void gostProvTeardown(void* providerCtx) {
	providerCtxFree((providerCtxSt*)providerCtx);
}

/* The function that returns the appropriate algorithm table per operation */
const OSSL_ALGORITHM* gostProvOperation(void* vprovctx, int operationId, int* noCache)
{
	*noCache = 0;
	switch (operationId) {
	case OSSL_OP_CIPHER:
		return ghostChipersTable;
	}
	return NULL;
}

//OSSL_FUNC_provider_get_params_fn magma_prov_get_params;
//OSSL_FUNC_provider_get_reason_strings_fn magma_prov_get_reason_strings;
//
//OSSL_FUNC_provider_get_params_fn kuznechik_prov_get_params;
//OSSL_FUNC_provider_get_reason_strings_fn kuznechik_prov_get_reason_strings;