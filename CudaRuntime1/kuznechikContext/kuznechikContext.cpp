#include "kuznechikContext.h"

const OSSL_DISPATCH kuznechikFunctions[] = {
    { OSSL_FUNC_CIPHER_NEWCTX, (void (*)(void))kuznechikNewCtx },
    { OSSL_FUNC_CIPHER_FREECTX, (void (*)(void))kuznechikFreeCtx },
    { OSSL_FUNC_CIPHER_ENCRYPT_INIT, (void (*)(void))kuznechikEncryptInit},
    { OSSL_FUNC_CIPHER_DECRYPT_INIT, (void (*)(void))kuznechikDecryptInit},
    { OSSL_FUNC_CIPHER_UPDATE, (void (*)(void))kuznechikUpdate },
    { OSSL_FUNC_CIPHER_FINAL, (void (*)(void))kuznechikFinal },
    { OSSL_FUNC_CIPHER_GET_PARAMS, (void (*)(void))kuznechikGetParams },
    { OSSL_FUNC_CIPHER_GETTABLE_PARAMS, (void (*)(void))kuznechikGetTableParams },
    { OSSL_FUNC_CIPHER_GET_CTX_PARAMS, (void (*)(void))kuznechikGetCtxParams },
    { OSSL_FUNC_CIPHER_GETTABLE_CTX_PARAMS, (void (*)(void))kuznechikGetTableCtxParams },
    { OSSL_FUNC_CIPHER_SET_CTX_PARAMS, (void (*)(void))kuznechikSetCtxParams },
    { OSSL_FUNC_CIPHER_SETTABLE_CTX_PARAMS, (void (*)(void))kuznechikSetTableCtxParams },
    { 0, NULL }
};

void* kuznechikNewCtx(void* provCtx) {
    struct kuznechikCtxSt* ctx = new kuznechikCtxSt;
    if (ctx != nullptr) {
        std::memset(ctx, 0, sizeof(*ctx));
        ctx->provCtx = (providerCtxSt*)provCtx;
        ctx->keyLen = DEFAULT_KEYLENGTH;
        ctx->blockSize = BLOCKSIZE_KUZ;
    }
    return ctx;
}

void kuznechikFreeCtx(void* kuznechikCtx) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    ctx->provCtx = nullptr;
    delete ctx;
}

int kuznechikEncryptInit(void* kuznechikCtx, const unsigned char* key, size_t keyLen, 
                            const unsigned char* iv, size_t ivLen, const OSSL_PARAM params[]) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    if (key != nullptr) {
        ctx->blockSize = BLOCKSIZE_KUZ;
        delete ctx->key;
        ctx->key = new unsigned char[ctx->keyLen];
        std::copy_n(key, ctx->keyLen, ctx->key);

        //ctx->bufferSize = 4096;
        ctx->bufferSize = 1024 * 32;
        delete ctx->buffer;
        ctx->buffer = new unsigned char[ctx->bufferSize];

        delete ctx->buffer2;
        ctx->buffer2 = new unsigned char[ctx->bufferSize];

        ctx->keyLen = keyLen;
        ctx->kzk.changeKey((uint8_t *)key);

        ctx->cudaGridSize = 512;
        ctx->kzk.setGridSize(ctx->cudaGridSize);
        ctx->cudaBlockSize = 512;
        ctx->kzk.setBlockSize(ctx->cudaBlockSize);

        ctx->enc = true;
    }
    return 1;
}

int kuznechikDecryptInit(void* kuznechikCtx, const unsigned char* key, size_t keyLen, 
                            const unsigned char* iv, size_t ivLen, const OSSL_PARAM params[]) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    if (key != nullptr) {
        ctx->blockSize = BLOCKSIZE_KUZ;

        delete ctx->key;
        ctx->key = new unsigned char[ctx->keyLen];
        //std::copy_n(key, ctx->keyLen, ctx->key);

        //ctx->bufferSize = 4096;
        ctx->bufferSize = 1024 * 32;
        delete ctx->buffer;
        ctx->buffer = new unsigned char[ctx->bufferSize];

        delete ctx->buffer2;
        ctx->buffer2 = new unsigned char[ctx->bufferSize];

        ctx->keyLen = keyLen;
        ctx->kzk.changeKey(key);

        ctx->cudaGridSize = 512;
        ctx->kzk.setGridSize(ctx->cudaGridSize);
        ctx->cudaBlockSize = 512;
        ctx->kzk.setBlockSize(ctx->cudaBlockSize);

        ctx->enc = false;
    }
    return 1;
}

int kuznechikUpdate(void* kuznechikCtx, unsigned char* out, size_t* outLen, size_t outSize, 
            const unsigned char* in, size_t inLen) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;

    size_t blockSize = ctx->blockSize;
    size_t processed = 0;
    unsigned char result[ctx->bufferSize];

    ctx->partialBlockLen += inLen;
    if(ctx->enc){
        for (size_t i = 0; i < inLen / ctx->bufferSize; ++i)
        {
            std::copy_n(in + i * ctx->bufferSize, ctx->bufferSize, (unsigned char*)&ctx->buffer[0]);

            ctx->kzk.processData((kuznechikByteVector*)ctx->buffer, (kuznechikByteVector*)result, 
                            ctx->bufferSize / BLOCKSIZE_KUZ, ctx->enc);

            std::copy_n((unsigned char*)result, ctx->bufferSize, out + i * ctx->bufferSize);
            processed += ctx->bufferSize;
            ctx->partialBlockLen -= ctx->bufferSize;
            ctx->last += ctx->bufferSize;
        }
    }
    else{
        for(size_t i = 0; i < inLen / ctx->bufferSize; ++i){
            std::copy_n(in + i * ctx->bufferSize, ctx->bufferSize, (unsigned char*)&ctx->buffer[0]);

            ctx->kzk.processData((kuznechikByteVector*)ctx->buffer, (kuznechikByteVector*)result, 
                            ctx->bufferSize / BLOCKSIZE_KUZ, ctx->enc);
            
            std::copy_n((unsigned char*)result, ctx->bufferSize, out + i * ctx->bufferSize);
            processed += ctx->bufferSize;
            ctx->partialBlockLen -= ctx->bufferSize;
            ctx->last += ctx->bufferSize;
        }
    }

    std::copy_n(in + processed, inLen % ctx->bufferSize, ctx->buffer2);
    *outLen = processed;

    return 1;

}

int kuznechikFinal(void* kuznechikCtx, unsigned char* out, size_t* outLen, size_t outSize) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    size_t blockSize = ctx->blockSize;
    size_t partialBlockLen = ctx->partialBlockLen;
    unsigned char result[ctx->bufferSize];
    size_t current_blocks = partialBlockLen / BLOCKSIZE_KUZ;
    
    if(ctx->enc){
        if(partialBlockLen % BLOCKSIZE_KUZ != 0){
            for(size_t i = partialBlockLen; i < partialBlockLen + BLOCKSIZE_KUZ * 2; ++i){
                ctx->buffer2[i] = (BLOCKSIZE_KUZ - partialBlockLen % BLOCKSIZE_KUZ) + BLOCKSIZE_KUZ;
            }
            current_blocks += 2;
        }
        else{
            for(size_t i = 0; i < BLOCKSIZE_KUZ; ++i){
                ctx->buffer2[i] = BLOCKSIZE_KUZ;
            }
            current_blocks++;
        }
        ctx->kzk.processData((kuznechikByteVector *)ctx->buffer2, (kuznechikByteVector*)result, 
                        current_blocks, ctx->enc);
        std::copy_n(result, current_blocks * BLOCKSIZE_KUZ, out);
        *outLen = current_blocks * BLOCKSIZE_KUZ;
    }
    else{
        ctx->kzk.processData((kuznechikByteVector *)ctx->buffer2, (kuznechikByteVector *)result,
                        current_blocks, ctx->enc);
        size_t deleteBytes = result[current_blocks * BLOCKSIZE_KUZ -1];
        std::copy_n(result, current_blocks * BLOCKSIZE_KUZ - deleteBytes, out);

        *outLen = current_blocks * BLOCKSIZE_KUZ - deleteBytes;
    }
    return 1;
}

const OSSL_PARAM* kuznechikGetTableParams(void* provCtx)
{
    static const OSSL_PARAM table[] =
    {
        { "blockSize", OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { "keyLen", OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { "bufferSize", OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { NULL, 0, NULL, 0, 0 },
    };

    return table;
}


int kuznechikGetParams(OSSL_PARAM params[]) {
    OSSL_PARAM* p;
    int ok = 1;

    for (p = params; p->key != NULL; p++)
        switch (gostParamsParse(p->key))
        {
        case V_PARAM_blocksize:
            ok &= provnum_set_size_t(p, 1) >= 0;
            break;
        case V_PARAM_keylen:
            ok &= provnum_set_size_t(p, DEFAULT_KEYLENGTH) >= 0;
            break;
        }
    return ok;
}

int kuznechikGetCtxParams(void* kuznechikCtx, OSSL_PARAM params[]) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    int ok = 1;

    if (ctx->keyLen > 0)
    {
        OSSL_PARAM* p;
        for (p = params; p->key != NULL; p++)
            switch (gostParamsParse(p->key))
            {
            case V_PARAM_keylen:
                ok &= provnum_set_size_t(p, ctx->keyLen) >= 0;
                break;
            }
    }

    return ok;
}

int kuznechikSetCtxParams(void* kuznechikCtx, const OSSL_PARAM params[]) {
    struct kuznechikCtxSt* ctx = (kuznechikCtxSt*)kuznechikCtx;
    const OSSL_PARAM* p;

    if ((p = OSSL_PARAM_locate_const(params, "key")) != NULL) {
        unsigned char key[DEFAULT_KEYLENGTH];
        size_t keyLen = sizeof(key);
        memcpy(ctx->key, key, sizeof(ctx->key));
    }

    return 1;
}

const OSSL_PARAM* kuznechikGetTableCtxParams(void* kuznechikCtx, void* provCtx) {
    static const OSSL_PARAM table[] =
    {
        { S_PARAM_blocksize, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { S_PARAM_keylen, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { NULL, 0, NULL, 0, 0 },
    };

    return table;
}

const OSSL_PARAM* kuznechikSetTableCtxParams(void* kuznechikCtx, void* provCtx) {
    static const OSSL_PARAM table[] =
    {
        { S_PARAM_blocksize, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { S_PARAM_keylen, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { NULL, 0, NULL, 0, 0 },
    };

    return table;
}
