#include "magmaContext.h"

const OSSL_DISPATCH magmaFunctions[] = {
    { OSSL_FUNC_CIPHER_NEWCTX, (void (*)(void))magmaNewCtx },
    { OSSL_FUNC_CIPHER_FREECTX, (void (*)(void))magmaFreeCtx },
    { OSSL_FUNC_CIPHER_ENCRYPT_INIT, (void (*)(void))magmaEncryptInit },
    { OSSL_FUNC_CIPHER_DECRYPT_INIT, (void (*)(void))magmaDecryptInit},
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

void* magmaNewCtx(void* provCtx) {
    struct magmaCtxSt *ctx = new magmaCtxSt;
    if (ctx != nullptr) {
        std::memset(ctx, 0, sizeof(*ctx));
        ctx->provCtx = (providerCtxSt*)provCtx;
        ctx->keyL = DEFAULT_KEYLENGTH;
        ctx->blockSize = BLOCKSIZE;
    }
    return ctx;
}

void magmaFreeCtx(void* magmaCtx) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    ctx->provCtx = nullptr;
    delete ctx;
}

int magmaEncryptInit(void* magmaCtx, const unsigned char* key, size_t keyLen, 
                            const unsigned char* iv, size_t ivLen, const OSSL_PARAM params[]) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    if (key != nullptr) {
        ctx->blockSize = BLOCKSIZE;
        delete ctx->key;
        ctx->key = new unsigned char[ctx->keyL];
        std::copy_n(key, ctx->keyL, ctx->key);

        ctx->bufferSize = 4096;
        delete ctx->buffer;
        ctx->buffer = new unsigned char[ctx->bufferSize];
        //std::copy_n(iv, 8, ctx->buffer);

        delete ctx->buffer2;
        ctx->buffer2 = new unsigned char[ctx->bufferSize];
        //std::copy_n(iv, 8, ctx->buffer2);

        ctx->keyL = keyLen;
        //std::cout << "key: " << key << " keyLen: " << keyLen << std::endl;
        ctx->mgm.changeKey(key);
        // std::cout << "key after change: " << mgm.g

        ctx->cudaGridSize = 512;
        ctx->mgm.setGridSize(ctx->cudaGridSize);
        ctx->cudaBlockSize = 512;
        ctx->mgm.setBlockSize(ctx->cudaBlockSize);

        ctx->enc = true;
    }
    return 1;
}

int magmaDecryptInit(void* magmaCtx, const unsigned char* key, size_t keyLen, 
                            const unsigned char* iv, size_t ivLen, const OSSL_PARAM params[]) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    if (key != nullptr) {
        ctx->blockSize = BLOCKSIZE;

        delete ctx->key;
        ctx->key = new unsigned char[ctx->keyL];
        //std::copy_n(key, ctx->keyL, ctx->key);

        ctx->bufferSize = 4096;
        delete ctx->buffer;
        ctx->buffer = new unsigned char[ctx->bufferSize];
        //std::copy_n(iv, ivLen, ctx->buffer);

        delete ctx->buffer2;
        ctx->buffer2 = new unsigned char[ctx->bufferSize];

        ctx->keyL = keyLen;
        ctx->mgm.changeKey(key);

        ctx->cudaGridSize = 512;
        ctx->mgm.setGridSize(ctx->cudaGridSize);
        ctx->cudaBlockSize = 512;
        ctx->mgm.setBlockSize(ctx->cudaBlockSize);

        ctx->enc = false;
    }
    return 1;
}

int magmaUpdate(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize, const unsigned char* in, size_t inL) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;

    //if (ctx->bufferSize > inL){
    //    ctx->bufferSize = inL;
    //}

    size_t blockSize = ctx->blockSize;
    size_t processed = 0;
    //size_t* partialBlockLen = &(ctx->partialBlockLen);
    //std::cout << "result: " << ctx->bufferSize << " size: " << std::endl;
    unsigned char result[ctx->bufferSize];
    //std::cout << "result: " << *result << " size: " << std::endl;

    //std::cout << "inL: " << inL << std::endl;

    //*partialBlockLen += inL;
    ctx->partialBlockLen += inL;
    if (ctx->enc){
        for (size_t i = 0; i < inL / ctx->bufferSize; ++i)
        {
            //std::cout << "1 i: " << i << std::endl;
            std::copy_n(in + i * ctx->bufferSize, ctx->bufferSize, (unsigned char*)&ctx->buffer[0]);
            //std::cout << "Buffer: " << ctx->buffer << " size: " << ctx->bufferSize << std::endl;
            
            ctx->mgm.encryptCuda((uint8_t*)ctx->buffer, (uint8_t*)result, ctx->mgm.getKeys(), ctx->bufferSize / BLOCKSIZE);
            
            //std::cout << "Buffer after encrypt: " << result << std::endl;
            //ctx->ivu += 0x04;
            std::copy_n(result, ctx->bufferSize, out + i * ctx->bufferSize);
            processed += ctx->bufferSize;
            //std::cout << "partialBlockLen " << ctx->partialBlockLen << std::endl;
            ctx->partialBlockLen -= ctx->bufferSize;
            //std::cout << "partialBlockLen after " << ctx->partialBlockLen << std::endl;
            ctx->last += ctx->bufferSize;
        }
        //std::cout << out << std::endl;
    }
    else{
        //std::cout << "dec" << std::endl;
        for (size_t i = 0; i < inL / ctx->bufferSize; ++i)
        {
            //std::cout << "1 i: " << i << std::endl;
            std::copy_n(in + i * ctx->bufferSize, ctx->bufferSize, (unsigned char*)&ctx->buffer[0]);
            //std::cout << "Buffer: " << ctx->buffer << " size: " << ctx->bufferSize << std::endl;
            
            ctx->mgm.decryptCuda((uint8_t*)ctx->buffer, (uint8_t*)result, ctx->mgm.getKeys(), ctx->bufferSize / BLOCKSIZE);
            
            //std::cout << "Buffer after encrypt: " << result << std::endl;
            //ctx->ivu += 0x04;
            std::copy_n(result, ctx->bufferSize, out + i * ctx->bufferSize);
            processed += ctx->bufferSize;
            //std::cout << "partialBlockLen " << ctx->partialBlockLen << std::endl;
            ctx->partialBlockLen -= ctx->bufferSize;
            //std::cout << "partialBlockLen after " << ctx->partialBlockLen << std::endl;
            ctx->last += ctx->bufferSize;
        }
    }
    std::cout << "line 155" << std::endl;
    std::copy_n(in + processed, inL % ctx->bufferSize, ctx->buffer2);
    *outL = processed;
    std::cout << "2: " << std::endl;
    return 1;
}

unsigned char* paddingPKCS(unsigned char* src, size_t gCount) {
    unsigned char result[BLOCKSIZE];

    for(size_t i = 0; i < gCount; ++i){
        result[i] = src[i];
    }

    for(size_t i = gCount; i < BLOCKSIZE; ++i){
        result[i] = BLOCKSIZE - gCount;
    }

    return result;
}

int magmaFinal(void* magmaCtx, unsigned char* out, size_t* outL, size_t outSize) {
    //std::cout << "FINAL: " << std::endl;
    //size_t bufferSize = 8;
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    size_t blockSize = ctx->blockSize;
    size_t partialBlockLen = ctx->partialBlockLen;
    unsigned char result[ctx->bufferSize];
    size_t current_blocks = partialBlockLen / BLOCKSIZE;
    
    std::cout << "partialBlockLen: " << partialBlockLen << std::endl;

    //ctx->mgm.encryptCuda(ctx->buffer2, result, ctx->ivu);
    //ctx->ivu += 0x04;
    if(ctx->enc){
        if(partialBlockLen % BLOCKSIZE != 0){
            for(size_t i = partialBlockLen; i < partialBlockLen + BLOCKSIZE * 2; ++i){
                ctx->buffer2[i] = (BLOCKSIZE - partialBlockLen % BLOCKSIZE) + BLOCKSIZE;
            }
            current_blocks += 2;
        }
        else{
            for(size_t i = 0; i < BLOCKSIZE; ++i){
                ctx->buffer2[i] = BLOCKSIZE;
            }
            current_blocks++;
        } 
        ctx->mgm.encryptCuda((uint8_t*)ctx->buffer2, (uint8_t*)result, ctx->mgm.getKeys(), current_blocks);
        std::copy_n(result, current_blocks * BLOCKSIZE, out);
        std::cout  << "current_blocks" << current_blocks * BLOCKSIZE << std::endl;
        *outL = current_blocks * BLOCKSIZE;
    }
    else{
        //std::cout << "line 207" << std::endl;
        ctx->mgm.decryptCuda((uint8_t*)ctx->buffer2, (uint8_t*)result, ctx->mgm.getKeys(), current_blocks);

        //std::cout << "line 210" << std::endl;

        size_t deleteBytes = result[current_blocks * BLOCKSIZE - 1];

        std::copy_n(result, current_blocks * BLOCKSIZE - deleteBytes, out);
        //std::cout << "line 215" << std::endl;
        //std::cout << "deleteBytes: " << deleteBytes << std::endl;
        *outL = current_blocks * BLOCKSIZE - deleteBytes;
    }
    //std::cout << "FINAL AFTER ENCRYPT: " << std::endl;
    //std::copy_n(result, current_blocks * BLOCKSIZE, out);
    //std::cout << "FINAL AFTER COPY: " << partialBlockLen << std::endl;
    //*outL = current_blocks * BLOCKSIZE;

    return 1;
}

const OSSL_PARAM* magmaGetTableParams(void* provCtx)
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


int magmaGetParams(OSSL_PARAM params[]) {
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

int magmaGetCtxParams(void* magmaCtx, OSSL_PARAM params[]) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    int ok = 1;

    if (ctx->keyL > 0)
    {
        OSSL_PARAM* p;
        for (p = params; p->key != NULL; p++)
            switch (gostParamsParse(p->key))
            {
            case V_PARAM_keylen:
                ok &= provnum_set_size_t(p, ctx->keyL) >= 0;
                break;
            }
    }

    return ok;
}

int magmaSetCtxParams(void* magmaCtx, const OSSL_PARAM params[]) {
    struct magmaCtxSt* ctx = (magmaCtxSt*)magmaCtx;
    const OSSL_PARAM* p;

    if ((p = OSSL_PARAM_locate_const(params, "key")) != NULL) {
        unsigned char key[DEFAULT_KEYLENGTH];
        size_t keylen = sizeof(key);
        memcpy(ctx->key, key, sizeof(ctx->key));
    }

    return 1;
}

const OSSL_PARAM* magmaGetTableCtxParams(void* magmaCtx, void* provCtx) {
    static const OSSL_PARAM table[] =
    {
        { S_PARAM_blocksize, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { S_PARAM_keylen, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { NULL, 0, NULL, 0, 0 },
    };

    return table;
}

const OSSL_PARAM* magmaSetTableCtxParams(void* magmaCtx, void* provCtx) {
    static const OSSL_PARAM table[] =
    {
        { S_PARAM_blocksize, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { S_PARAM_keylen, OSSL_PARAM_UNSIGNED_INTEGER, NULL, sizeof(size_t), 0 },
        { NULL, 0, NULL, 0, 0 },
    };

    return table;
}

