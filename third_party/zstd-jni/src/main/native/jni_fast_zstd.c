#include <jni.h>
#include <zstd_internal.h>
#include <zstd_errors.h>
#include <stdint.h>

// They can't change in the same VM
static jfieldID compress_dict = 0;
static jfieldID decompress_dict = 0;
static jfieldID compress_ctx_nativePtr = 0;
static jfieldID decompress_ctx_nativePtr = 0;


/*
 * Class:     com_github_luben_zstd_ZstdDictCompress
 * Method:    init
 * Signature: ([BI)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDictCompress_init
  (JNIEnv *env, jobject obj, jbyteArray dict, jint dict_offset, jint dict_size, jint level)
{
    jclass clazz = (*env)->GetObjectClass(env, obj);
    compress_dict = (*env)->GetFieldID(env, clazz, "nativePtr", "J");
    if (NULL == dict) return;
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (NULL == dict_buff) return;
    ZSTD_CDict* cdict = ZSTD_createCDict(dict_buff + dict_offset, dict_size, level);
    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    if (NULL == cdict) return;
    (*env)->SetLongField(env, obj, compress_dict, (jlong)(intptr_t) cdict);
}

/*
 * Class:     com_github_luben_zstd_ZstdDictCompress
 * Method:    free
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDictCompress_free
  (JNIEnv *env, jobject obj)
{
    if (compress_dict == 0) return;
    ZSTD_CDict* cdict = (ZSTD_CDict*)(intptr_t)(*env)->GetLongField(env, obj, compress_dict);
    if (NULL == cdict) return;
    ZSTD_freeCDict(cdict);
}

/*
 * Class:     com_github_luben_zstd_ZstdDictDecompress
 * Method:    init
 * Signature: ([B)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDictDecompress_init
  (JNIEnv *env, jobject obj, jbyteArray dict, jint dict_offset, jint dict_size)
{
    jclass clazz = (*env)->GetObjectClass(env, obj);
    decompress_dict = (*env)->GetFieldID(env, clazz, "nativePtr", "J");
    if (NULL == dict) return;
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (NULL == dict_buff) return;

    ZSTD_DDict* ddict = ZSTD_createDDict(dict_buff + dict_offset, dict_size);

    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    if (NULL == ddict) return;
    (*env)->SetLongField(env, obj, decompress_dict, (jlong)(intptr_t) ddict);
}

/*
 * Class:     com_github_luben_zstd_ZstdDictDecompress
 * Method:    free
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDictDecompress_free
  (JNIEnv *env, jobject obj)
{
    if (decompress_dict == 0) return;
    ZSTD_DDict* ddict = (ZSTD_DDict*)(intptr_t)(*env)->GetLongField(env, obj, decompress_dict);
    if (NULL == ddict) return;
    ZSTD_freeDDict(ddict);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    decompressFastDict0
 * Signature: ([BI[BIILcom/github/luben/zstd/ZstdDictDecompress;)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_decompressFastDict0
  (JNIEnv *env, jclass obj, jbyteArray dst, jint dst_offset, jbyteArray src, jint src_offset, jint src_length, jobject dict)
{
    if (NULL == dict) return ZSTD_error_dictionary_wrong;
    ZSTD_DDict* ddict = (ZSTD_DDict*)(intptr_t)(*env)->GetLongField(env, dict, decompress_dict);
    if (NULL == ddict) return ZSTD_error_dictionary_wrong;
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_length) return ZSTD_error_srcSize_wrong;

    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jsize dst_size = (*env)->GetArrayLength(env, dst);
    jsize src_size = (*env)->GetArrayLength(env, src);
    if (dst_offset > dst_size) return ZSTD_error_dstSize_tooSmall;
    if (src_size < (src_offset + src_length)) return ZSTD_error_srcSize_wrong;
    dst_size -= dst_offset;
    void *dst_buff = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (dst_buff == NULL) goto E1;
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E2;
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    size = ZSTD_decompress_usingDDict(dctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_length, ddict);
    ZSTD_freeDCtx(dctx);
    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E2: (*env)->ReleasePrimitiveArrayCritical(env, dst, dst_buff, 0);
E1: return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    compressFastDict0
 * Signature: ([BI[BIILcom/github/luben/zstd/ZstdDictCompress;)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_compressFastDict0
  (JNIEnv *env, jclass obj, jbyteArray dst, jint dst_offset, jbyteArray src, jint src_offset, jint src_length, jobject dict) {
    if (NULL == dict) return ZSTD_error_dictionary_wrong;
    ZSTD_CDict* cdict = (ZSTD_CDict*)(intptr_t)(*env)->GetLongField(env, dict, compress_dict);
    if (NULL == cdict) return ZSTD_error_dictionary_wrong;
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_length) return ZSTD_error_srcSize_wrong;


    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jsize dst_size = (*env)->GetArrayLength(env, dst);
    jsize src_size = (*env)->GetArrayLength(env, src);
    if (dst_offset > dst_size) return ZSTD_error_dstSize_tooSmall;
    if (src_size < (src_offset + src_length)) return ZSTD_error_srcSize_wrong;
    dst_size -= dst_offset;
    void *dst_buff = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (dst_buff == NULL) goto E1;
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E2;
    ZSTD_CCtx* ctx = ZSTD_createCCtx();
    size = ZSTD_compress_usingCDict(ctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_length, cdict);
    ZSTD_freeCCtx(ctx);
    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E2: (*env)->ReleasePrimitiveArrayCritical(env, dst, dst_buff, 0);
E1: return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    compressDirectByteBufferFastDict0
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_compressDirectByteBufferFastDict0
  (JNIEnv *env, jclass obj, jobject dst, jint dst_offset, jint dst_size, jobject src, jint src_offset, jint src_size, jobject dict) {
    if (NULL == dict) return ZSTD_error_dictionary_wrong;
    ZSTD_CDict* cdict = (ZSTD_CDict*)(intptr_t)(*env)->GetLongField(env, dict, compress_dict);
    if (NULL == cdict) return ZSTD_error_dictionary_wrong;
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_size) return ZSTD_error_srcSize_wrong;
    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    char *dst_buff = (char*)(*env)->GetDirectBufferAddress(env, dst);
    char *src_buff = (char*)(*env)->GetDirectBufferAddress(env, src);
    ZSTD_CCtx* ctx = ZSTD_createCCtx();
    size = ZSTD_compress_usingCDict(ctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_size, cdict);
    ZSTD_freeCCtx(ctx);
    return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    decompressDirectByteBufferFastDict0
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_decompressDirectByteBufferFastDict0
  (JNIEnv *env, jclass obj, jobject dst, jint dst_offset, jint dst_size, jobject src, jint src_offset, jint src_size, jobject dict)
{
    if (NULL == dict) return ZSTD_error_dictionary_wrong;
    ZSTD_DDict* ddict = (ZSTD_DDict*)(intptr_t)(*env)->GetLongField(env, dict, decompress_dict);
    if (NULL == ddict) return ZSTD_error_dictionary_wrong;
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_size) return ZSTD_error_srcSize_wrong;

    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    char *dst_buff = (char*)(*env)->GetDirectBufferAddress(env, dst);
    char *src_buff = (char*)(*env)->GetDirectBufferAddress(env, src);
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    size = ZSTD_decompress_usingDDict(dctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_size, ddict);
    ZSTD_freeDCtx(dctx);
    return size;
}

/* ================ ZstCompressCtx ============================ */

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_init
  (JNIEnv *env, jobject obj)
{
    if (compress_ctx_nativePtr == 0) {
        jclass clazz = (*env)->GetObjectClass(env, obj);
        compress_ctx_nativePtr = (*env)->GetFieldID(env, clazz, "nativePtr", "J");
    }
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    (*env)->SetLongField(env, obj, compress_ctx_nativePtr, (jlong)(intptr_t) cctx);
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    free
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_free
  (JNIEnv *env, jobject obj)
{
    if (compress_ctx_nativePtr == 0) return;
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    if (NULL == cctx) return;
    ZSTD_freeCCtx(cctx);
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    setLevel0
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_setLevel0
  (JNIEnv *env, jobject obj, jint level)
{
    if (compress_ctx_nativePtr == 0) return;
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    setChecksum0
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_setChecksum0
  (JNIEnv *env, jobject obj, jboolean checksumFlag)
{
    if (compress_ctx_nativePtr == 0) return;
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, (checksumFlag == JNI_TRUE));
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    setContentSize0
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_setContentSize0
  (JNIEnv *env, jobject obj, jboolean contentSizeFlag)
{
    if (compress_ctx_nativePtr == 0) return;
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_contentSizeFlag, (contentSizeFlag == JNI_TRUE));
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    setDictID0
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_setDictID0
  (JNIEnv *env, jobject obj, jboolean dictIDFlag)
{
    if (compress_ctx_nativePtr == 0) return;
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_dictIDFlag, (dictIDFlag == JNI_TRUE));
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    loadCDictFast0
 * Signature: (Lcom/github/luben/zstd/ZstdDictCompress)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_loadCDictFast0
  (JNIEnv *env, jobject obj, jobject dict)
{
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    if (dict == NULL) {
        // remove dictionary
        return ZSTD_CCtx_refCDict(cctx, NULL);
    }
    ZSTD_CDict* cdict = (ZSTD_CDict*)(intptr_t)(*env)->GetLongField(env, dict, compress_dict);
    if (NULL == cdict) return ZSTD_error_dictionary_wrong;
    return ZSTD_CCtx_refCDict(cctx, cdict);
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    loadCDict0
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_loadCDict0
  (JNIEnv *env, jobject obj, jbyteArray dict)
{
    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, obj, compress_ctx_nativePtr);
    if (dict == NULL) {
        // remove dictionary
        return ZSTD_CCtx_loadDictionary(cctx, NULL, 0);
    }
    jsize dict_size = (*env)->GetArrayLength(env, dict);
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (dict_buff == NULL) return ERROR(memory_allocation);
    size_t result = ZSTD_CCtx_loadDictionary(cctx, dict_buff, dict_size);
    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    return result;
}

/*
 * Class:     com_github_luben_zstd_ZstdCompressCtx
 * Method:    compressDirectByteBuffer0
 * Signature: (Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;II)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdCompressCtx_compressDirectByteBuffer0
  (JNIEnv *env, jclass jctx, jobject dst, jint dst_offset, jint dst_size, jobject src, jint src_offset, jint src_size) {
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_size) return ZSTD_error_srcSize_wrong;

    jsize dst_cap = (*env)->GetDirectBufferCapacity(env, dst);
    if (dst_offset + dst_size > dst_cap) return ERROR(dstSize_tooSmall);
    jsize src_cap = (*env)->GetDirectBufferCapacity(env, src);
    if (src_offset + src_size > src_cap) return ERROR(srcSize_wrong);

    char *dst_buff = (char*)(*env)->GetDirectBufferAddress(env, dst);
    if (dst_buff == NULL) return ERROR(memory_allocation);
    char *src_buff = (char*)(*env)->GetDirectBufferAddress(env, src);
    if (src_buff == NULL) return ERROR(memory_allocation);

    ZSTD_CCtx* cctx = (ZSTD_CCtx*)(intptr_t)(*env)->GetLongField(env, jctx, compress_ctx_nativePtr);
    ZSTD_CCtx_reset(cctx, ZSTD_reset_session_only);
    return ZSTD_compress2(cctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_size);
}


/* ================ ZstdDecompressCtx ============================ */

/*
 * Class:     com_github_luben_zstd_ZstdDecompressCtx
 * Method:    init
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDecompressCtx_init
  (JNIEnv *env, jobject obj)
{
    if (decompress_ctx_nativePtr == 0) {
        jclass clazz = (*env)->GetObjectClass(env, obj);
        decompress_ctx_nativePtr = (*env)->GetFieldID(env, clazz, "nativePtr", "J");
    }
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    (*env)->SetLongField(env, obj, decompress_ctx_nativePtr, (jlong)(intptr_t) dctx);
}

/*
 * Class:     com_github_luben_zstd_ZstdDecompressCtx
 * Method:    free
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_github_luben_zstd_ZstdDecompressCtx_free
  (JNIEnv *env, jobject obj)
{
    if (decompress_ctx_nativePtr == 0) return;
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)(intptr_t)(*env)->GetLongField(env, obj, decompress_ctx_nativePtr);
    if (NULL == dctx) return;
    ZSTD_freeDCtx(dctx);
}

/*
 * Class:     com_github_luben_zstd_ZstdDecompressCtx
 * Method:    loadDDictFast0
 * Signature: (Lcom/github/luben/zstd/ZstdDictDecompress)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdDecompressCtx_loadDDictFast0
  (JNIEnv *env, jobject obj, jobject dict)
{
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)(intptr_t)(*env)->GetLongField(env, obj, decompress_ctx_nativePtr);
    if (dict == NULL) {
        // remove dictionary
        return ZSTD_DCtx_refDDict(dctx, NULL);
    }
    ZSTD_DDict* ddict = (ZSTD_DDict*)(intptr_t)(*env)->GetLongField(env, dict, decompress_dict);
    if (NULL == ddict) return ZSTD_error_dictionary_wrong;
    return ZSTD_DCtx_refDDict(dctx, ddict);
}

/*
 * Class:     com_github_luben_zstd_ZstdDecompressCtx
 * Method:    loadDDict0
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdDecompressCtx_loadDDict0
  (JNIEnv *env, jobject obj, jbyteArray dict)
{
    ZSTD_DCtx* dctx = (ZSTD_DCtx*)(intptr_t)(*env)->GetLongField(env, obj, decompress_ctx_nativePtr);
    if (dict == NULL) {
        // remove dictionary
        return ZSTD_DCtx_loadDictionary(dctx, NULL, 0);
    }
    jsize dict_size = (*env)->GetArrayLength(env, dict);
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (dict_buff == NULL) return ERROR(memory_allocation);
    size_t result = ZSTD_DCtx_loadDictionary(dctx, dict_buff, dict_size);
    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    return result;
}


/*
 * Class:     com_github_luben_zstd_ZstdDecompressCtx
 * Method:    decompressDirectByteBuffe0
 * Signature: (Ljava/nio/ByteBuffer;IILjava/nio/ByteBuffer;II)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdDecompressCtx_decompressDirectByteBuffer0
(JNIEnv *env, jclass jctx, jobject dst, jint dst_offset, jint dst_size, jobject src, jint src_offset, jint src_size)
{
    if (NULL == dst) return ZSTD_error_dstSize_tooSmall;
    if (NULL == src) return ZSTD_error_srcSize_wrong;
    if (0 > dst_offset) return ZSTD_error_dstSize_tooSmall;
    if (0 > src_offset) return ZSTD_error_srcSize_wrong;
    if (0 > src_size) return ZSTD_error_srcSize_wrong;

    jsize dst_cap = (*env)->GetDirectBufferCapacity(env, dst);
    if (dst_offset + dst_size > dst_cap) return ERROR(dstSize_tooSmall);
    jsize src_cap = (*env)->GetDirectBufferCapacity(env, src);
    if (src_offset + src_size > src_cap) return ERROR(srcSize_wrong);

    char *dst_buff = (char*)(*env)->GetDirectBufferAddress(env, dst);
    if (dst_buff == NULL) return ERROR(memory_allocation);
    char *src_buff = (char*)(*env)->GetDirectBufferAddress(env, src);
    if (src_buff == NULL) return ERROR(memory_allocation);

    ZSTD_DCtx* dctx = (ZSTD_DCtx*)(intptr_t)(*env)->GetLongField(env, jctx, decompress_ctx_nativePtr);
    ZSTD_DCtx_reset(dctx, ZSTD_reset_session_only);
    return ZSTD_decompressDCtx(dctx, dst_buff + dst_offset, (size_t) dst_size, src_buff + src_offset, (size_t) src_size);
}
