#include <jni.h>
#include <zstd_internal.h>
#include <zstd_errors.h>


/*
 * Private shim for JNI <-> ZSTD
 */
static size_t JNI_ZSTD_compress(void* dst, size_t dstCapacity,
                          const void* src, size_t srcSize,
                                int compressionLevel,
                                jboolean checksumFlag) {

    ZSTD_CCtx* const cctx = ZSTD_createCCtx();

    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, compressionLevel);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, (checksumFlag == JNI_TRUE));

    size_t const size = ZSTD_compress2(cctx, dst, dstCapacity, src, srcSize);

    ZSTD_freeCCtx(cctx);
    return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    compressUnsafe
 * Signature: (JJJJI)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_compressUnsafe
  (JNIEnv *env, jclass obj, jlong dst_buf_ptr, jlong dst_size, jlong src_buf_ptr, jlong src_size, jint level, jboolean checksumFlag) {
    return JNI_ZSTD_compress((void *)(intptr_t) dst_buf_ptr, (size_t) dst_size, (void *)(intptr_t) src_buf_ptr, (size_t) src_size, (int) level, checksumFlag);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    decompressUnsafe
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_decompressUnsafe
  (JNIEnv *env, jclass obj, jlong dst_buf_ptr, jlong dst_size, jlong src_buf_ptr, jlong src_size) {
    return ZSTD_decompress((void *)(intptr_t) dst_buf_ptr, (size_t) dst_size, (void *)(intptr_t) src_buf_ptr, (size_t) src_size);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    decompressedSize
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_decompressedSize
  (JNIEnv *env, jclass obj, jbyteArray src) {
    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jsize src_size = (*env)->GetArrayLength(env, src);
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E1;
    size = ZSTD_getDecompressedSize(src_buff, (size_t) src_size);
    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E1: return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    getDictIdFromFrame
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_getDictIdFromFrame
  (JNIEnv *env, jclass obj, jbyteArray src) {
    unsigned dict_id = 0;
    jsize src_size = (*env)->GetArrayLength(env, src);
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E1;
    dict_id = ZSTD_getDictID_fromFrame(src_buff, (size_t) src_size);
    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E1: return (jlong) dict_id;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    getDictIdFromFrameBuffer
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_getDictIdFromFrameBuffer
  (JNIEnv *env, jclass obj, jobject src) {
    unsigned dict_id = 0;
    jsize src_size = (*env)->GetDirectBufferCapacity(env, src);
    if (src_size == 0) goto E1;
    char *src_buff = (char*)(*env)->GetDirectBufferAddress(env, src);
    if (src_buff == NULL) goto E1;
    dict_id = ZSTD_getDictID_fromFrame(src_buff, (size_t) src_size);
E1: return (jlong) dict_id;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    getDictIdFromDict
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_getDictIdFromDict
  (JNIEnv *env, jclass obj, jbyteArray src) {
    unsigned dict_id = 0;
    jsize src_size = (*env)->GetArrayLength(env, src);
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E1;
    dict_id = ZSTD_getDictID_fromDict(src_buff, (size_t) src_size);
    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E1: return (jlong) dict_id;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    decompressedDirectByteBufferSize
 * Signature: (Ljava/nio/ByteBuffer;II)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_decompressedDirectByteBufferSize
  (JNIEnv *env, jclass obj, jobject src_buf, jint src_offset, jint src_size) {
    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jsize src_cap = (*env)->GetDirectBufferCapacity(env, src_buf);
    if (src_offset + src_size > src_cap) return ZSTD_error_GENERIC;
    char *src_buf_ptr = (char*)(*env)->GetDirectBufferAddress(env, src_buf);
    if (src_buf_ptr == NULL) goto E1;
    size = ZSTD_getDecompressedSize(src_buf_ptr + src_offset, (size_t) src_size);
E1: return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    compressBound
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_compressBound
  (JNIEnv *env, jclass obj, jlong size) {
    return ZSTD_compressBound((size_t) size);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    isError
 * Signature: (J)I
 */
JNIEXPORT jboolean JNICALL Java_com_github_luben_zstd_Zstd_isError
  (JNIEnv *env, jclass obj, jlong code) {
    return ZSTD_isError((size_t) code) != 0;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    getErrorName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_github_luben_zstd_Zstd_getErrorName
  (JNIEnv *env, jclass obj, jlong code) {
    const char *msg = ZSTD_getErrorName(code);
    return (*env)->NewStringUTF(env, msg);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    getErrorCode
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_getErrorCode
  (JNIEnv *env, jclass obj, jlong code) {
    return ZSTD_getErrorCode((size_t) code);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    loadDictDecompress
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_loadDictDecompress
  (JNIEnv *env, jclass obj, jlong stream, jbyteArray dict, jint dict_size) {
    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jclass clazz = (*env)->GetObjectClass(env, obj);
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (dict_buff == NULL) goto E1;

    size = ZSTD_DCtx_loadDictionary((ZSTD_DCtx *)(intptr_t) stream, dict_buff, dict_size);
E1:
    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    loadFastDictDecompress
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_loadFastDictDecompress
  (JNIEnv *env, jclass obj, jlong stream, jobject dict) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    jclass dict_clazz = (*env)->GetObjectClass(env, dict);
    jfieldID decompress_dict = (*env)->GetFieldID(env, dict_clazz, "nativePtr", "J");
    ZSTD_DDict* ddict = (ZSTD_DDict*)(intptr_t)(*env)->GetLongField(env, dict, decompress_dict);
    if (ddict == NULL) return ZSTD_error_dictionary_wrong;
    return ZSTD_DCtx_refDDict((ZSTD_DCtx *)(intptr_t) stream, ddict);
}


/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    loadDictCompress
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_loadDictCompress
  (JNIEnv *env, jclass obj, jlong stream, jbyteArray dict, jint dict_size) {
    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    jclass clazz = (*env)->GetObjectClass(env, obj);
    void *dict_buff = (*env)->GetPrimitiveArrayCritical(env, dict, NULL);
    if (dict_buff == NULL) goto E1;

    size = ZSTD_CCtx_loadDictionary((ZSTD_CCtx *)(intptr_t) stream, dict_buff, dict_size);
E1:
    (*env)->ReleasePrimitiveArrayCritical(env, dict, dict_buff, JNI_ABORT);
    return size;
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    loadFastDictCompress
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_loadFastDictCompress
  (JNIEnv *env, jclass obj, jlong stream, jobject dict) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    jclass dict_clazz = (*env)->GetObjectClass(env, dict);
    jfieldID compress_dict = (*env)->GetFieldID(env, dict_clazz, "nativePtr", "J");
    ZSTD_CDict* cdict = (ZSTD_CDict*)(intptr_t)(*env)->GetLongField(env, dict, compress_dict);
    if (cdict == NULL) return ZSTD_error_dictionary_wrong;
    return ZSTD_CCtx_refCDict((ZSTD_CCtx *)(intptr_t) stream, cdict);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    setCompressionChecksums
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_setCompressionChecksums
  (JNIEnv *env, jclass obj, jlong stream, jboolean enabled) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    int checksum = enabled ? 1 : 0;
    return ZSTD_CCtx_setParameter((ZSTD_CCtx *)(intptr_t) stream, ZSTD_c_checksumFlag, checksum);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    setCompressionLevel
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_setCompressionLevel
  (JNIEnv *env, jclass obj, jlong stream, jint level) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    return ZSTD_CCtx_setParameter((ZSTD_CCtx *)(intptr_t) stream, ZSTD_c_compressionLevel, level);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Method:    setCompressionWorkers
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_setCompressionWorkers
  (JNIEnv *env, jclass obj, jlong stream, jint workers) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    return ZSTD_CCtx_setParameter((ZSTD_CCtx *)(intptr_t) stream, ZSTD_c_nbWorkers, workers);
}

/*
 * Class:     com_github_luben_zstd_Zstd
 * Methods:   header constants access
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_windowLogMin
  (JNIEnv *env, jclass obj) {
    return ZSTD_WINDOWLOG_MIN;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_windowLogMax
  (JNIEnv *env, jclass obj) {
    return ZSTD_WINDOWLOG_MAX;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_chainLogMin
  (JNIEnv *env, jclass obj) {
    return ZSTD_CHAINLOG_MIN;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_chainLogMax
  (JNIEnv *env, jclass obj) {
    return ZSTD_CHAINLOG_MAX;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_hashLogMin
  (JNIEnv *env, jclass obj) {
    return ZSTD_HASHLOG_MIN;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_hashLogMax
  (JNIEnv *env, jclass obj) {
    return ZSTD_HASHLOG_MAX;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_searchLogMin
  (JNIEnv *env, jclass obj) {
    return ZSTD_SEARCHLOG_MIN;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_searchLogMax
  (JNIEnv *env, jclass obj) {
    return ZSTD_SEARCHLOG_MAX;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_magicNumber
  (JNIEnv *env, jclass obj) {
    return ZSTD_MAGICNUMBER;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_blockSizeMax
  (JNIEnv *env, jclass obj) {
    return ZSTD_BLOCKSIZE_MAX;
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_minCompressionLevel
  (JNIEnv *env, jclass obj) {
    return ZSTD_minCLevel();
}

JNIEXPORT jint JNICALL Java_com_github_luben_zstd_Zstd_maxCompressionLevel
  (JNIEnv *env, jclass obj) {
    return ZSTD_maxCLevel();
}

#define JNI_ZSTD_ERROR(err, name) \
  JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_Zstd_err##name \
    (JNIEnv *env, jclass obj) { \
      return ZSTD_error_##err; \
  }


JNI_ZSTD_ERROR(no_error,                      NoError)
JNI_ZSTD_ERROR(GENERIC,                       Generic)
JNI_ZSTD_ERROR(prefix_unknown,                PrefixUnknown)
JNI_ZSTD_ERROR(version_unsupported,           VersionUnsupported)
JNI_ZSTD_ERROR(frameParameter_unsupported,    FrameParameterUnsupported)
JNI_ZSTD_ERROR(frameParameter_windowTooLarge, FrameParameterWindowTooLarge)
JNI_ZSTD_ERROR(corruption_detected,           CorruptionDetected)
JNI_ZSTD_ERROR(checksum_wrong,                ChecksumWrong)
JNI_ZSTD_ERROR(dictionary_corrupted,          DictionaryCorrupted)
JNI_ZSTD_ERROR(dictionary_wrong,              DictionaryWrong)
JNI_ZSTD_ERROR(dictionaryCreation_failed,     DictionaryCreationFailed)
JNI_ZSTD_ERROR(parameter_unsupported,         ParameterUnsupported)
JNI_ZSTD_ERROR(parameter_outOfBound,          ParameterOutOfBound)
JNI_ZSTD_ERROR(tableLog_tooLarge,             TableLogTooLarge)
JNI_ZSTD_ERROR(maxSymbolValue_tooLarge,       MaxSymbolValueTooLarge)
JNI_ZSTD_ERROR(maxSymbolValue_tooSmall,       MaxSymbolValueTooSmall)
JNI_ZSTD_ERROR(stage_wrong,                   StageWrong)
JNI_ZSTD_ERROR(init_missing,                  InitMissing)
JNI_ZSTD_ERROR(memory_allocation,             MemoryAllocation)
JNI_ZSTD_ERROR(workSpace_tooSmall,            WorkSpaceTooSmall)
JNI_ZSTD_ERROR(dstSize_tooSmall,              DstSizeTooSmall)
JNI_ZSTD_ERROR(srcSize_wrong,                 SrcSizeWrong)
JNI_ZSTD_ERROR(dstBuffer_null,                DstBufferNull)
