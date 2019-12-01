#include <jni.h>
#include <zstd_internal.h>
#include <zstd_errors.h>
#include <stdlib.h>
#include <stdint.h>

/* field IDs can't change in the same VM */
static jfieldID src_pos_id;
static jfieldID dst_pos_id;

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    recommendedCOutSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_recommendedCOutSize
  (JNIEnv *env, jclass obj) {
    return (jint) ZSTD_CStreamOutSize();
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    createCStream
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_github_luben_zstd_ZstdOutputStream_createCStream
  (JNIEnv *env, jclass obj) {
    return (jlong)(intptr_t) ZSTD_createCStream();
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    freeCStream
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_freeCStream
  (JNIEnv *env, jclass obj, jlong stream) {
    return ZSTD_freeCStream((ZSTD_CStream *)(intptr_t) stream);
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    resetCStream
 * Signature: (JII)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_resetCStream
  (JNIEnv *env, jclass obj, jlong stream) {
    jclass clazz = (*env)->GetObjectClass(env, obj);
    src_pos_id = (*env)->GetFieldID(env, clazz, "srcPos", "J");
    dst_pos_id = (*env)->GetFieldID(env, clazz, "dstPos", "J");
    return ZSTD_CCtx_reset((ZSTD_CStream *)(intptr_t) stream, ZSTD_reset_session_only);
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    compressStream
 * Signature: (J[BI[BI)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_compressStream
  (JNIEnv *env, jclass obj, jlong stream, jbyteArray dst, jint dst_size, jbyteArray src, jint src_size) {

    size_t size = (size_t)(0-ZSTD_error_memory_allocation);

    size_t src_pos = (size_t) (*env)->GetLongField(env, obj, src_pos_id);
    void *dst_buff = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (dst_buff == NULL) goto E1;
    void *src_buff = (*env)->GetPrimitiveArrayCritical(env, src, NULL);
    if (src_buff == NULL) goto E2;

    ZSTD_outBuffer output = { dst_buff, dst_size, 0 };
    ZSTD_inBuffer input = { src_buff, src_size, src_pos };

    size = ZSTD_compressStream2((ZSTD_CStream *)(intptr_t) stream, &output, &input, ZSTD_e_continue);

    (*env)->ReleasePrimitiveArrayCritical(env, src, src_buff, JNI_ABORT);
E2: (*env)->ReleasePrimitiveArrayCritical(env, dst, dst_buff, 0);
    (*env)->SetLongField(env, obj, src_pos_id, input.pos);
    (*env)->SetLongField(env, obj, dst_pos_id, output.pos);
E1: return (jint) size;
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    endStream
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_endStream
  (JNIEnv *env, jclass obj, jlong stream, jbyteArray dst, jint dst_size) {

    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    void *dst_buff = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (dst_buff != NULL) {
        ZSTD_outBuffer output = { dst_buff, dst_size, 0 };
        ZSTD_inBuffer input = {NULL, 0, 0};
        size = ZSTD_compressStream2((ZSTD_CStream *)(intptr_t) stream, &output, &input, ZSTD_e_end);
        (*env)->ReleasePrimitiveArrayCritical(env, dst, dst_buff, 0);
        (*env)->SetLongField(env, obj, dst_pos_id, output.pos);
    }
    return (jint) size;
}

/*
 * Class:     com_github_luben_zstd_ZstdOutputStream
 * Method:    flushStream
 * Signature: (J[BI)I
 */
JNIEXPORT jint JNICALL Java_com_github_luben_zstd_ZstdOutputStream_flushStream
  (JNIEnv *env, jclass obj, jlong stream, jbyteArray dst, jint dst_size) {

    size_t size = (size_t)(0-ZSTD_error_memory_allocation);
    void *dst_buff = (*env)->GetPrimitiveArrayCritical(env, dst, NULL);
    if (dst_buff != NULL) {
        ZSTD_outBuffer output = { dst_buff, dst_size, 0 };
        ZSTD_inBuffer input = {NULL, 0, 0};
        size = ZSTD_compressStream2((ZSTD_CStream *)(intptr_t) stream, &output, &input, ZSTD_e_flush);
        (*env)->ReleasePrimitiveArrayCritical(env, dst, dst_buff, 0);
        (*env)->SetLongField(env, obj, dst_pos_id, output.pos);
    }
    return (jint) size;
}
