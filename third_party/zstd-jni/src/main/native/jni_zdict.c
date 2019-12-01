#ifndef ZDICT_STATIC_LINKING_ONLY
#define ZDICT_STATIC_LINKING_ONLY
#endif
#include <jni.h>
#include <dictBuilder/zdict.h>
#include <zstd_errors.h>
#include <stdlib.h>
#include <string.h>

JNIEXPORT jlong Java_com_github_luben_zstd_Zstd_trainFromBuffer
  (JNIEnv *env, jclass obj, jobjectArray samples, jbyteArray dictBuffer, jboolean legacy) {
    jsize num_samples = (*env)->GetArrayLength(env, samples);
    size_t *samples_sizes = malloc(sizeof(size_t) * num_samples);
    if (!samples_sizes) {
        jclass eClass = (*env)->FindClass(env, "Ljava/lang/OutOfMemoryError;");
        (*env)->ThrowNew(env, eClass, "native heap");
        goto E1;
    }
    size_t samples_buffer_size = 0;
    for (int i = 0; i < num_samples; i++) {
        jbyteArray sample = (*env)->GetObjectArrayElement(env, samples, i);
        jsize length = (*env)->GetArrayLength(env, sample);
        (*env)->DeleteLocalRef(env, sample);
        samples_sizes[i] = length;
        samples_buffer_size += length;
    }
    void *samples_buffer = malloc(samples_buffer_size);
    if (!samples_buffer) {
        jclass eClass = (*env)->FindClass(env, "Ljava/lang/OutOfMemoryError;");
        (*env)->ThrowNew(env, eClass, "native heap");
        goto E2;
    }
    size_t cursor = 0;
    for (int i = 0; i < num_samples; i++) {
        jbyteArray sample = (*env)->GetObjectArrayElement(env, samples, i);
        jsize length = (*env)->GetArrayLength(env, sample);
        (*env)->GetByteArrayRegion(env, sample, 0, length, (jbyte*)(samples_buffer + cursor));
        (*env)->DeleteLocalRef(env, sample);
        cursor += length;
    }
    size_t dict_capacity = (*env)->GetArrayLength(env, dictBuffer);
    void *dict_buff =  (*env)->GetPrimitiveArrayCritical(env, dictBuffer, NULL);

    size_t size;
    if (legacy == JNI_TRUE) {
        ZDICT_legacy_params_t params;
        memset(&params,0,sizeof(params));
        size = ZDICT_trainFromBuffer_legacy(dict_buff, dict_capacity, samples_buffer, samples_sizes, num_samples, params);
    } else {
        size = ZDICT_trainFromBuffer(dict_buff, dict_capacity, samples_buffer, samples_sizes, num_samples);
    }
    (*env)->ReleasePrimitiveArrayCritical(env, dictBuffer, dict_buff, 0);
    free(samples_buffer);
E2: free(samples_sizes);
E1: return size;
}

JNIEXPORT jlong Java_com_github_luben_zstd_Zstd_trainFromBufferDirect
  (JNIEnv *env, jclass obj, jobject samples, jintArray sampleSizes, jobject dictBuffer, jboolean legacy) {

    void *samples_buffer = (*env)->GetDirectBufferAddress(env, samples);
    void *dict_buff = (*env)->GetDirectBufferAddress(env, dictBuffer);
    size_t dict_capacity = (*env)->GetDirectBufferCapacity(env, dictBuffer);

    /* convert sized from int to size_t */
    jsize num_samples = (*env)->GetArrayLength(env, sampleSizes);
    size_t *samples_sizes = malloc(sizeof(size_t) * num_samples);
    if (!samples_sizes) {
        jclass eClass = (*env)->FindClass(env, "Ljava/lang/OutOfMemoryError;");
        (*env)->ThrowNew(env, eClass, "native heap");
        goto E1;
    }
    jint *sample_sizes_array = (*env)->GetPrimitiveArrayCritical(env, sampleSizes, NULL);
    if (sample_sizes_array == NULL) goto E2;
    for (int i = 0; i < num_samples; i++) {
        samples_sizes[i] = sample_sizes_array[i];
    }
    (*env)->ReleasePrimitiveArrayCritical(env, sampleSizes, sample_sizes_array, JNI_ABORT);

    size_t size;
    if (legacy == JNI_TRUE) {
        ZDICT_legacy_params_t params;
        memset(&params, 0, sizeof(params));
        size = ZDICT_trainFromBuffer_legacy(dict_buff, dict_capacity, samples_buffer, samples_sizes, num_samples, params);
    } else {
        size = ZDICT_trainFromBuffer(dict_buff, dict_capacity, samples_buffer, samples_sizes, num_samples);
    }
E2: free(samples_sizes);
E1: return size;
}

