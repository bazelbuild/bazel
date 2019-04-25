/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

#include <jni.h>

#include <new>

#include <brotli/encode.h>

namespace {
/* A structure used to persist the encoder's state in between calls. */
typedef struct EncoderHandle {
  BrotliEncoderState* state;

  uint8_t* input_start;
  size_t input_offset;
  size_t input_last;
} EncoderHandle;

/* Obtain handle from opaque pointer. */
EncoderHandle* getHandle(void* opaque) {
  return static_cast<EncoderHandle*>(opaque);
}

}  /* namespace */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new Encoder.
 *
 * Cookie to address created encoder is stored in out_cookie. In case of failure
 * cookie is 0.
 *
 * @param ctx {out_cookie, in_directBufferSize, in_quality, in_lgwin} tuple
 * @returns direct ByteBuffer if directBufferSize is not 0; otherwise null
 */
JNIEXPORT jobject JNICALL
Java_org_brotli_wrapper_enc_EncoderJNI_nativeCreate(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  bool ok = true;
  EncoderHandle* handle = nullptr;
  jlong context[5];
  env->GetLongArrayRegion(ctx, 0, 5, context);
  size_t input_size = context[1];
  context[0] = 0;
  handle = new (std::nothrow) EncoderHandle();
  ok = !!handle;

  if (ok) {
    handle->input_offset = 0;
    handle->input_last = 0;
    handle->input_start = nullptr;

    if (input_size == 0) {
      ok = false;
    } else {
      handle->input_start = new (std::nothrow) uint8_t[input_size];
      ok = !!handle->input_start;
    }
  }

  if (ok) {
    handle->state = BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
    ok = !!handle->state;
  }

  if (ok) {
    int quality = context[2];
    if (quality >= 0) {
      BrotliEncoderSetParameter(handle->state, BROTLI_PARAM_QUALITY, quality);
    }
    int lgwin = context[3];
    if (lgwin >= 0) {
      BrotliEncoderSetParameter(handle->state, BROTLI_PARAM_LGWIN, lgwin);
    }
  }

  if (ok) {
    /* TODO: future versions (e.g. when 128-bit architecture comes)
                     might require thread-safe cookie<->handle mapping. */
    context[0] = reinterpret_cast<jlong>(handle);
  } else if (!!handle) {
    if (!!handle->input_start) delete[] handle->input_start;
    delete handle;
  }

  env->SetLongArrayRegion(ctx, 0, 1, context);

  if (!ok) {
    return nullptr;
  }

  return env->NewDirectByteBuffer(handle->input_start, input_size);
}

/**
 * Push data to encoder.
 *
 * @param ctx {in_cookie, in_operation_out_success, out_has_more_output,
 *             out_has_remaining_input} tuple
 * @param input_length number of bytes provided in input or direct input;
 *                     0 to process further previous input
 */
JNIEXPORT void JNICALL
Java_org_brotli_wrapper_enc_EncoderJNI_nativePush(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx, jint input_length) {
  jlong context[5];
  env->GetLongArrayRegion(ctx, 0, 5, context);
  EncoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  int operation = context[1];
  context[1] = 0;  /* ERROR */
  env->SetLongArrayRegion(ctx, 0, 5, context);

  BrotliEncoderOperation op;
  switch (operation) {
    case 0: op = BROTLI_OPERATION_PROCESS; break;
    case 1: op = BROTLI_OPERATION_FLUSH; break;
    case 2: op = BROTLI_OPERATION_FINISH; break;
    default: return;  /* ERROR */
  }

  if (input_length != 0) {
    /* Still have unconsumed data. Workflow is broken. */
    if (handle->input_offset < handle->input_last) {
      return;
    }
    handle->input_offset = 0;
    handle->input_last = input_length;
  }

  /* Actual compression. */
  const uint8_t* in = handle->input_start + handle->input_offset;
  size_t in_size = handle->input_last - handle->input_offset;
  size_t out_size = 0;
  BROTLI_BOOL status = BrotliEncoderCompressStream(
      handle->state, op, &in_size, &in, &out_size, nullptr, nullptr);
  handle->input_offset = handle->input_last - in_size;
  if (!!status) {
    context[1] = 1;
    context[2] = BrotliEncoderHasMoreOutput(handle->state) ? 1 : 0;
    context[3] = (handle->input_offset != handle->input_last) ? 1 : 0;
    context[4] = BrotliEncoderIsFinished(handle->state) ? 1 : 0;
  }
  env->SetLongArrayRegion(ctx, 0, 5, context);
}

/**
 * Pull decompressed data from encoder.
 *
 * @param ctx {in_cookie, out_success, out_has_more_output,
 *             out_has_remaining_input} tuple
 * @returns direct ByteBuffer; all the produced data MUST be consumed before
 *          any further invocation; null in case of error
 */
JNIEXPORT jobject JNICALL
Java_org_brotli_wrapper_enc_EncoderJNI_nativePull(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  jlong context[5];
  env->GetLongArrayRegion(ctx, 0, 5, context);
  EncoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  size_t data_length = 0;
  const uint8_t* data = BrotliEncoderTakeOutput(handle->state, &data_length);
  context[1] = 1;
  context[2] = BrotliEncoderHasMoreOutput(handle->state) ? 1 : 0;
  context[3] = (handle->input_offset != handle->input_last) ? 1 : 0;
  context[4] = BrotliEncoderIsFinished(handle->state) ? 1 : 0;
  env->SetLongArrayRegion(ctx, 0, 5, context);
  return env->NewDirectByteBuffer(const_cast<uint8_t*>(data), data_length);
}

/**
 * Releases all used resources.
 *
 * @param ctx {in_cookie} tuple
 */
JNIEXPORT void JNICALL
Java_org_brotli_wrapper_enc_EncoderJNI_nativeDestroy(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  jlong context[2];
  env->GetLongArrayRegion(ctx, 0, 2, context);
  EncoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  BrotliEncoderDestroyInstance(handle->state);
  delete[] handle->input_start;
  delete handle;
}

#ifdef __cplusplus
}
#endif
