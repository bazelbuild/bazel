/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

#include <jni.h>

#include <new>

#include <brotli/decode.h>

namespace {
/* A structure used to persist the decoder's state in between calls. */
typedef struct DecoderHandle {
  BrotliDecoderState* state;

  uint8_t* input_start;
  size_t input_offset;
  size_t input_length;
} DecoderHandle;

/* Obtain handle from opaque pointer. */
DecoderHandle* getHandle(void* opaque) {
  return static_cast<DecoderHandle*>(opaque);
}

}  /* namespace */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new Decoder.
 *
 * Cookie to address created decoder is stored in out_cookie. In case of failure
 * cookie is 0.
 *
 * @param ctx {out_cookie, in_directBufferSize} tuple
 * @returns direct ByteBuffer if directBufferSize is not 0; otherwise null
 */
JNIEXPORT jobject JNICALL
Java_org_brotli_wrapper_dec_DecoderJNI_nativeCreate(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  bool ok = true;
  DecoderHandle* handle = nullptr;
  jlong context[3];
  env->GetLongArrayRegion(ctx, 0, 3, context);
  size_t input_size = context[1];
  context[0] = 0;
  context[2] = 0;
  handle = new (std::nothrow) DecoderHandle();
  ok = !!handle;

  if (ok) {
    handle->input_offset = 0;
    handle->input_length = 0;
    handle->input_start = nullptr;

    if (input_size == 0) {
      ok = false;
    } else {
      handle->input_start = new (std::nothrow) uint8_t[input_size];
      ok = !!handle->input_start;
    }
  }

  if (ok) {
    handle->state = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
    ok = !!handle->state;
  }

  if (ok) {
    /* TODO: future versions (e.g. when 128-bit architecture comes)
                     might require thread-safe cookie<->handle mapping. */
    context[0] = reinterpret_cast<jlong>(handle);
  } else if (!!handle) {
    if (!!handle->input_start) delete[] handle->input_start;
    delete handle;
  }

  env->SetLongArrayRegion(ctx, 0, 3, context);

  if (!ok) {
    return nullptr;
  }

  return env->NewDirectByteBuffer(handle->input_start, input_size);
}

/**
 * Push data to decoder.
 *
 * status codes:
 *  - 0 error happened
 *  - 1 stream is finished, no more input / output expected
 *  - 2 needs more input to process further
 *  - 3 needs more output to process further
 *  - 4 ok, can proceed further without additional input
 *
 * @param ctx {in_cookie, out_status} tuple
 * @param input_length number of bytes provided in input or direct input;
 *                     0 to process further previous input
 */
JNIEXPORT void JNICALL
Java_org_brotli_wrapper_dec_DecoderJNI_nativePush(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx, jint input_length) {
  jlong context[3];
  env->GetLongArrayRegion(ctx, 0, 3, context);
  DecoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  context[1] = 0;  /* ERROR */
  context[2] = 0;
  env->SetLongArrayRegion(ctx, 0, 3, context);

  if (input_length != 0) {
    /* Still have unconsumed data. Workflow is broken. */
    if (handle->input_offset < handle->input_length) {
      return;
    }
    handle->input_offset = 0;
    handle->input_length = input_length;
  }

  /* Actual decompression. */
  const uint8_t* in = handle->input_start + handle->input_offset;
  size_t in_size = handle->input_length - handle->input_offset;
  size_t out_size = 0;
  BrotliDecoderResult status = BrotliDecoderDecompressStream(
      handle->state, &in_size, &in, &out_size, nullptr, nullptr);
  handle->input_offset = handle->input_length - in_size;
  switch (status) {
    case BROTLI_DECODER_RESULT_SUCCESS:
      /* Bytes after stream end are not allowed. */
      context[1] = (handle->input_offset == handle->input_length) ? 1 : 0;
      break;

    case BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT:
      context[1] = 2;
      break;

    case BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
      context[1] = 3;
      break;

    default:
      context[1] = 0;
      break;
  }
  context[2] = BrotliDecoderHasMoreOutput(handle->state) ? 1 : 0;
  env->SetLongArrayRegion(ctx, 0, 3, context);
}

/**
 * Pull decompressed data from decoder.
 *
 * @param ctx {in_cookie, out_status} tuple
 * @returns direct ByteBuffer; all the produced data MUST be consumed before
 *          any further invocation; null in case of error
 */
JNIEXPORT jobject JNICALL
Java_org_brotli_wrapper_dec_DecoderJNI_nativePull(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  jlong context[3];
  env->GetLongArrayRegion(ctx, 0, 3, context);
  DecoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  size_t data_length = 0;
  const uint8_t* data = BrotliDecoderTakeOutput(handle->state, &data_length);
  bool hasMoreOutput = !!BrotliDecoderHasMoreOutput(handle->state);
  if (hasMoreOutput) {
    context[1] = 3;
  } else if (BrotliDecoderIsFinished(handle->state)) {
    /* Bytes after stream end are not allowed. */
    context[1] = (handle->input_offset == handle->input_length) ? 1 : 0;
  } else {
    /* Can proceed, or more data is required? */
    context[1] = (handle->input_offset == handle->input_length) ? 2 : 4;
  }
  context[2] = hasMoreOutput ? 1 : 0;
  env->SetLongArrayRegion(ctx, 0, 3, context);
  return env->NewDirectByteBuffer(const_cast<uint8_t*>(data), data_length);
}

/**
 * Releases all used resources.
 *
 * @param ctx {in_cookie} tuple
 */
JNIEXPORT void JNICALL
Java_org_brotli_wrapper_dec_DecoderJNI_nativeDestroy(
    JNIEnv* env, jobject /*jobj*/, jlongArray ctx) {
  jlong context[3];
  env->GetLongArrayRegion(ctx, 0, 3, context);
  DecoderHandle* handle = getHandle(reinterpret_cast<void*>(context[0]));
  BrotliDecoderDestroyInstance(handle->state);
  delete[] handle->input_start;
  delete handle;
}

#ifdef __cplusplus
}
#endif
