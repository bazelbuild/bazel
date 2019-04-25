/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

#include <jni.h>

#include "../common/dictionary.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set data to be brotli dictionary data.
 *
 * @param buffer direct ByteBuffer
 * @returns false if dictionary data was already set; otherwise true
 */
JNIEXPORT jint JNICALL
Java_org_brotli_wrapper_common_CommonJNI_nativeSetDictionaryData(
    JNIEnv* env, jobject /*jobj*/, jobject buffer) {
  jobject buffer_ref = env->NewGlobalRef(buffer);
  if (!buffer_ref) {
    return false;
  }
  uint8_t* data = static_cast<uint8_t*>(env->GetDirectBufferAddress(buffer));
  if (!data) {
    env->DeleteGlobalRef(buffer_ref);
    return false;
  }

  BrotliSetDictionaryData(data);

  const BrotliDictionary* dictionary = BrotliGetDictionary();
  if (dictionary->data != data) {
    env->DeleteGlobalRef(buffer_ref);
  } else {
    /* Don't release reference; it is an intended memory leak. */
  }
  return true;
}

#ifdef __cplusplus
}
#endif
