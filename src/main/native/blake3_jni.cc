// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include "src/main/cpp/util/logging.h"
#include "third_party/blake3/src/c/blake3.h"

namespace blaze_jni {

blake3_hasher* hasher_ptr(jlong self)
{
  return (blake3_hasher*)self;
}

jbyte* get_byte_array(JNIEnv *env, jbyteArray java_array)
{
  return env->GetByteArrayElements(java_array, nullptr);
}

void release_byte_array(JNIEnv *env, jbyteArray array, jbyte* addr)
{
  return env->ReleaseByteArrayElements(array, addr, 0);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_allocate_1hasher
  (JNIEnv *env, jobject obj)
{
  blake3_hasher *hasher = new blake3_hasher;

  return (jlong)hasher;
}


extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_delete_1hasher
  (JNIEnv *env, jobject obj, jlong self)
{
  blake3_hasher *hasher = hasher_ptr(self);

  if (hasher != NULL) {
    delete hasher;
  }

  return;
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1init
  (JNIEnv *env, jobject obj, jlong self)
{
  return blake3_hasher_init(hasher_ptr(self));
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1init_1keyed
  (JNIEnv *env, jobject obj, jlong self, jbyteArray key)
{
  jbyte *key_addr = get_byte_array(env, key);

  blake3_hasher_init_keyed(hasher_ptr(self), (uint8_t*)key_addr);

  return release_byte_array(env, key, key_addr);
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1init_1derive_1key
  (JNIEnv *env, jobject obj, jlong self, jstring context)
{
  const char *ctx = env->GetStringUTFChars(context, nullptr);

  blake3_hasher_init_derive_key(hasher_ptr(self), ctx);

  return env->ReleaseStringUTFChars(context, ctx);
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1update
  (JNIEnv *env, jobject obj, jlong self, jbyteArray input, jint input_len)
{
  jbyte *input_addr = get_byte_array(env, input);

  blake3_hasher_update(hasher_ptr(self), input_addr, input_len);

  return release_byte_array(env, input, input_addr);
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1finalize
  (JNIEnv *env, jobject obj, jlong self, jbyteArray out, jint out_len)
{
  jbyte *out_addr = get_byte_array(env, out);

  blake3_hasher_finalize(hasher_ptr(self), (uint8_t*)out_addr, out_len);

  return release_byte_array(env, out, out_addr);
}

extern "C" JNIEXPORT void JNICALL Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1finalize_1seek
  (JNIEnv *env, jobject obj, jlong self, jlong seek, jbyteArray out, jint out_len)
{
  jbyte *out_addr = get_byte_array(env, out);

  blake3_hasher_finalize_seek(hasher_ptr(self), seek, (uint8_t*)out_addr, out_len);

  return release_byte_array(env, out, out_addr);
}

}  // namespace blaze_jni
