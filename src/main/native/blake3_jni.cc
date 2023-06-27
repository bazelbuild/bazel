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

#include "external/blake3/c/blake3.h"

namespace blaze_jni {

blake3_hasher *hasher_ptr(jlong self) { return (blake3_hasher *)self; }

jbyte *get_byte_array(JNIEnv *env, jbyteArray java_array) {
  return env->GetByteArrayElements(java_array, nullptr);
}

void release_byte_array(JNIEnv *env, jbyteArray array, jbyte *addr) {
  return env->ReleaseByteArrayElements(array, addr, 0);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_vfs_Blake3JNI_allocate_1and_1initialize_1hasher(
    JNIEnv *env, jobject obj) {
  blake3_hasher *hasher = new blake3_hasher;
  blake3_hasher_init(hasher);
  return (jlong)hasher;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_hash_Blake3JNI_blake3_1hasher_1reset(
    JNIEnv *env, jobject obj, jlong self) {
  blake3_hasher *hasher = hasher_ptr(self);
  blake3_hasher_reset(hasher);
  return (jlong)hasher;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_Blake3JNI_blake3_1hasher_1update(
    JNIEnv *env, jobject obj, jlong self, jbyteArray input, jint input_len) {

  jbyte *input_addr = get_byte_array(env, input);
  blake3_hasher_update(hasher_ptr(self), input_addr, input_len);
  release_byte_array(env, input, input_addr);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_Blake3JNI_oneshot(
    JNIEnv *env, jobject obj, jbyteArray input, jint input_len, jbyteArray out,
    jint out_len) {
  blake3_hasher *hasher = new blake3_hasher;
  blake3_hasher_init(hasher);

  jbyte *input_addr = get_byte_array(env, input);
  blake3_hasher_update(hasher, input_addr, input_len);
  release_byte_array(env, input, input_addr);

  jbyte *out_addr = get_byte_array(env, out);
  blake3_hasher_finalize(hasher, (uint8_t *)out_addr, out_len);
  release_byte_array(env, out, out_addr);

  delete hasher;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_Blake3JNI_blake3_1hasher_1finalize_1and_1close(
    JNIEnv *env, jobject obj, jlong self, jbyteArray out, jint out_len) {
  blake3_hasher *hasher = hasher_ptr(self);

  jbyte *out_addr = get_byte_array(env, out);
  blake3_hasher_finalize(hasher, (uint8_t *)out_addr, out_len);
  release_byte_array(env, out, out_addr);

  delete hasher;
}

} // namespace blaze_jni
