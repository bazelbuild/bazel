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

#include "c/blake3.h"

namespace blaze_jni {

jbyte *get_byte_array(JNIEnv *env, jbyteArray java_array) {
  return (jbyte *)env->GetPrimitiveArrayCritical(java_array, nullptr);
}

void release_byte_array(JNIEnv *env, jbyteArray array, jbyte *addr) {
  env->ReleasePrimitiveArrayCritical(array, addr, 0);
}

extern "C" JNIEXPORT int JNICALL
Java_com_google_devtools_build_lib_vfs_bazel_Blake3MessageDigest_hasher_1size(
    JNIEnv *env, jobject obj) {
  return (int)sizeof(blake3_hasher);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_bazel_Blake3MessageDigest_initialize_1hasher(
    JNIEnv *env, jobject obj, jbyteArray jhasher) {
  blake3_hasher *hasher = (blake3_hasher *)get_byte_array(env, jhasher);
  if (hasher) {
    blake3_hasher_init(hasher);
    release_byte_array(env, jhasher, (jbyte *)hasher);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_bazel_Blake3MessageDigest_blake3_1hasher_1update(
    JNIEnv *env, jobject obj, jbyteArray jhasher, jbyteArray input, jint offset,
    jint input_len) {
  if (input == nullptr) {
    return;
  }
  jsize input_size = env->GetArrayLength(input);
  if (offset < 0 || input_len < 0 || offset > input_size - input_len) {
    jclass ex_class = env->FindClass("java/lang/ArrayIndexOutOfBoundsException");
    if (ex_class != nullptr) {
      env->ThrowNew(ex_class, "Offset or length out of bounds");
    }
    return;
  }

  blake3_hasher *hasher = (blake3_hasher *)get_byte_array(env, jhasher);
  if (hasher) {
    jbyte *input_addr = get_byte_array(env, input);
    if (input_addr) {
      blake3_hasher_update(hasher, input_addr + offset, input_len);
      release_byte_array(env, input, input_addr);
    }
    release_byte_array(env, jhasher, (jbyte *)hasher);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_vfs_bazel_Blake3MessageDigest_blake3_1hasher_1finalize(
    JNIEnv *env, jobject obj, jbyteArray jhasher, jbyteArray out,
    jint out_len) {
  if (out == nullptr) {
    return;
  }
  jsize out_size = env->GetArrayLength(out);
  if (out_len < 0 || out_len > out_size) {
    jclass ex_class = env->FindClass("java/lang/ArrayIndexOutOfBoundsException");
    if (ex_class != nullptr) {
      env->ThrowNew(ex_class, "Output length out of bounds");
    }
    return;
  }

  blake3_hasher *hasher = (blake3_hasher *)get_byte_array(env, jhasher);
  if (hasher) {
    jbyte *out_addr = get_byte_array(env, out);
    if (out_addr) {
      blake3_hasher_finalize(hasher, (uint8_t *)out_addr, out_len);
      release_byte_array(env, out, out_addr);
    }
    release_byte_array(env, jhasher, (jbyte *)hasher);
  }
}

}  // namespace blaze_jni
