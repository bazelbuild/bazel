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

#include <cstdint>

#include "c/blake3.h"

// Exported C functions called from Java through the java.lang.foreign API by
// Blake3MessageDigest, in addition to the blake3_hasher_* functions. They are
// not JNI methods. JNIEXPORT only makes them visible (default visibility on
// ELF/Mach-O, dllexport on Windows).

extern "C" {

JNIEXPORT int32_t blaze_blake3_hasher_size() {
  return static_cast<int32_t>(sizeof(blake3_hasher));
}

JNIEXPORT void blaze_blake3_hash(const void *input, size_t input_len,
                                 void *out) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, input, input_len);
  blake3_hasher_finalize(&hasher, static_cast<uint8_t *>(out), BLAKE3_OUT_LEN);
}

}  // extern "C"
