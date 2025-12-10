// Copyright 2019 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_
#define THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_

#include <jni.h>

#include <string_view>

namespace blaze_jni {

// Returns a new Java String for a null-terminated sequence of Latin1
// characters.
jstring NewStringLatin1(JNIEnv* env, const char* str);

// Provides access to a Java String as a sequence of Latin1 characters.
// Any non-Latin1 characters are replaced with '?'.
class JStringLatin1Holder {
 public:
  // Constructs a JStringLatin1Holder.
  // Callers must check env->ExceptionOccurred() before using this object.
  JStringLatin1Holder(JNIEnv* env, jstring string);

  ~JStringLatin1Holder();

  // Returns a pointer to the null-terminated sequence of Latin1 characters.
  operator const char*() const;

  // Returns a string view into the sequence of Latin1 characters.
  operator std::string_view() const;

 private:
  const char* const chars;
};

}  // namespace blaze_jni

#endif  // THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_LATIN1_JNI_PATH_H_
