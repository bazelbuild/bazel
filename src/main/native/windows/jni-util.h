// Copyright 2017 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_JNI_UTIL_H_
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_JNI_UTIL_H_

#include <string>

#include "src/main/native/jni.h"

namespace bazel {
namespace windows {

std::wstring GetJavaWstring(JNIEnv* env, jstring str);

std::wstring GetJavaWpath(JNIEnv* env, jstring str);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_JNI_UTIL_H_
