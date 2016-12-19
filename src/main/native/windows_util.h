// Copyright 2016 The Bazel Authors. All rights reserved.
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
//
// INTERNAL header file for use by C++ code in this package.

#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__

#include <jni.h>

#include <memory>
#include <string>

namespace windows_util {

using std::string;
using std::unique_ptr;

string GetLastErrorString(const string& cause);

}  // namespace windows_util

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_UTIL_H__
