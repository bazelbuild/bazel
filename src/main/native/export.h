// Copyright 2026 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_EXPORT_H_
#define THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_EXPORT_H_

// Exports a function from the native library so that Java code can call it
// through the java.lang.foreign API.
#if defined(_WIN32)
#define BAZEL_EXPORT __declspec(dllexport)
#else
#define BAZEL_EXPORT __attribute__((visibility("default")))
#endif

#endif  // THIRD_PARTY_BAZEL_SRC_MAIN_NATIVE_EXPORT_H_
