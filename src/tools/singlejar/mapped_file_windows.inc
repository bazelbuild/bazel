// Copyright 2018 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_ 1

#if !defined(COMPILER_MSVC) || !defined(_WIN64)
#error This code is for 64 bit Windows.
#endif

#include <windows.h>

#include <string>

typedef HANDLE FileHandleType;

inline MappedFile::MappedFile()
    : mapped_start_(nullptr), mapped_end_(nullptr), fd_(INVALID_HANDLE_VALUE) {}

inline bool MappedFile::Open(const std::string& path) {
  // TODO(laszlocsomor): implement as part of
  // https://github.com/bazelbuild/bazel/issues/2241
  return false;
}

inline void MappedFile::Close() {
  // TODO(laszlocsomor): implement as part of
  // https://github.com/bazelbuild/bazel/issues/2241
}

inline bool MappedFile::is_open() const {
  // TODO(laszlocsomor): implement as part of
  // https://github.com/bazelbuild/bazel/issues/2241
  return false;
}

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_WINDOWS_H_