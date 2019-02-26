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
#ifndef BAZEL_SRC_TEST_CPP_UTIL_WINDOWS_TEST_UTIL_H_
#define BAZEL_SRC_TEST_CPP_UTIL_WINDOWS_TEST_UTIL_H_

#include <string>

namespace blaze_util {

// Returns $TEST_TMPDIR as a wstring.
// The result will have backslashes as directory separators, but no UNC prefix.
// The result will also not have a trailing backslash.
std::wstring GetTestTmpDirW();

// Deletes all files and directories under `path`.
// `path` must be a valid Windows path, but doesn't need to have a UNC prefix.
bool DeleteAllUnder(std::wstring path);

// Creates a dummy file under `path`.
// `path` must be a valid Windows path, and have a UNC prefix if necessary.
bool CreateDummyFile(const std::wstring& path,
                     const std::string& content = "hello");

// Creates a dummy file under `path`.
// `path` must be a valid Windows path, and have a UNC prefix if necessary.
bool CreateDummyFile(const std::wstring& path,
                     const void* content, const DWORD size);

}  // namespace blaze_util

#endif  // BAZEL_SRC_TEST_CPP_UTIL_WINDOWS_TEST_UTIL_H_
