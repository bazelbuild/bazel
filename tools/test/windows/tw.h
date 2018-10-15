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

#ifndef BAZEL_TOOLS_TEST_WINDOWS_TW_H_
#define BAZEL_TOOLS_TEST_WINDOWS_TW_H_

#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace tools {
namespace test_wrapper {

struct FileInfo {
  std::wstring rel_path;
  // devtools_ijar::Stat::total_size is declared as `int`, which is what we
  // ultimately store the file size in, therefore this field is also `int`.
  int size;
};

namespace testing {

bool TestOnly_GetEnv(const wchar_t* name, std::wstring* result);

bool TestOnly_GetFileListRelativeTo(const std::wstring& abs_root,
                                    std::vector<FileInfo>* result);

}  // namespace testing
}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_TEST_WINDOWS_TW_H_

