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

#ifndef BAZEL_TOOLS_TEST_TEST_WRAPPER_COMMON_H_
#define BAZEL_TOOLS_TEST_TEST_WRAPPER_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace tools {
namespace test_wrapper {

// Owns the absolute file paths and relative ZIP entry paths passed to
// devtools_ijar::ZipBuilder.
class ZipEntryPaths {
 public:
  // `root` is an absolute path using forward slashes. `files` contains relative
  // ZIP entry paths using forward slashes.
  void Create(const std::string& root, const std::vector<std::string>& files);

  size_t Size() const { return size_; }

  // Both arrays are null-terminated and remain valid until this object is
  // destroyed or Create is called again.
  char const* const* AbsPathPtrs() const { return abs_path_ptrs_.get(); }
  char const* const* EntryPathPtrs() const { return entry_path_ptrs_.get(); }

 private:
  size_t size_ = 0;
  std::unique_ptr<char[]> abs_paths_;
  std::unique_ptr<char*[]> abs_path_ptrs_;
  std::unique_ptr<char*[]> entry_path_ptrs_;
};

// Returns one line for TEST_UNDECLARED_OUTPUTS_MANIFEST.
std::string FormatUndeclaredOutputManifestEntry(
    const std::string& relative_path, uint64_t size,
    const std::string& mime_type);

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_TEST_TEST_WRAPPER_COMMON_H_
