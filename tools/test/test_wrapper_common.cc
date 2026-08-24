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

#include "tools/test/test_wrapper_common.h"

#include <cstring>

namespace bazel {
namespace tools {
namespace test_wrapper {

void ZipEntryPaths::Create(const std::string& root,
                           const std::vector<std::string>& relative_paths) {
  size_ = relative_paths.size();

  size_t total_size = 0;
  for (const std::string& relative_path : relative_paths) {
    total_size += root.size() + 1 + relative_path.size() + 1;
  }

  abs_paths_.reset(new char[total_size]);
  abs_path_ptrs_.reset(new char*[size_ + 1]);
  entry_path_ptrs_.reset(new char*[size_ + 1]);

  char* next = abs_paths_.get();
  for (size_t i = 0; i < size_; ++i) {
    abs_path_ptrs_[i] = next;
    std::memcpy(next, root.data(), root.size());
    next += root.size();
    *next++ = '/';

    entry_path_ptrs_[i] = next;
    std::memcpy(next, relative_paths[i].c_str(), relative_paths[i].size() + 1);
    next += relative_paths[i].size() + 1;
  }

  abs_path_ptrs_[size_] = nullptr;
  entry_path_ptrs_[size_] = nullptr;
}

std::string FormatUndeclaredOutputManifestEntry(
    const std::string& relative_path, uint64_t size,
    const std::string& mime_type) {
  return relative_path + '\t' + std::to_string(size) + '\t' + mime_type + '\n';
}

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
