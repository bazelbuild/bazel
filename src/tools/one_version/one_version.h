// Copyright 2024 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ONE_VERSION_H_
#define THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ONE_VERSION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "src/tools/one_version/allowlist.h"
#include "src/tools/one_version/duplicate_class_collector.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/zip_headers.h"

namespace one_version {

class OneVersion {
 public:
  explicit OneVersion(std::unique_ptr<one_version::Allowlist> whitelist)
      : whitelist_file_(std::move(whitelist)) {}

  // Record the jar entry (if it's a class file).
  void Add(absl::string_view file_name_of_entry, const CDH* jar_entry,
           const Label& label);
  std::vector<one_version::Violation> Report();

 private:
  std::unique_ptr<one_version::Allowlist> whitelist_file_;
  one_version::DuplicateClassCollector duplicate_class_collector_;
};

}  // namespace one_version

#endif  // THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ONE_VERSION_H_
