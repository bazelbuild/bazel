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

#ifndef THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ALLOWLIST_H_
#define THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ALLOWLIST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/tools/one_version/duplicate_class_collector.h"

namespace one_version {

// A allowlist of one version violations to ignore.
class Allowlist {
 public:
  virtual ~Allowlist() {}
  // All allowlisted labels for the given class name.
  virtual absl::flat_hash_set<std::string> AllLabels(
      absl::string_view class_name) = 0;
  // Apply the allowlist to the given violations.
  std::vector<Violation> Apply(absl::Span<const Violation> violations);
};

// A allowlist backed by a map.
class MapAllowlist : public Allowlist {
 public:
  explicit MapAllowlist(
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>
          allowlist)
      : allowlist_(std::move(allowlist)) {}

  absl::flat_hash_set<std::string> AllLabels(
      absl::string_view package_name) override;

 private:
  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> allowlist_;
};

}  // namespace one_version

#endif  // THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_ALLOWLIST_H_
