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

#include "src/tools/one_version/allowlist.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/tools/one_version/duplicate_class_collector.h"

namespace one_version {

// For classes in the default package, this returns the class name as well
// (although this shouldn't matter in almost any real build)
absl::string_view package_name_from_class_name(absl::string_view class_name) {
  absl::string_view package_name(class_name);
  return package_name.substr(0, package_name.find_last_of('/'));
}

std::vector<Violation> Allowlist::Apply(
    absl::Span<const Violation> violations) {
  std::vector<Violation> result;
  for (const Violation& violation : violations) {
    absl::flat_hash_set<std::string> allowlisted_labels =
        AllLabels(package_name_from_class_name(violation.class_name()));
    std::vector<Version> versions;
    int new_versions = 0;
    for (const Version& version : violation.versions()) {
      std::vector<Label> labels;
      bool new_labels = false;
      for (const Label& label : version.labels()) {
        bool allowlisted =
            allowlisted_labels.find(label.name()) != allowlisted_labels.end();
        if (!allowlisted) {
          new_labels = true;
        }
        labels.emplace_back(label.name(), label.jar(), allowlisted);
      }
      if (new_labels) {
        ++new_versions;
      }
      versions.push_back(Version(version.crc32(), std::move(labels)));
    }
    if (new_versions > 1) {
      result.push_back(Violation(violation.class_name(), std::move(versions)));
    }
  }
  return result;
}

absl::flat_hash_set<std::string> MapAllowlist::AllLabels(
    absl::string_view package_name) {
  return allowlist_[package_name];
}

}  // namespace one_version
