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

#include "src/tools/one_version/duplicate_class_collector.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"

namespace one_version {

void DuplicateClassCollector::Add(const std::string& class_name, uint32_t crc32,
                                  const Label& label) {
  auto it = violations_.find(class_name);
  if (it != violations_.end()) {
    it->second.Add(crc32, label);
    return;
  }
  Violation v(class_name, std::vector<Version>());
  v.Add(crc32, label);
  violations_.emplace(class_name, std::move(v));
}

void Violation::Add(uint32_t crc32, const Label& label) {
  for (Version& version : versions_) {
    if (version.crc32() == crc32) {
      version.Add(label);
      return;
    }
  }
  versions_.push_back(Version(crc32, std::vector<Label>{label}));
}

void Version::Add(const Label& label) { labels_.push_back(label); }

void Violation::Sort() {
  std::sort(
      versions_.begin(), versions_.end(),
      [](const Version& a, const Version& b) { return a.crc32() < b.crc32(); });
  for (Version& version : versions_) {
    version.Sort();
  }
}

void Version::Sort() {
  std::sort(labels_.begin(), labels_.end(),
            [](const Label& a, const Label& b) { return a.name() < b.name(); });
}

std::vector<Violation> DuplicateClassCollector::Violations() {
  std::vector<Violation> violations;
  for (const auto& e : violations_) {
    Violation violation = e.second;
    if (violation.versions().size() <= 1) {
      // We only saw one crc32.
      continue;
    }
    violation.Sort();
    violations.push_back(violation);
  }
  std::sort(violations.begin(), violations.end(),
            [](const Violation& a, const Violation& b) {
              return a.class_name() < b.class_name();
            });
  return violations;
}

std::string DuplicateClassCollector::Report(
    const std::vector<Violation>& violations) {
  std::string report;
  for (const Violation& violation : violations) {
    absl::StrAppend(&report, "  ", violation.class_name(),
                    " has incompatible definitions in:\n");
    for (const Version& version : violation.versions()) {
      absl::StrAppend(&report, "    crc32=", version.crc32(), "\n");
      for (const Label& label : version.labels()) {
        absl::StrAppend(&report, "      ", label.name(), " ",
                        label.allowlisted() ? "[allowlisted]" : "[new]", "\n");
        if (!label.jar().empty()) {
          absl::StrAppend(&report, "      via ", label.jar(), "\n");
        }
      }
    }
  }
  return report;
}

}  // namespace one_version
