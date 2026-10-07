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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace one_version {

DuplicateClassCollector::DuplicateClassCollector(
    size_t file_count_to_reserve_in_maps) {
  classes_.reserve(file_count_to_reserve_in_maps);
}

uint32_t DuplicateClassCollector::InternLabel(const Label& label) {
  if (labels_.empty() || labels_.back().name() != label.name() ||
      labels_.back().jar() != label.jar() ||
      labels_.back().allowlisted() != label.allowlisted()) {
    labels_.push_back(label);
  }
  return static_cast<uint32_t>(labels_.size() - 1);
}

absl::string_view DuplicateClassCollector::CopyToArena(absl::string_view s) {
  // 1 MiB is not finely tuned. Class names average ~50 bytes, so a large
  // classpath (~1M classes) needs only ~50 blocks, making the per-block
  // allocation cost negligible, while a small classpath wastes at most part of
  // one block (and untouched pages are typically never made resident).
  constexpr size_t kBlockSize = 1 << 20;
  if (s.empty()) {
    // Avoid handing a null arena_next_ to std::copy before the first block is
    // allocated (an entry named exactly ".class" yields an empty class name).
    return absl::string_view();
  }
  if (s.size() > arena_remaining_) {
    // Names longer than a block get a dedicated block; the unused tail of the
    // current block is abandoned.
    size_t block_size = std::max(kBlockSize, s.size());
    arena_blocks_.push_back(std::unique_ptr<char[]>(new char[block_size]));
    arena_next_ = arena_blocks_.back().get();
    arena_remaining_ = block_size;
  }
  char* copy = arena_next_;
  std::copy(s.begin(), s.end(), copy);
  arena_next_ += s.size();
  arena_remaining_ -= s.size();
  return absl::string_view(copy, s.size());
}

void DuplicateClassCollector::Add(absl::string_view class_name, uint32_t crc32,
                                  const Label& label) {
  uint32_t label_index = InternLabel(label);
  // lazy_emplace only copies the name into the arena when the class is new,
  // and needs a single hash lookup for both the find and the insert.
  auto it = classes_.lazy_emplace(class_name, [&](const auto& ctor) {
    ctor(CopyToArena(class_name), Entries());
  });
  Entries& entries = it->second;
  // Record the class the first time it is seen with a second crc32. If all
  // existing entries share the front entry's crc32, this is the first
  // conflict; otherwise the class is already in conflicts_. The all_of scan
  // only runs when the crc32 differs, so the common path stays O(1).
  if (!entries.empty() && crc32 != entries.front().crc32 &&
      std::all_of(entries.begin(), entries.end(), [&](const Entry& e) {
        return e.crc32 == entries.front().crc32;
      })) {
    conflicts_.push_back(it->first);
  }
  entries.push_back({crc32, label_index});
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
  violations.reserve(conflicts_.size());
  for (absl::string_view class_name : conflicts_) {
    const Entries& entries = classes_.at(class_name);
    // Replay the entries in insertion order through Violation::Add, so the
    // grouping (and therefore the sorted output) matches what storing a
    // Violation per class produced.
    Violation violation{std::string(class_name), std::vector<Version>()};
    for (const Entry& e : entries) {
      violation.Add(e.crc32, labels_[e.label_index]);
    }
    violation.Sort();
    violations.push_back(std::move(violation));
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
