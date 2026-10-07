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

#ifndef THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_DUPLICATE_CLASS_COLLECTOR_H_
#define THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_DUPLICATE_CLASS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"

namespace one_version {

class Label {
 public:
  Label(std::string name, std::string jar)
      : name_(std::move(name)), jar_(std::move(jar)), allowlisted_(false) {}

  Label(std::string name, std::string jar, bool allowlisted)
      : name_(std::move(name)),
        jar_(std::move(jar)),
        allowlisted_(allowlisted) {}

  const std::string &name() const { return name_; }
  const std::string &jar() const { return jar_; }
  bool allowlisted() const { return allowlisted_; }

 private:
  std::string name_;
  std::string jar_;
  bool allowlisted_;
};

class Version {
 public:
  Version(uint32_t crc32, std::vector<Label> labels)
      : crc32_(crc32), labels_(std::move(labels)) {}

  void Add(const Label &label);
  void Sort();

  const uint32_t crc32() const { return crc32_; }
  const std::vector<Label> &labels() const { return labels_; }

 private:
  uint32_t crc32_;
  std::vector<Label> labels_;
};

class Violation {
 public:
  Violation(std::string class_name, std::vector<Version> versions)
      : class_name_(std::move(class_name)), versions_(std::move(versions)) {}

  void Add(uint32_t crc32, const Label &label);
  void Sort();

  const std::string &class_name() const { return class_name_; }
  const std::vector<Version> &versions() const { return versions_; }

 private:
  std::string class_name_;
  std::vector<Version> versions_;
};

// A collector for one version violations.
class DuplicateClassCollector {
 public:
  explicit DuplicateClassCollector(size_t file_count_to_reserve_in_maps = 0);

  // Records the class name, crc, and label of a classpath entry.
  void Add(absl::string_view class_name, uint32_t crc32, const Label& label);

  // Returns the collection of one version violations.
  std::vector<Violation> Violations();

  // Returns a report of one version errors.
  static std::string Report(const std::vector<Violation> &violations);

 private:
  // A single classpath entry for a class: its crc32 and an index into labels_.
  // Labels are stored once in labels_ rather than per entry, because a label
  // (target name and jar path) is shared by every class in a jar and copying
  // it for each class dominated the runtime on large classpaths.
  struct Entry {
    uint32_t crc32;
    uint32_t label_index;
  };

  // Most classes appear in exactly one jar, so one inline Entry avoids a heap
  // allocation per class. Classes that appear in several jars spill to the
  // heap, which is rare.
  using Entries = absl::InlinedVector<Entry, 1>;

  // Returns the index of label in labels_, adding it if necessary. Callers
  // typically add all entries of a jar consecutively, so only the most
  // recently added label is checked. If a caller interleaves labels, a label
  // may be stored more than once, which is harmless.
  uint32_t InternLabel(const Label& label);

  // Returns a copy of s that lives as long as this collector. Class names are
  // stored in large blocks to avoid a heap allocation (and a deallocation at
  // exit) per class.
  absl::string_view CopyToArena(absl::string_view s);

  std::vector<Label> labels_;
  // Blocks are never moved or freed before the collector is destroyed, so
  // string_views into them stay valid (also when the collector is moved).
  std::vector<std::unique_ptr<char[]>> arena_blocks_;
  char* arena_next_ = nullptr;
  size_t arena_remaining_ = 0;
  // Keys point into arena_blocks_.
  absl::flat_hash_map<absl::string_view, Entries> classes_;
  // Classes that were seen with more than one crc32, in the order they were
  // first seen to conflict. Tracking these during Add means Violations() does
  // not have to scan every class.
  std::vector<absl::string_view> conflicts_;

  // Lets the unit test observe arena_blocks_ to verify block rollover.
  friend class DuplicateClassCollectorTest;
};

}  // namespace one_version

#endif  // THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_DUPLICATE_CLASS_COLLECTOR_H_
