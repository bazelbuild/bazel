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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

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
  // Records the class name, crc, and label of a classpath entry.
  void Add(const std::string &class_name, uint32_t crc32, const Label &label);

  // Returns the collection of one version violations.
  std::vector<Violation> Violations();

  // Returns a report of one version errors.
  static std::string Report(const std::vector<Violation> &violations);

 private:
  absl::flat_hash_map<std::string, Violation> violations_;
};

}  // namespace one_version

#endif  // THIRD_PARTY_BAZEL_SRC_TOOLS_ONE_VERSION_DUPLICATE_CLASS_COLLECTOR_H_
