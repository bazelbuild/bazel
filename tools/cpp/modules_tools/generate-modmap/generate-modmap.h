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

#ifndef BAZEL_TOOLS_CPP_MODULE_TOOLS_GENERATE_MODMAP_GENERATE_MODMAP_H_
#define BAZEL_TOOLS_CPP_MODULE_TOOLS_GENERATE_MODMAP_GENERATE_MODMAP_H_

#include <iostream>
#include <unordered_set>

#include "common/common.h"

struct ModmapItem {
  std::string name;
  std::string path;
  bool operator==(const ModmapItem &other) const {
    return name == other.name && path == other.path;
  }
  friend std::ostream &operator<<(std::ostream &os, const ModmapItem &item) {
    os << "ModmapItem{name: " << item.name << ", path: " << item.path << "}";
    return os;
  }
};
// Define the hash function for the ModmapItem
namespace std {
template <>
struct hash<ModmapItem> {
  size_t operator()(const ModmapItem &item) const {
    return hash<string>()(item.name) ^ (hash<string>()(item.path) << 1);
  }
};
}  // namespace std
std::unordered_set<ModmapItem> process(const ModuleDep &dep,
                                       const Cpp20ModulesInfo &info);
void write_modmap(std::ostream &modmap_file_stream,
                  std::ostream &modmap_file_dot_input_stream,
                  const std::unordered_set<ModmapItem> &modmap,
                  const std::string &compiler,
                  const std::optional<ModmapItem> &generated);

#endif  // BAZEL_TOOLS_CPP_MODULE_TOOLS_GENERATE_MODMAP_GENERATE_MODMAP_H_
