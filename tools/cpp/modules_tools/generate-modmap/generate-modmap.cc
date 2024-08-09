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

#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "generate-modmap.h"
// This function writes parameters about the required modules.
//
// Format of the modmap file
// Clang: -fmodule-file=<module-name>=<path/to/bmi>
// GCC:   <module-name> <path/to/bmi>
// MSVC:  /reference <module-name>=<path/to/bmi>
//
// NOTE: For the GCC compiler, additional `$root .` is added
//
//       Another special consideration for GCC is that it cannot specify
//       the output module name when compiling a Module Interface.
//       Therefore, the generated module name and BMI file path is added
//
// see also https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Module-Mapper.html
void write_modmap(std::ostream &modmap_file_stream,
                  std::ostream &modmap_file_dot_input_stream,
                  const std::unordered_set<ModmapItem> &modmap,
                  const std::string &compiler,
                  const std::optional<ModmapItem> &generated) {

  if (compiler == "gcc") {
    modmap_file_stream << "$root ."
                       << "\n";
    if (generated.has_value()) {
      modmap_file_stream << generated.value().name << " "
                         << generated.value().path << "\n";
    }
  }
  for (const auto &item : modmap) {
    if (compiler == "clang") {
      modmap_file_stream << "-fmodule-file=" << item.name << "=" << item.path
                         << "\n";
    } else if (compiler == "gcc") {
      modmap_file_stream << item.name << " " << item.path << "\n";
    } else if (compiler == "msvc-cl") {
      modmap_file_stream << "/reference " << item.name << "=" << item.path
                         << "\n";
    } else {
      std::cerr << "bad compiler: " << compiler << std::endl;
      std::exit(1);
    }
    modmap_file_dot_input_stream << item.path << "\n";
  }
}

std::unordered_set<ModmapItem> process(const ModuleDep &dep,
                                       const Cpp20ModulesInfo &info) {

  std::queue<std::string> q;
  for (const auto &item : dep.require_list) {
    q.push(item);
  }
  // Get all dependencies
  std::unordered_set<std::string> s;
  while (!q.empty()) {
    std::string name = q.front();
    q.pop();
    s.insert(name);
    auto it = info.usages.find(name);
    if (it == info.usages.end()) {
      continue;
    }
    auto deps = it->second;
    for (const auto &dep : deps) {
      if (s.count(dep)) {
        continue;
      }
      q.push(dep);
    }
  }

  // Construct modmap
  std::unordered_set<ModmapItem> modmap;
  for (const auto &name : s) {
    auto it = info.modules.find(name);
    if (it == info.modules.end()) {
      std::cerr << "ERROR: Module not found: " << name << std::endl;
      std::exit(1);
    }
    modmap.insert(ModmapItem{name, it->second});
  }
  return modmap;
}
