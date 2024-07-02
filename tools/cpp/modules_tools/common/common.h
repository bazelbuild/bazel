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

#pragma once

#include "json.hpp"

#include <string>
#include <unordered_map>
#include <vector>

struct Cpp20ModulesInfo {
  std::unordered_map<std::string, std::string> modules;
  std::unordered_map<std::string, std::vector<std::string>> usages;

  void merge(const Cpp20ModulesInfo &info) {
    for (const auto &item : info.modules) {
      modules[item.first] = item.second;
    }
    for (const auto &item : info.usages) {
      usages[item.first] = item.second;
    }
  }
};
struct ModuleDep {
  bool gen_bmi;
  std::string name;
  std::vector<std::string> require_list;
};

ModuleDep parse_ddi(std::istream &ddi_stream);
Cpp20ModulesInfo parse_info(std::istream &info_stream);
