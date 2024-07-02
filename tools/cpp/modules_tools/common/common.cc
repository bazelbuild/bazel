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

#include "common.h"
#include "json.hpp"

#include <iostream>

ModuleDep parse_ddi(std::istream &ddi_stream) {
  ModuleDep dep{};
  std::string ddi_string((std::istreambuf_iterator<char>(ddi_stream)),
                         std::istreambuf_iterator<char>());
  JsonValue data = parse_json(ddi_string);
  if (!data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }

  auto rules_data = data.as_object().at("rules");
  if (!rules_data.is_array()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  auto rules = rules_data.as_array();
  if (rules.size() > 1) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  if (rules.empty()) {
    return dep;
  }
  auto rule_data = rules[0];
  if (!rule_data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  auto rule = rule_data.as_object();
  auto provides_data = rule["provides"];
  if (!provides_data.is_array()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  auto provides = provides_data.as_array();
  if (provides.size() > 1) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  if (provides.size() == 1) {
    auto provide_data = provides[0];
    if (!provide_data.is_object()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    auto name_data = provide_data.as_object().at("logical-name");
    if (!name_data.is_string()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    dep.gen_bmi = true;
    dep.name = name_data.as_string();
  }
  auto requires_data = rule["requires"];
  if (!requires_data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  for (const auto &item_data : requires_data.as_array()) {

    if (!item_data.is_object()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    auto name_data = item_data.as_object().at("logical-name");
    if (!name_data.is_string()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    dep.require_list.push_back(name_data.as_string());
  }
  return dep;
}
Cpp20ModulesInfo parse_info(std::istream &info_stream) {
  std::string info_string((std::istreambuf_iterator<char>(info_stream)),
                          std::istreambuf_iterator<char>());
  JsonValue data = parse_json(info_string);
  if (!data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  auto modules_data = data.as_object().at("modules");
  if (!modules_data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  auto usages_data = data.as_object().at("usages");
  if (!usages_data.is_object()) {
    std::cerr << "bad ddi" << std::endl;
    std::exit(1);
  }
  Cpp20ModulesInfo info;
  for (const auto &item_data : modules_data.as_object()) {
    auto name = item_data.first;
    auto bmi_data = item_data.second;
    if (!bmi_data.is_string()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    info.modules[name] = bmi_data.as_string();
  }
  for (const auto &item_data : usages_data.as_object()) {
    auto name = item_data.first;
    auto require_list_data = item_data.second;
    if (!require_list_data.is_array()) {
      std::cerr << "bad ddi" << std::endl;
      std::exit(1);
    }
    std::vector<std::string> require_list;
    for (const auto &require_item_data : require_list_data.as_array()) {
      if (!require_item_data.is_string()) {
        std::cerr << "bad ddi" << std::endl;
        std::exit(1);
      }
      require_list.push_back(require_item_data.as_string());
    }
    info.usages[name] = require_list;
  }
  return info;
}
