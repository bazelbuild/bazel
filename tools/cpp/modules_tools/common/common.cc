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
void die(const std::string &msg) {
  std::cerr << msg << std::endl;
  std::exit(1);
}
ModuleDep parse_ddi(std::istream &ddi_stream) {
  ModuleDep dep{};
  std::string ddi_string((std::istreambuf_iterator<char>(ddi_stream)),
                         std::istreambuf_iterator<char>());
  JsonValue data = parse_json(ddi_string);
  if (!data.is_object()) {
    die("require ddi content is JSON object");
  }

  auto data_obj = data.as_object();
  if (data_obj.find("rules") == data_obj.end()) {
    die("require 'rules' in ddi content");
  }

  auto rules_data = data.as_object().at("rules");
  if (!rules_data.is_array()) {
    die("require ddi content 'rules' is JSON array");
  }
  auto rules = rules_data.as_array();
  // Only 1 rule in DDI file
  // DDI files can contain multiple rules (in general).
  // bazel does per-TU scanning rather than batch scanning.
  // Therefore, report error if multiple rules here
  if (rules.size() > 1) {
    die("require ddi content 'rules' has only 1 rule");
  }
  if (rules.empty()) {
    return dep;
  }
  auto rule_data = rules[0];
  if (!rule_data.is_object()) {
    die("require ddi content 'rules[0]' is JSON object");
  }
  auto rule = rule_data.as_object();
  auto provides_data = rule["provides"];
  if (!provides_data.is_array()) {
    die("require ddi content 'rules[0][\"provides\"]' is JSON array");
  }
  // Only 1 provide in rule
  // In C++20 Modules, one TU provide only one module.
  // Fortran can provide more than one module per TU.
  // This check is fine for C++20 Modules.
  auto provides = provides_data.as_array();
  if (provides.size() > 1) {
    die("require ddi content 'rules[0][\"provides\"]' has only 1 provide");
  }
  if (provides.size() == 1) {
    auto provide_data = provides[0];
    if (!provide_data.is_object()) {
      die("require ddi content 'rules[0][\"provides\"][0]' is JSON object");
    }
    auto provide_obj = provide_data.as_object();
    if (provide_obj.find("logical-name") == provide_obj.end()) {
      die("require 'logical-name' in 'rules[0][\"provides\"][0]'");
    }
    auto name_data = provide_obj.at("logical-name");
    if (!name_data.is_string()) {
      die("require ddi content 'rules[0][\"provides\"][0][\"logical-name\"]' "
          "is JSON string");
    }
    dep.gen_bmi = true;
    dep.name = name_data.as_string();
  }
  auto requires_data = rule["requires"];
  if (!requires_data.is_array()) {
    die("require ddi content 'rules[0][\"requires\"]' is JSON array");
  }
  for (const auto &item_data : requires_data.as_array()) {
    if (!item_data.is_object()) {
      die("require JSON object, but got " + item_data.dump());
    }
    auto item_obj = item_data.as_object();
    if (item_obj.find("logical-name") == item_obj.end()) {
      die("requrie 'logical-name' in 'rules[0][\"requires\"]' item");
    }
    auto name_data = item_obj.at("logical-name");
    if (!name_data.is_string()) {
      die("require JSON string, but got " + name_data.dump());
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
    die("require content is JSON object");
  }
  auto data_obj = data.as_object();
  if (data_obj.find("modules") == data_obj.end()) {
    die("require 'modules' in JSON object");
  }
  auto modules_data = data_obj.at("modules");
  if (!modules_data.is_object()) {
    die("require 'modules' is JSON object");
  }
  if (data_obj.find("usages") == data_obj.end()) {
    die("require 'usages' in JSON object");
  }
  auto usages_data = data_obj.at("usages");
  if (!usages_data.is_object()) {
    die("require 'usages' is JSON object");
  }
  Cpp20ModulesInfo info;
  for (const auto &item_data : modules_data.as_object()) {
    auto name = item_data.first;
    auto bmi_data = item_data.second;
    if (!bmi_data.is_string()) {
      die("require JSON string, but got " + bmi_data.dump());
    }
    info.modules[name] = bmi_data.as_string();
  }
  for (const auto &item_data : usages_data.as_object()) {
    auto name = item_data.first;
    auto require_list_data = item_data.second;
    if (!require_list_data.is_array()) {
      die("require JSON array");
    }
    std::vector<std::string> require_list;
    for (const auto &require_item_data : require_list_data.as_array()) {
      if (!require_item_data.is_string()) {
        die("require JSON string, but got " + require_item_data.dump());
      }
      require_list.push_back(require_item_data.as_string());
    }
    info.usages[name] = require_list;
  }
  return info;
}
