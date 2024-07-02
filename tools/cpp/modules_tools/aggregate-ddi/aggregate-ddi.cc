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

#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "aggregate-ddi.h"
void write_output(std::ostream &output, const Cpp20ModulesInfo &info) {
  JsonValue::ObjectType obj;
  JsonValue::ObjectType modules;
  JsonValue::ObjectType usages;
  for (const auto &item : info.modules) {
    modules[item.first] = JsonValue(item.second);
  }
  for (const auto &item : info.usages) {
    JsonValue::ArrayType list;
    for (const auto &require_item : item.second) {
      list.push_back(JsonValue(require_item));
    }
    usages[item.first] = list;
  }
  obj["modules"] = modules;
  obj["usages"] = usages;
  output << to_json(obj);
}
