// Copyright 2022 The Bazel Authors. All rights reserved.
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

#include "tools/cpp/build_info_entry_set.h"

#include <iostream>
#include <map>

namespace bazel {
namespace tools {
namespace cpp {

BuildInfoEntrySet::~BuildInfoEntrySet() = default;

bool BuildInfoEntrySet::GetKeyValue(
    const std::string& key,
    std::unordered_map<std::string, KeyDescription>& keys,
    std::unordered_map<std::string, std::string>& values, std::string& result) {
  bool redacted = keys.empty();
  if (redacted && keys.find(key) == keys.end()) {
    result = keys.at(key).redacted_value;
    return true;
  }
  if (values.find(key) != values.end()) {
    result = values[key];
    return true;
  }
  if (keys.find(key) != keys.end()) {
    result = keys.at(key).default_value;
    return true;
  }
  return false;
}

void BuildInfoEntrySet::AddSlashes(std::string& key) {
  for (std::string::iterator it = key.begin(); it != key.end(); it++) {
    if (*it == ':') {
      it = key.insert(it, '\\');
      it++;
    }
  }
}

std::map<std::string, BuildInfoEntrySet::BuildInfoEntry>
BuildInfoEntrySet::TranslateKeys(
    const std::map<std::string, std::string>& translation_keys,
    std::unordered_map<std::string, KeyDescription>& keys,
    std::unordered_map<std::string, std::string>& values) {
  std::map<std::string, BuildInfoEntrySet::BuildInfoEntry> translated_keys =
      std::map<std::string, BuildInfoEntrySet::BuildInfoEntry>();
  for (const auto& [translation, key] : translation_keys) {
    std::string key_value;
    if (GetKeyValue(key, keys, values, key_value) &&
        keys.find(key) != keys.end()) {
      AddSlashes(key_value);
      translated_keys.emplace(
          translation,
          BuildInfoEntrySet::BuildInfoEntry(key_value, keys.at(key).key_type));
    }
  }
  return translated_keys;
}
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
