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
#include "tools/cpp/build_info_translation_helper.h"

#include <string>
#include <unordered_map>

#include "absl/strings/str_split.h"

namespace bazel {
namespace tools {
namespace cpp {
const absl::Status BuildInfoTranslationHelper::ParseFile(
    const std::string &file_path,
    std::unordered_map<std::string, std::string> &file_map) {
  std::ifstream file_reader(file_path);
  if (!file_reader.is_open()) {
    return absl::Status(absl::StatusCode::kNotFound,
                        absl::StrCat("Could not open file: ", file_path));
  }
  std::string line;
  // Split the line on the first separator, in case there is
  // no separator found return a non-zero exit code.
  constexpr static char kKeyValueSeparator[] = " ";
  while (std::getline(file_reader, line)) {
    if (absl::StrContains(line, kKeyValueSeparator)) {
      std::vector<std::string> key_and_value =
          absl::StrSplit(line, absl::MaxSplits(kKeyValueSeparator, 1));
      std::string key = key_and_value[0];
      std::string value = key_and_value[1];
      if (file_map.find(key) != file_map.end()) {
        return absl::Status(absl::StatusCode::kFailedPrecondition,
                            absl::StrCat(key, " is duplicated in the file."));
      }
      file_map.insert({key_and_value[0], key_and_value[1]});
    } else {
      file_map.insert({line, ""});
    }
  }

  return absl::Status(absl::StatusCode::kOk, "");
}

const absl::Status BuildInfoTranslationHelper::ParseInfoFile(
    std::unordered_map<std::string, std::string> &file_map) {
  return BuildInfoTranslationHelper::ParseFile(info_file_path_, file_map);
}

const absl::Status BuildInfoTranslationHelper::ParseVersionFile(
    std::unordered_map<std::string, std::string> &file_map) {
  return BuildInfoTranslationHelper::ParseFile(version_file_path_, file_map);
}

}  // namespace cpp
}  // namespace tools
}  // namespace bazel
