// Copyright 2018 The Bazel Authors. All rights reserved.
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

namespace bazel {
namespace tools {
namespace cpp {

// TODO(b/203032819): Add implementation.
std::unordered_map<std::string, std::string>
BuildInfoTranslationHelper::ParseFile(std::string file_path) {
  return std::unordered_map<std::string, std::string>();
}

// TODO(b/203032819): Add implementation.
std::unordered_map<std::string, std::string>
BuildInfoTranslationHelper::ParseInfoFile() {
  return BuildInfoTranslationHelper::ParseFile(info_file_path_);
}

// TODO(b/203032819): Add implementation.
std::unordered_map<std::string, std::string>
BuildInfoTranslationHelper::ParseVersionFile() {
  return BuildInfoTranslationHelper::ParseFile(version_file_path_);
}

BuildInfoTranslationHelper::~BuildInfoTranslationHelper() {}

}  // namespace cpp
}  // namespace tools
}  // namespace bazel
