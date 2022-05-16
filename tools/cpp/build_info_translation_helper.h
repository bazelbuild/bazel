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
#ifndef BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_
#define BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_

#include <string>
#include <unordered_map>

#include "third_party/absl/status/status.h"

namespace bazel {
namespace tools {
namespace cpp {

class BuildInfoTranslationHelper {
 public:
  BuildInfoTranslationHelper(const std::string& info_file_path,
                             const std::string& version_file_path)
      : info_file_path_(info_file_path),
        version_file_path_(version_file_path) {}

  absl::Status ParseInfoFile(
      std::unordered_map<std::string, std::string>& file_map);
  absl::Status ParseVersionFile(
      std::unordered_map<std::string, std::string>& file_map);

 private:
  std::string info_file_path_;
  std::string version_file_path_;
  absl::Status ParseFile(
      const std::string& file_path,
      std::unordered_map<std::string, std::string>& file_map);

  const char kKeyValueSeparator = ' ';
};

}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_
