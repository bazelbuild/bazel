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

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace bazel {
namespace tools {
namespace cpp {
inline void WriteFile(absl::Span<const std::string> entries,
                      const std::string& file_path) {
  std::ofstream file_writer(file_path);
  for (const std::string& i : entries) {
    file_writer << i << "\n";
    file_writer.flush();
  }
  file_writer.close();
}
class BuildInfoTranslationHelper {
 public:
  BuildInfoTranslationHelper(const std::string& info_file_path,
                             const std::string& version_file_path)
      : info_file_path_(info_file_path),
        version_file_path_(version_file_path) {}

  const absl::Status ParseInfoFile(
      std::unordered_map<std::string, std::string>& file_map);
  const absl::Status ParseVersionFile(
      std::unordered_map<std::string, std::string>& file_map);

 private:
  const std::string info_file_path_;
  const std::string version_file_path_;
  const absl::Status ParseFile(
      const std::string& file_path,
      std::unordered_map<std::string, std::string>& file_map);
};

}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_
