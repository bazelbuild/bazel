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
#ifndef BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_
#define BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_

#include <string>
#include <unordered_map>

namespace bazel {
namespace tools {
namespace cpp {

class BuildInfoTranslationHelper {
 public:
  BuildInfoTranslationHelper(std::string info_file_path,
                             std::string version_file_path)
      : info_file_path_(info_file_path),
        version_file_path_(version_file_path) {}

  std::unordered_map<std::string, std::string> ParseInfoFile();
  std::unordered_map<std::string, std::string> ParseVersionFile();

  enum KeyType {
    STRING = 0,
    INTEGER = 1,
  };

  virtual std::unordered_map<
      std::string, std::pair<KeyType, std::pair<std::string, std::string> > >
  getStableKeys() = 0;
  virtual std::unordered_map<
      std::string, std::pair<KeyType, std::pair<std::string, std::string> > >
  getVolatileKeys() = 0;

  virtual ~BuildInfoTranslationHelper() = 0;

 private:
  std::string info_file_path_;
  std::string version_file_path_;
  std::unordered_map<std::string, std::string> ParseFile(std::string file_path);
};

}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_CPP_BUILD_INFO_TRANSLATION_HELPER_H_
