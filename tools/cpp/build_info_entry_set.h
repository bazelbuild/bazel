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

#ifndef BAZEL_TOOLS_CPP_BUILD_INFO_ENTRY_SET_H_
#define BAZEL_TOOLS_CPP_BUILD_INFO_ENTRY_SET_H_

#include <string>
#include <unordered_map>

namespace bazel {
namespace tools {
namespace cpp {

class BuildInfoEntrySet {
 public:
  BuildInfoEntrySet(
      std::unordered_map<std::string, std::string>& info_file_map,
      std::unordered_map<std::string, std::string>& version_file_map)
      : info_file_map_(info_file_map), version_file_map_(version_file_map) {}
  enum KeyType {
    STRING = 0,
    INTEGER = 1,
  };
  struct KeyDescription {
    KeyDescription(KeyType key_type, const std::string& default_value,
                   const std::string& redacted_value)
        : key_type(key_type),
          default_value(default_value),
          redacted_value(redacted_value) {}
    const KeyType key_type;
    const std::string default_value;
    const std::string redacted_value;
    bool operator==(const KeyDescription& kd) const {
      return (key_type == kd.key_type && default_value == kd.default_value &&
              redacted_value == kd.redacted_value);
    }
  };

  virtual std::unordered_map<std::string, std::string>
  GetVolatileFileEntries() = 0;
  virtual std::unordered_map<std::string, std::string>
  GetNonVolatileFileEntries() = 0;
  virtual std::unordered_map<std::string, std::string>
  GetRedactedFileEntries() = 0;
  virtual ~BuildInfoEntrySet() = 0;

 private:
  std::unordered_map<std::string, std::string> info_file_map_;
  std::unordered_map<std::string, std::string> version_file_map_;
};

}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_CPP_BUILD_INFO_ENTRY_SET_H_
