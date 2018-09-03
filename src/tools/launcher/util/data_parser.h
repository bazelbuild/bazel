// Copyright 2017 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_LAUNCHER_UTIL_DATA_PARSER_H_
#define BAZEL_SRC_TOOLS_LAUNCHER_UTIL_DATA_PARSER_H_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

namespace bazel {
namespace launcher {

class LaunchDataParser {
 public:
  typedef std::unordered_map<std::string, std::wstring> LaunchInfo;
  LaunchDataParser() = delete;
  ~LaunchDataParser() = delete;
  static bool GetLaunchInfo(const std::wstring& binary_path,
                            LaunchInfo* launch_info);

 private:
  // Read the last 64 bit from the given binary to get the data size
  static int64_t ReadDataSize(std::ifstream* binary);

  // Read launch data at the end of the given binary into a buffer
  static void ReadLaunchData(std::ifstream* binary, char* launch_data,
                             int64_t data_size);

  // Parse the launch data into a map
  static bool ParseLaunchData(LaunchInfo* launch_info, const char* launch_data,
                              int64_t data_size);
};

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_UTIL_DATA_PARSER_H_
