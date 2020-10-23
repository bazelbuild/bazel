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

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::ifstream;
using std::ios;
using std::make_unique;
using std::string;
using std::unique_ptr;
using std::wstring;

int64_t LaunchDataParser::ReadDataSize(ifstream* binary) {
  int64_t data_size;
  binary->seekg(0 - sizeof(data_size), ios::end);
  binary->read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
  return data_size;
}

void LaunchDataParser::ReadLaunchData(ifstream* binary, char* launch_data,
                                      int64_t data_size) {
  binary->seekg(0 - data_size - sizeof(data_size), ios::end);
  binary->read(launch_data, data_size);
}

bool LaunchDataParser::ParseLaunchData(LaunchInfo* launch_info,
                                       const char* launch_data,
                                       int64_t data_size) {
  int64_t start, end, equal;
  start = 0;
  while (start < data_size) {
    // Move start to point to the next non-null character.
    while (launch_data[start] == '\0' && start < data_size) {
      start++;
    }
    // Move end to the next null character or end of the string,
    // also find the first equal symbol appears.
    end = start;
    equal = -1;
    while (launch_data[end] != '\0' && end < data_size) {
      if (equal == -1 && launch_data[end] == '=') {
        equal = end;
      }
      end++;
    }
    if (equal == -1) {
      PrintError(L"Cannot find equal symbol in line: %hs",
                 string(launch_data + start, end - start).c_str());
      return false;
    } else if (start == equal) {
      PrintError(L"Key is empty string in line: %hs",
                 string(launch_data + start, end - start).c_str());
      return false;
    } else {
      string key(launch_data + start, equal - start);
      string value(launch_data + equal + 1, end - equal - 1);
      if (launch_info->find(key) != launch_info->end()) {
        PrintError(L"Duplicated launch info key: %hs", key.c_str());
        return false;
      }
      launch_info->insert(make_pair(key, blaze_util::CstringToWstring(value)));
    }
    start = end + 1;
  }
  return true;
}

bool LaunchDataParser::GetLaunchInfo(const wstring& binary_path,
                                     LaunchInfo* launch_info) {
  unique_ptr<ifstream> binary =
      make_unique<ifstream>(AsAbsoluteWindowsPath(binary_path.c_str()).c_str(),
                            ios::binary | ios::in);
  if (!binary->good()) {
    PrintError(L"Cannot open the binary to read launch data");
    return false;
  }
  int64_t data_size = ReadDataSize(binary.get());
  if (data_size == 0) {
    PrintError(L"No data appended, cannot launch anything!");
    return false;
  }
  unique_ptr<char[]> launch_data(new char[data_size]);
  ReadLaunchData(binary.get(), launch_data.get(), data_size);
  if (!ParseLaunchData(launch_info, launch_data.get(), data_size)) {
    return false;
  }
  return true;
}

}  // namespace launcher
}  // namespace bazel
