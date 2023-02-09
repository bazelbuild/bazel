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
#include <windows.h>

#include "gtest/gtest.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

class LauncherMakerTest : public ::testing::Test {
 protected:
  LauncherMakerTest() {}

  virtual ~LauncherMakerTest() {}

  static std::wstring GetExecutableFileName() {
    // https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
    constexpr std::wstring::size_type maximum_file_name_length = 0x8000;
    std::wstring buffer(maximum_file_name_length, L'\0');
    DWORD length = GetModuleFileNameW(nullptr, &buffer.front(), buffer.size());
    if (length == 0 || length >= buffer.size()) {
      die(L"Failed to obtain executable filename");
    }
    return buffer.substr(0, length);
  }

  void SetUp() override {
    const std::wstring executable_file = GetExecutableFileName();
    if (!LaunchDataParser::GetLaunchInfo(executable_file,
                                         &parsed_launch_info)) {
      die(L"Failed to parse launch info.");
    }
  }

  void TearDown() override {}

  LaunchDataParser::LaunchInfo parsed_launch_info;

  std::string GetLaunchInfo(const std::string& key) const {
    auto item = parsed_launch_info.find(key);
    if (item == parsed_launch_info.end()) {
      return "Cannot find key: " + key;
    }
    return blaze_util::WstringToCstring(item->second);
  }
};

TEST_F(LauncherMakerTest, GetSimpleKeyTest) {
  ASSERT_EQ(GetLaunchInfo("foo_key"), "bar");
}

TEST_F(LauncherMakerTest, GetJoinedKeyTest) {
  ASSERT_EQ(GetLaunchInfo("foo_list"), "1\t2\t3");
}

TEST_F(LauncherMakerTest, GetEmptyJoinedKeyTest) {
  ASSERT_EQ(GetLaunchInfo("empty_list"), "");
}

}  // namespace launcher
}  // namespace bazel
