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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/util/data_parser.h"
#include "gtest/gtest.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::getenv;
using std::ios;
using std::make_unique;
using std::ofstream;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

class LaunchDataParserTest : public ::testing::Test {
 protected:
  LaunchDataParserTest() {}

  virtual ~LaunchDataParserTest() {}

  void SetUp() override {
    char* tmpdir = getenv("TEST_TMPDIR");
    if (tmpdir != NULL) {
      test_tmpdir = string(tmpdir);
    } else {
      tmpdir = getenv("TEMP");
      ASSERT_FALSE(tmpdir == NULL);
      test_tmpdir = string(tmpdir);
    }
  }

  void TearDown() override {}

  static void WriteBinaryFileWithList(const string& binary_file,
                                      const vector<string>& launch_info) {
    ofstream binary_file_stream(binary_file, ios::out | ios::binary);

    int64_t data_size = 0;
    for (auto const& entry : launch_info) {
      binary_file_stream << entry;
      binary_file_stream << '\0';
      data_size += entry.length() + 1;
    }

    binary_file_stream.write(reinterpret_cast<char*>(&data_size),
                             sizeof(data_size));
  }

  static void WriteBinaryFileWithMap(
      const string& binary_file,
      const vector<pair<string, string>>& launch_info) {
    ofstream binary_file_stream(binary_file, ios::out | ios::binary);

    int64_t data_size = 0;
    for (auto const& entry : launch_info) {
      binary_file_stream << entry.first;
      binary_file_stream.put('=');
      binary_file_stream << entry.second;
      binary_file_stream.put('\0');
      data_size += entry.first.length() + entry.second.length() + 2;
    }

    binary_file_stream.write(reinterpret_cast<char*>(&data_size),
                             sizeof(data_size));
  }

  static bool ParseBinaryFile(
      const string& binary_file,
      LaunchDataParser::LaunchInfo* parsed_launch_info) {
    if (LaunchDataParser::GetLaunchInfo(
        blaze_util::CstringToWstring(binary_file), parsed_launch_info)) {
      return true;
    }
    exit(-1);
  }

  string GetLaunchInfo(const string& key) const {
    auto item = parsed_launch_info->find(key);
    if (item == parsed_launch_info->end()) {
      return "Cannot find key: " + key;
    }
    return blaze_util::WstringToCstring(item->second);
  }

  string test_tmpdir;
  unique_ptr<LaunchDataParser::LaunchInfo> parsed_launch_info;
};

TEST_F(LaunchDataParserTest, GetLaunchInfoTest) {
  vector<pair<string, string>> launch_info = {
      {"binary_type", "Bash"},
      {"workspace_name", "__main__"},
      {"bash_bin_path", "C:\\foo\\bar\\bash.exe"},
      {"bash_main_file", "./bazel-bin/foo/bar/bin.sh"},
      {"empty_value_key", ""},
  };

  string binary_file = test_tmpdir + "/binary_file";
  WriteBinaryFileWithMap(binary_file, launch_info);

  parsed_launch_info = make_unique<LaunchDataParser::LaunchInfo>();
  ASSERT_TRUE(ParseBinaryFile(binary_file, parsed_launch_info.get()));

  for (auto const& entry : launch_info) {
    ASSERT_EQ(entry.second, GetLaunchInfo(entry.first));
  }
  ASSERT_EQ(GetLaunchInfo("no_such_key"), "Cannot find key: no_such_key");
}

TEST_F(LaunchDataParserTest, EmptyLaunchInfoTest) {
  string binary_file = test_tmpdir + "/empty_binary_file";
  WriteBinaryFileWithMap(binary_file, {});

  parsed_launch_info = make_unique<LaunchDataParser::LaunchInfo>();
  // ASSERT_DEATH requires TEMP environment variable to be set.
  // Otherwise, it will try to write to C:/Windows, then fails.
  // A workaround in Bazel is to use --action_env to set TEMP.
  ASSERT_DEATH(ParseBinaryFile(binary_file, parsed_launch_info.get()),
               "LAUNCHER ERROR: No data appended, cannot launch anything!");
}

TEST_F(LaunchDataParserTest, DuplicatedLaunchInfoTest) {
  string binary_file = test_tmpdir + "/duplicated_binary_file";
  WriteBinaryFileWithMap(binary_file, {
                                          {"foo", "bar1"},
                                          {"foo", "bar2"},
                                      });

  parsed_launch_info = make_unique<LaunchDataParser::LaunchInfo>();
  // ASSERT_DEATH requires TEMP environment variable to be set.
  // Otherwise, it will try to write to C:/Windows, then fails.
  // A workaround in Bazel is to use --action_env to set TEMP.
  ASSERT_DEATH(ParseBinaryFile(binary_file, parsed_launch_info.get()),
               "LAUNCHER ERROR: Duplicated launch info key: foo");
}

TEST_F(LaunchDataParserTest, EmptyKeyLaunchInfoTest) {
  string binary_file = test_tmpdir + "/empty_key_binary_file";
  WriteBinaryFileWithMap(binary_file, {
                                          {"foo", "bar"},
                                          {"", "bar2"},
                                      });

  parsed_launch_info = make_unique<LaunchDataParser::LaunchInfo>();
  // ASSERT_DEATH requires TEMP environment variable to be set.
  // Otherwise, it will try to write to C:/Windows, then fails.
  // A workaround in Bazel is to use --action_env to set TEMP.
  ASSERT_DEATH(ParseBinaryFile(binary_file, parsed_launch_info.get()),
               "LAUNCHER ERROR: Key is empty string in line: =bar2");
}

TEST_F(LaunchDataParserTest, NoEqualSignLaunchInfoTest) {
  string binary_file = test_tmpdir + "/no_equal_binary_file";
  WriteBinaryFileWithList(binary_file, {
                                           "foo1=bar1",
                                           "foo2bar2",
                                       });

  parsed_launch_info = make_unique<LaunchDataParser::LaunchInfo>();
  // ASSERT_DEATH requires TEMP environment variable to be set.
  // Otherwise, it will try to write to C:/Windows, then fails.
  // A workaround in Bazel is to use --action_env to set TEMP.
  ASSERT_DEATH(ParseBinaryFile(binary_file, parsed_launch_info.get()),
               "LAUNCHER ERROR: Cannot find equal symbol in line: foo2bar2");
}

}  // namespace launcher
}  // namespace bazel
