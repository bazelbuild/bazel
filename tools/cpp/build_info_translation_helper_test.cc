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

#include <fstream>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace bazel {
namespace tools {
namespace cpp {
namespace {
using ::testing::UnorderedElementsAre;

static const char kTestFilePrefix[] = "";

class BuildInfoTranslationHelperTest : public testing::Test {};

TEST_F(BuildInfoTranslationHelperTest, CorrectFileFormat) {
  BuildInfoTranslationHelper helper(
      absl::StrCat(kTestFilePrefix,
                   "bazel/tools/cpp/test_data/correct_file_format.txt"),
      "");
  std::unordered_map<std::string, std::string> expected_info_file_map(
      {{"key1", "value1"},
       {"key2", "value2"},
       {"key3", "value3 and some spaces"}});
  absl::Status expected_info_status = absl::OkStatus();

  std::unordered_map<std::string, std::string> actual_info_file_map;
  absl::Status actual_info_status = helper.ParseInfoFile(actual_info_file_map);

  EXPECT_EQ(actual_info_status, expected_info_status);
  EXPECT_EQ(actual_info_file_map, expected_info_file_map);
}

TEST_F(BuildInfoTranslationHelperTest, NonExistingFile) {
  BuildInfoTranslationHelper helper(
      "", "bazel/tools/cpp/test_data/this_file_does_not_exist.txt");
  absl::Status expected_version_status =
      absl::Status(absl::StatusCode::kNotFound,
                   "Could not open file: "
                   "bazel/tools/cpp/test_data/this_file_does_not_exist.txt");

  std::unordered_map<std::string, std::string> actual_version_file_map;
  absl::Status actual_version_status =
      helper.ParseVersionFile(actual_version_file_map);

  ASSERT_EQ(actual_version_status, expected_version_status);
}

TEST_F(BuildInfoTranslationHelperTest, DuplicatedKey) {
  BuildInfoTranslationHelper helper(
      absl::StrCat(kTestFilePrefix,
                   "bazel/tools/cpp/test_data/duplicated_key.txt"),
      "");
  absl::Status expected_info_status = absl::Status(
      absl::StatusCode::kFailedPrecondition, "key2 is duplicated in the file.");

  std::unordered_map<std::string, std::string> actual_info_file_map =
      std::unordered_map<std::string, std::string>();
  absl::Status actual_info_status = helper.ParseInfoFile(actual_info_file_map);

  ASSERT_EQ(actual_info_status, expected_info_status);
}

TEST_F(BuildInfoTranslationHelperTest, MissingSeparator) {
  BuildInfoTranslationHelper helper(
      absl::StrCat(kTestFilePrefix,
                   "bazel/tools/cpp/test_data/no_separator.txt"),
      "");

  std::unordered_map<std::string, std::string> expected_info_file_map(
      {{"key1", "value1"}, {"key2", "value2"}, {"key2value3", ""}});
  absl::Status expected_info_status = absl::OkStatus();

  std::unordered_map<std::string, std::string> actual_info_file_map;
  absl::Status actual_info_status = helper.ParseInfoFile(actual_info_file_map);

  EXPECT_EQ(actual_info_status, expected_info_status);
  EXPECT_THAT(actual_info_file_map,
              UnorderedElementsAre(std::make_pair("key1", "value1"),
                                   std::make_pair("key2", "value2"),
                                   std::make_pair("key2value3", "")));
}

TEST_F(BuildInfoTranslationHelperTest, WriteFileWorksCorrectly) {
  std::vector<std::string> expected_entries({"aaa", "bbb", "ccc", "ddd"});
  WriteFile(expected_entries,
            absl::StrCat(FLAGS_test_tmpdir, "/", "write_file.txt"));
  std::ifstream file_reader(
      absl::StrCat(FLAGS_test_tmpdir, "/", "write_file.txt"));
  std::string line = "";
  std::vector<std::string> actual_entries;
  while (std::getline(file_reader, line)) {
    actual_entries.push_back(line);
  }

  EXPECT_EQ(actual_entries, expected_entries);
}

}  // namespace
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
