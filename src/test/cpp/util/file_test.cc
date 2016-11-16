// Copyright 2014 The Bazel Authors. All rights reserved.
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
#include <algorithm>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

namespace blaze_util {

using std::string;
using std::vector;

TEST(FileTest, JoinPath) {
  std::string path = JoinPath("", "");
  ASSERT_EQ("", path);

  path = JoinPath("a", "b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a/", "b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a", "/b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("a/", "/b");
  ASSERT_EQ("a/b", path);

  path = JoinPath("/", "/");
  ASSERT_EQ("/", path);
}

void MockDirectoryListingFunction(const string &path,
                                  DirectoryEntryConsumer *consume) {
  if (path == "root") {
    consume->Consume("root/file1", false);
    consume->Consume("root/dir2", true);
    consume->Consume("root/dir1", true);
  } else if (path == "root/dir1") {
    consume->Consume("root/dir1/dir3", true);
    consume->Consume("root/dir1/file2", false);
  } else if (path == "root/dir2") {
    consume->Consume("root/dir2/file3", false);
  } else if (path == "root/dir1/dir3") {
    consume->Consume("root/dir1/dir3/file4", false);
    consume->Consume("root/dir1/dir3/file5", false);
  } else {
    // Unexpected path
    GTEST_FAIL();
  }
}

TEST(FileTest, GetAllFilesUnder) {
  vector<string> result;
  _GetAllFilesUnder("root", &result, &MockDirectoryListingFunction);
  std::sort(result.begin(), result.end());

  vector<string> expected({"root/dir1/dir3/file4",
                           "root/dir1/dir3/file5",
                           "root/dir1/file2",
                           "root/dir2/file3",
                           "root/file1"});
  ASSERT_EQ(expected, result);
}

}  // namespace blaze_util
