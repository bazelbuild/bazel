// Copyright 2016 The Bazel Authors. All rights reserved.
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

#include <dirent.h>
#include <errno.h>
#include <unistd.h>

#include <string>

#include "src/tools/singlejar/input_jar.h"

#include "googletest/include/gtest/gtest.h"

namespace {

static const char kJarsDirPath[] =
    "third_party/bazel/src/tools/singlejar/jars_to_test";

TEST(InputJarRandomJarsTest, ScanAllJars) {
  int processed_jars = 0;
  DIR *dirp = opendir(kJarsDirPath);
  ASSERT_NE(nullptr, dirp);

  struct dirent *dirent;
  InputJar input_jar;
  while ((dirent = readdir(dirp)) != nullptr) {
    if (dirent->d_type != DT_REG && dirent->d_type != DT_LNK) {
      continue;
    }
    std::string path = std::string(kJarsDirPath) + "/" + dirent->d_name;
    if (dirent->d_type == DT_LNK) {
      struct stat st;
      if (stat(path.c_str(), &st)) {
        perror(path.c_str());
        continue;
      } else if (!S_ISREG(st.st_mode)) {
        continue;
      }
    }
    EXPECT_TRUE(input_jar.Open(path));
    const LH *lh;
    const CDH *cdh;
    int file_count = 0;
    int entry_count = 0;
    for (; (cdh = input_jar.NextEntry(&lh)); ++entry_count) {
      ASSERT_TRUE(cdh->is());
      ASSERT_NE(nullptr, lh);
      ASSERT_TRUE(lh->is());
      EXPECT_EQ(lh->file_name_string(), cdh->file_name_string());
      if ('/' != lh->file_name()[lh->file_name_length() - 1]) {
        ++file_count;
      }
    }
    input_jar.Close();
    fprintf(stderr, "%s: %d files, %d entries\n", dirent->d_name, file_count,
            entry_count);
    ++processed_jars;
  }
  closedir(dirp);
  EXPECT_LT(0, processed_jars);
}

}  // namespace
