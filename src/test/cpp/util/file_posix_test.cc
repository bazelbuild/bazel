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
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>

#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

namespace blaze_util {

using std::pair;
using std::string;
using std::vector;

TEST(FilePosixTest, Which) {
  ASSERT_EQ("", Which(""));
  ASSERT_EQ("", Which("foo"));
  ASSERT_EQ("", Which("/"));

  // /usr/bin/yes exists on Linux, Darwin, and MSYS, but "which yes" does not
  // always return that (if $PATH is different).
  string actual = Which("yes");
  // Assert that it's an absolute path
  ASSERT_EQ(0, actual.find("/"));
  // Assert that it ends with /yes, we cannot assume more than that.
  ASSERT_EQ(actual.size() - string("/yes").size(), actual.rfind("/yes"));
}

TEST(FilePosixTest, PathExists) {
  ASSERT_FALSE(PathExists("/this/should/not/exist/mkay"));
  ASSERT_FALSE(PathExists("non.existent"));
  ASSERT_FALSE(PathExists(""));

  // /usr/bin/yes exists on Linux, Darwin, and MSYS
  ASSERT_TRUE(PathExists("/"));
  ASSERT_TRUE(PathExists("/usr"));
  ASSERT_TRUE(PathExists("/usr/"));
  ASSERT_TRUE(PathExists("/usr/bin/yes"));
}

TEST(FilePosixTest, CanAccess) {
  for (int i = 0; i < 8; ++i) {
    ASSERT_FALSE(CanAccess("/this/should/not/exist/mkay", i & 1, i & 2, i & 4));
    ASSERT_FALSE(CanAccess("non.existent", i & 1, i & 2, i & 4));
  }

  for (int i = 0; i < 4; ++i) {
    // /usr/bin/yes exists on Linux, Darwin, and MSYS
    ASSERT_TRUE(CanAccess("/", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr/", i & 1, false, i & 2));
    ASSERT_TRUE(CanAccess("/usr/bin/yes", i & 1, false, i & 2));
  }

  char* tmpdir_cstr = getenv("TEST_TMPDIR");
  ASSERT_FALSE(tmpdir_cstr == NULL);

  string tmpdir(tmpdir_cstr);
  ASSERT_NE("", tmpdir);

  string mock_file = tmpdir + (tmpdir.back() == '/' ? "" : "/") +
                     "FilePosixTest.CanAccess.mock_file";
  int fd = open(mock_file.c_str(), O_CREAT, 0500);
  ASSERT_GT(fd, 0);
  close(fd);

  // Sanity check: assert that we successfully created the file with the given
  // permissions.
  ASSERT_EQ(0, access(mock_file.c_str(), R_OK | X_OK));
  ASSERT_NE(0, access(mock_file.c_str(), R_OK | W_OK | X_OK));

  // Actual assertion
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(CanAccess(mock_file, i & 1, false, i & 2));
    ASSERT_FALSE(CanAccess(mock_file, i & 1, true, i & 2));
  }
}

TEST(FilePosixTest, GetCwd) {
  char cwdbuf[PATH_MAX];
  ASSERT_EQ(cwdbuf, getcwd(cwdbuf, PATH_MAX));

  // Assert that GetCwd() and getcwd() return the same value.
  string cwd(cwdbuf);
  ASSERT_EQ(cwd, blaze_util::GetCwd());

  // Change to a different directory.
  ASSERT_EQ(0, chdir("/usr"));

  // Assert that GetCwd() returns the new CWD.
  ASSERT_EQ(string("/usr"), blaze_util::GetCwd());

  ASSERT_EQ(0, chdir(cwd.c_str()));
  ASSERT_EQ(cwd, blaze_util::GetCwd());
}

TEST(FilePosixTest, ChangeDirectory) {
  // Retrieve the current working directory.
  char old_wd[PATH_MAX];
  ASSERT_EQ(old_wd, getcwd(old_wd, PATH_MAX));

  // Change to a different directory and assert it was successful.
  ASSERT_FALSE(blaze_util::ChangeDirectory("/non/existent/path"));
  ASSERT_TRUE(blaze_util::ChangeDirectory("/usr"));
  char new_wd[PATH_MAX];
  ASSERT_EQ(new_wd, getcwd(new_wd, PATH_MAX));
  ASSERT_EQ(string("/usr"), string(new_wd));

  // Change back to the original CWD.
  ASSERT_TRUE(blaze_util::ChangeDirectory(old_wd));
  ASSERT_EQ(new_wd, getcwd(new_wd, PATH_MAX));
  ASSERT_EQ(string(old_wd), string(new_wd));
}

class MockDirectoryEntryConsumer : public DirectoryEntryConsumer {
 public:
  void Consume(const std::string &name, bool is_directory) override {
    entries.push_back(pair<string, bool>(name, is_directory));
  }

  vector<pair<string, bool> > entries;
};

TEST(FilePosixTest, ForEachDirectoryEntry) {
  // Get the test's temp dir.
  char* tmpdir_cstr = getenv("TEST_TMPDIR");
  ASSERT_FALSE(tmpdir_cstr == NULL);
  string tempdir(tmpdir_cstr);
  ASSERT_FALSE(tempdir.empty());
  if (tempdir.back() == '/') {
    tempdir = tempdir.substr(0, tempdir.size() - 1);
  }

  // Create the root directory for the mock directory tree.
  string root = tempdir + "/FilePosixTest.ForEachDirectoryEntry.root";
  ASSERT_EQ(0, mkdir(root.c_str(), 0700));

  // Names of mock files and directories.
  string dir = root + "/dir";
  string file = root + "/file";
  string dir_sym = root + "/dir_sym";
  string file_sym = root + "/file_sym";
  string subfile = dir + "/subfile";
  string subfile_through_sym = dir_sym + "/subfile";

  // Create mock directory, file, and symlinks.
  int fd = open(file.c_str(), O_CREAT, 0700);
  ASSERT_GT(fd, 0);
  close(fd);
  ASSERT_EQ(0, mkdir(dir.c_str(), 0700));
  ASSERT_EQ(0, symlink("dir", dir_sym.c_str()));
  ASSERT_EQ(0, symlink("file", file_sym.c_str()));
  fd = open(subfile.c_str(), O_CREAT, 0700);
  ASSERT_GT(fd, 0);
  close(fd);

  // Assert that stat'ing the symlinks (with following them) point to the right
  // filesystem entry types.
  struct stat stat_buf;
  ASSERT_EQ(0, stat(dir_sym.c_str(), &stat_buf));
  ASSERT_TRUE(S_ISDIR(stat_buf.st_mode));
  ASSERT_EQ(0, stat(file_sym.c_str(), &stat_buf));
  ASSERT_FALSE(S_ISDIR(stat_buf.st_mode));

  // Actual test: list the directory.
  MockDirectoryEntryConsumer consumer;
  ForEachDirectoryEntry(root, &consumer);
  ASSERT_EQ(4, consumer.entries.size());

  // Sort the collected directory entries.
  struct {
    bool operator()(const pair<string, bool> &a, const pair<string, bool> &b) {
      return a.first < b.first;
    }
  } sort_pairs;

  std::sort(consumer.entries.begin(), consumer.entries.end(), sort_pairs);

  // Assert that the directory entries have the right name and type.
  pair<string, bool> expected;
  expected = pair<string, bool>(dir, true);
  ASSERT_EQ(expected, consumer.entries[0]);
  expected = pair<string, bool>(dir_sym, false);
  ASSERT_EQ(expected, consumer.entries[1]);
  expected = pair<string, bool>(file, false);
  ASSERT_EQ(expected, consumer.entries[2]);
  expected = pair<string, bool>(file_sym, false);
  ASSERT_EQ(expected, consumer.entries[3]);

  // Actual test: list a directory symlink.
  consumer.entries.clear();
  ForEachDirectoryEntry(dir_sym, &consumer);
  ASSERT_EQ(1, consumer.entries.size());
  expected = pair<string, bool>(subfile_through_sym, false);
  ASSERT_EQ(expected, consumer.entries[0]);

  // Actual test: list a path that's actually a file, not a directory.
  consumer.entries.clear();
  ForEachDirectoryEntry(file, &consumer);
  ASSERT_TRUE(consumer.entries.empty());

  // Cleanup: delete mock directory tree.
  rmdir(subfile.c_str());
  rmdir(dir.c_str());
  unlink(dir_sym.c_str());
  unlink(file.c_str());
  unlink(file_sym.c_str());
  rmdir(root.c_str());
}

}  // namespace blaze_util
