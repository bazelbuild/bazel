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
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

namespace blaze_util {

using std::string;
using std::vector;

static bool Symlink(const string& old_path, const string& new_path) {
  return symlink(old_path.c_str(), new_path.c_str()) == 0;
}

static bool CreateEmptyFile(const string& path) {
  // From the man page of open (man 2 open):
  // int open(const char *pathname, int flags, mode_t mode);
  //
  // mode specifies the permissions to use in case a new file is created.
  // This argument must be supplied when O_CREAT is specified in flags;
  // if O_CREAT is not specified, then mode is ignored.
  int fd = open(path.c_str(), O_CREAT | O_WRONLY, 0700);
  if (fd == -1) {
    return false;
  }
  return close(fd) == 0;
}

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

TEST(FileTest, MakeDirectories) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  ASSERT_STRNE(tmp_dir, NULL);
  const char* test_src_dir = getenv("TEST_SRCDIR");
  ASSERT_STRNE(NULL, test_src_dir);

  string dir = JoinPath(tmp_dir, "x/y/z");
  bool ok = MakeDirectories(dir, 0755);
  ASSERT_TRUE(ok);

  // Changing permissions on an existing dir should work.
  ok = MakeDirectories(dir, 0750);
  ASSERT_TRUE(ok);
  struct stat filestat = {};
  ASSERT_EQ(0, stat(dir.c_str(), &filestat));
  ASSERT_EQ(0750, filestat.st_mode & 0777);

  // srcdir shouldn't be writable.
  // TODO(ulfjack): Fix this!
  //  string srcdir = JoinPath(test_src_dir, "x/y/z");
  //  ok = MakeDirectories(srcdir, 0755);
  //  ASSERT_FALSE(ok);
  //  ASSERT_EQ(EACCES, errno);

  // Can't make a dir out of a file.
  string non_dir = JoinPath(dir, "w");
  ASSERT_TRUE(CreateEmptyFile(non_dir));
  ok = MakeDirectories(non_dir, 0755);
  ASSERT_FALSE(ok);
  ASSERT_EQ(ENOTDIR, errno);

  // Valid symlink should work.
  string symlink = JoinPath(tmp_dir, "z");
  ASSERT_TRUE(Symlink(dir, symlink));
  ok = MakeDirectories(symlink, 0755);
  ASSERT_TRUE(ok);

  // Error: Symlink to a file.
  symlink = JoinPath(tmp_dir, "w");
  ASSERT_TRUE(Symlink(non_dir, symlink));
  ok = MakeDirectories(symlink, 0755);
  ASSERT_FALSE(ok);
  ASSERT_EQ(ENOTDIR, errno);

  // Error: Symlink to a dir with wrong perms.
  symlink = JoinPath(tmp_dir, "s");
  ASSERT_TRUE(Symlink("/", symlink));

  // These perms will force a chmod()
  // TODO(ulfjack): Fix this!
  //  ok = MakeDirectories(symlink, 0000);
  //  ASSERTFALSE(ok);
  //  ASSERT_EQ(EPERM, errno);

  // Edge cases.
  ASSERT_FALSE(MakeDirectories("", 0755));
  ASSERT_EQ(EACCES, errno);
  ASSERT_FALSE(MakeDirectories("/", 0755));
  ASSERT_EQ(EACCES, errno);
}

TEST(FileTest, HammerMakeDirectories) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  ASSERT_STRNE(tmp_dir, NULL);

  string path = JoinPath(tmp_dir, "x/y/z");
  // TODO(ulfjack): Fix this!
  //  ASSERT_LE(0, fork());
  //  ASSERT_TRUE(MakeDirectories(path, 0755));
}

}  // namespace blaze_util
