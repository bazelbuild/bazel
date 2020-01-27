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
#include <limits.h>
#include <unistd.h>

#include <algorithm>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/test/cpp/util/test_util.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

using std::pair;
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

void MockDirectoryListingFunction(const string& path,
                                  DirectoryEntryConsumer* consume) {
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

TEST(FilePosixTest, GetAllFilesUnder) {
  vector<string> result;
  _GetAllFilesUnder("root", &result, &MockDirectoryListingFunction);
  std::sort(result.begin(), result.end());

  vector<string> expected({"root/dir1/dir3/file4", "root/dir1/dir3/file5",
                           "root/dir1/file2", "root/dir2/file3",
                           "root/file1"});
  ASSERT_EQ(expected, result);
}

TEST(FilePosixTest, MakeDirectories) {
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
  ASSERT_EQ(mode_t(0750), filestat.st_mode & 0777);

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

TEST(FilePosixTest, HammerMakeDirectories) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  ASSERT_STRNE(tmp_dir, NULL);

  string path = JoinPath(tmp_dir, "x/y/z");
  // TODO(ulfjack): Fix this!
  //  ASSERT_LE(0, fork());
  //  ASSERT_TRUE(MakeDirectories(path, 0755));
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
  ASSERT_FALSE(CanReadFile("/this/should/not/exist/mkay"));
  ASSERT_FALSE(CanExecuteFile("/this/should/not/exist/mkay"));
  ASSERT_FALSE(CanAccessDirectory("/this/should/not/exist/mkay"));

  ASSERT_FALSE(CanReadFile("non.existent"));
  ASSERT_FALSE(CanExecuteFile("non.existent"));
  ASSERT_FALSE(CanAccessDirectory("non.existent"));

  const char* tmpdir = getenv("TEST_TMPDIR");
  ASSERT_NE(nullptr, tmpdir);
  ASSERT_NE(0, *tmpdir);

  string dir(JoinPath(tmpdir, "canaccesstest"));
  ASSERT_EQ(0, mkdir(dir.c_str(), 0700));

  ASSERT_FALSE(CanReadFile(dir));
  ASSERT_FALSE(CanExecuteFile(dir));
  ASSERT_TRUE(CanAccessDirectory(dir));

  string file(JoinPath(dir, "foo.txt"));
  AutoFileStream fh(fopen(file.c_str(), "wt"));
  EXPECT_TRUE(fh.IsOpen());
  ASSERT_LT(0, fprintf(fh, "hello"));
  fh.Close();

  ASSERT_TRUE(CanReadFile(file));
  ASSERT_FALSE(CanExecuteFile(file));
  ASSERT_FALSE(CanAccessDirectory(file));

  ASSERT_EQ(0, chmod(file.c_str(), 0100));
  ASSERT_FALSE(CanReadFile(file));
  ASSERT_TRUE(CanExecuteFile(file));
  ASSERT_FALSE(CanAccessDirectory(file));

  ASSERT_EQ(0, chmod(dir.c_str(), 0500));
  ASSERT_FALSE(CanReadFile(dir));
  ASSERT_FALSE(CanExecuteFile(dir));
  ASSERT_FALSE(CanAccessDirectory(dir));
  ASSERT_EQ(0, chmod(dir.c_str(), 0700));

  ASSERT_EQ(0, unlink(file.c_str()));
  ASSERT_EQ(0, rmdir(dir.c_str()));
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
  void Consume(const std::string& name, bool is_directory) override {
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
  AutoFd fd(open(file.c_str(), O_CREAT, 0700));
  ASSERT_TRUE(fd.IsOpen());
  fd.Close();
  ASSERT_EQ(0, mkdir(dir.c_str(), 0700));
  ASSERT_EQ(0, symlink("dir", dir_sym.c_str()));
  ASSERT_EQ(0, symlink("file", file_sym.c_str()));
  fd = open(subfile.c_str(), O_CREAT, 0700);
  ASSERT_TRUE(fd.IsOpen());
  fd.Close();

  // Assert that stat'ing the symlinks (with following them) point to the
  // right filesystem entry types.
  struct stat stat_buf;
  ASSERT_EQ(0, stat(dir_sym.c_str(), &stat_buf));
  ASSERT_TRUE(S_ISDIR(stat_buf.st_mode));
  ASSERT_EQ(0, stat(file_sym.c_str(), &stat_buf));
  ASSERT_FALSE(S_ISDIR(stat_buf.st_mode));

  // Actual test: list the directory.
  MockDirectoryEntryConsumer consumer;
  ForEachDirectoryEntry(root, &consumer);
  ASSERT_EQ(size_t(4), consumer.entries.size());

  // Sort the collected directory entries.
  struct {
    bool operator()(const pair<string, bool>& a,
                    const pair<string, bool>& b) {
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
  ASSERT_EQ(size_t(1), consumer.entries.size());
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
