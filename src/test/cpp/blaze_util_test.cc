// Copyright 2015 The Bazel Authors. All rights reserved.
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
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/util/file.h"
#include "gtest/gtest.h"

namespace blaze {

static bool Symlink(const string& old_path, const string& new_path) {
  return symlink(old_path.c_str(), new_path.c_str()) == 0;
}

static bool CreateEmptyFile(const string& path) {
  int fd = open(path.c_str(), O_CREAT | O_WRONLY);
  if (fd == -1) {
    return false;
  }
  return close(fd) == 0;
}

class BlazeUtilTest : public ::testing::Test {
 protected:
  BlazeUtilTest() {
  }

  virtual ~BlazeUtilTest() {
  }

  static void ForkAndWrite(int fds[], string input1, string input2) {
    int r = fork();
    if (r == 0) {
      close(fds[0]);
      write(fds[1], input1.c_str(), input1.size());
      usleep(500);  // sleep for 50ms
      write(fds[1], input2.c_str(), input2.size());
      close(fds[1]);
      exit(0);
    } else if (r < 0) {
      perror("fork()");
      FAIL();
    } else {
      close(fds[1]);
    }
  }

  static int WriteFileDescriptor2(string input1, string input2) {
    // create a fd for the input string
    int fds[2];
    if (pipe(fds) == -1) {
      return -1;
    }
    if (fcntl(fds[0], F_SETFL, O_NONBLOCK) == -1
        || fcntl(fds[1], F_SETFL, O_NONBLOCK) == -1) {
      return -1;
    }
    if (input2.size() > 0) {
      ForkAndWrite(fds, input1, input2);
    } else {
      write(fds[1], input1.c_str(), input1.size());
      close(fds[1]);
    }
    return fds[0];
  }

  static void AssertReadFileDescriptor2(string input1, string input2) {
    int fd = WriteFileDescriptor2(input1, input2);
    if (fd < 0) {
      FAIL() << "Unable to create a pipe!";
    } else {
      string result;
      if (!ReadFileDescriptor(fd, &result)) {
        perror("ReadFileDescriptor");
        FAIL() << "Unable to read file descriptor!";
      } else {
        ASSERT_EQ(input1 + input2, result);
      }
    }
  }

  static void AssertReadFileDescriptor(string input) {
    AssertReadFileDescriptor2(input, "");
  }

  static void AssertReadJvmVersion2(string expected,
                                    string input1, string input2) {
    int fd = WriteFileDescriptor2(input1, input2);
    if (fd < 0) {
      FAIL() << "Unable to create a pipe!";
    } else {
      ASSERT_EQ(expected, ReadJvmVersion(fd));
    }
  }

  static void AssertReadJvmVersion(string expected, string input) {
    AssertReadJvmVersion2(expected, input, "");
  }

  void ReadFileDescriptorTest() const {
    AssertReadFileDescriptor("DummyJDK Blabla\n"
                         "More DummyJDK Blabla\n");
    AssertReadFileDescriptor("dummyjdk version \"1.42.qual\"\n"
                         "DummyJDK Blabla\n"
                         "More DummyJDK Blabla\n");
    AssertReadFileDescriptor2("first_line\n",
                         "second line version \"1.4.2_0\"\n");
  }

  void ReadJvmVersionTest() const {
    AssertReadJvmVersion("1.42", "dummyjdk version \"1.42\"\n"
                         "DummyJDK Blabla\n"
                         "More DummyJDK Blabla\n");
    AssertReadJvmVersion("1.42.qual", "dummyjdk version \"1.42.qual\"\n"
                         "DummyJDK Blabla\n"
                         "More DummyJDK Blabla\n");
    AssertReadJvmVersion("1.42.qualifie", "dummyjdk version \"1.42.qualifie");
    AssertReadJvmVersion("", "dummyjdk version ");
    AssertReadJvmVersion2("1.4.2_0", "first_line\n",
                         "second line version \"1.4.2_0\"\n");
  }

  void CheckJavaVersionIsAtLeastTest() const {
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", ""));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "0"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.7"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.7.0"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.0"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.6"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.42", "1"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.42", "1.7"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.42", "1.11"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.42.42", "1.11"));
    ASSERT_TRUE(CheckJavaVersionIsAtLeast("1.42.42", "1.11.11"));

    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "42"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "2"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.8"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.7.1"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.7.0-ver-specifier-42", "1.42"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.42", "2"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.42", "1.69"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.42", "1.42.1"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.42.42", "1.42.43"));
    ASSERT_FALSE(CheckJavaVersionIsAtLeast("1.42.42.0", "1.42.42.1"));
  }
};

TEST_F(BlazeUtilTest, CheckJavaVersionIsAtLeast) {
  CheckJavaVersionIsAtLeastTest();
}

TEST_F(BlazeUtilTest, ReadFileDescriptor) {
  ReadFileDescriptorTest();
}

TEST_F(BlazeUtilTest, ReadJvmVersion) {
  ReadJvmVersionTest();
}

TEST_F(BlazeUtilTest, MakeDirectories) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  ASSERT_STRNE(tmp_dir, NULL);
  const char* test_src_dir = getenv("TEST_SRCDIR");
  ASSERT_STRNE(NULL, test_src_dir);

  string dir = blaze_util::JoinPath(tmp_dir, "x/y/z");
  int ok = MakeDirectories(dir, 0755);
  ASSERT_EQ(0, ok);

  // Changing permissions on an existing dir should work.
  ok = MakeDirectories(dir, 0750);
  ASSERT_EQ(0, ok);
  struct stat filestat = {};
  ASSERT_EQ(0, stat(dir.c_str(), &filestat));
  ASSERT_EQ(0750, filestat.st_mode & 0777);

  // srcdir shouldn't be writable.
  // TODO(ulfjack): Fix this!
//  string srcdir = blaze_util::JoinPath(test_src_dir, "x/y/z");
//  ok = MakeDirectories(srcdir, 0755);
//  ASSERT_EQ(-1, ok);
//  ASSERT_EQ(EACCES, errno);

  // Can't make a dir out of a file.
  string non_dir = blaze_util::JoinPath(dir, "w");
  ASSERT_TRUE(CreateEmptyFile(non_dir));
  ok = MakeDirectories(non_dir, 0755);
  ASSERT_EQ(-1, ok);
  ASSERT_EQ(ENOTDIR, errno);

  // Valid symlink should work.
  string symlink = blaze_util::JoinPath(tmp_dir, "z");
  ASSERT_TRUE(Symlink(dir, symlink));
  ok = MakeDirectories(symlink, 0755);
  ASSERT_EQ(0, ok);

  // Error: Symlink to a file.
  symlink = blaze_util::JoinPath(tmp_dir, "w");
  ASSERT_TRUE(Symlink(non_dir, symlink));
  ok = MakeDirectories(symlink, 0755);
  ASSERT_EQ(-1, ok);
  ASSERT_EQ(ENOTDIR, errno);

  // Error: Symlink to a dir with wrong perms.
  symlink = blaze_util::JoinPath(tmp_dir, "s");
  ASSERT_TRUE(Symlink("/", symlink));

  // These perms will force a chmod()
  // TODO(ulfjack): Fix this!
//  ok = MakeDirectories(symlink, 0000);
//  ASSERT_EQ(-1, ok);
//  ASSERT_EQ(EPERM, errno);

  // Edge cases.
  ASSERT_EQ(-1, MakeDirectories("", 0755));
  ASSERT_EQ(EACCES, errno);
  ASSERT_EQ(-1, MakeDirectories("/", 0755));
  ASSERT_EQ(EACCES, errno);
}

TEST_F(BlazeUtilTest, HammerMakeDirectories) {
  const char* tmp_dir = getenv("TEST_TMPDIR");
  ASSERT_STRNE(tmp_dir, NULL);

  string path = blaze_util::JoinPath(tmp_dir, "x/y/z");
  // TODO(ulfjack): Fix this!
//  ASSERT_LE(0, fork());
//  ASSERT_EQ(0, MakeDirectories(path, 0755));
}

}  // namespace blaze
