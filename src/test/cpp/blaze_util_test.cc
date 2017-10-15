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
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "gtest/gtest.h"

namespace blaze {

using std::string;

class BlazeUtilTest : public ::testing::Test {
 protected:
  BlazeUtilTest() {
  }

  virtual ~BlazeUtilTest() {
  }

  static void ForkAndWrite(int fds[], const string& input1,
                           const string& input2) {
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

  static int WriteFileDescriptor2(const string& input1, const string& input2) {
    // create a fd for the input string
    int fds[2];
    if (pipe(fds) == -1) {
      return -1;
    }
    if (fcntl(fds[0], F_SETFL, O_NONBLOCK) == -1
        || fcntl(fds[1], F_SETFL, O_NONBLOCK) == -1) {
      return -1;
    }
    if (!input2.empty()) {
      ForkAndWrite(fds, input1, input2);
    } else {
      write(fds[1], input1.c_str(), input1.size());
      close(fds[1]);
    }
    return fds[0];
  }

  static void AssertReadFrom2(const string& input1, const string& input2) {
    int fd = WriteFileDescriptor2(input1, input2);
    if (fd < 0) {
      FAIL() << "Unable to create a pipe!";
    } else {
      string result;
      bool success = blaze_util::ReadFrom(fd, &result);
      close(fd);
      if (!success) {
        perror("ReadFrom");
        FAIL() << "Unable to read file descriptor!";
      } else {
        ASSERT_EQ(input1 + input2, result);
      }
    }
  }

  static void AssertReadFrom(string input) { AssertReadFrom2(input, ""); }

  static void AssertReadJvmVersion(string expected, const string& input) {
    ASSERT_EQ(expected, ReadJvmVersion(input));
  }

  void ReadFromTest() const {
    AssertReadFrom(
        "DummyJDK Blabla\n"
        "More DummyJDK Blabla\n");
    AssertReadFrom(
        "dummyjdk version \"1.42.qual\"\n"
        "DummyJDK Blabla\n"
        "More DummyJDK Blabla\n");
    AssertReadFrom2("first_line\n", "second line version \"1.4.2_0\"\n");
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
    AssertReadJvmVersion("1.4.2_0",
                          "first_line\nsecond line version \"1.4.2_0\"\n");
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

TEST_F(BlazeUtilTest, ReadFrom) { ReadFromTest(); }

TEST_F(BlazeUtilTest, ReadJvmVersion) {
  ReadJvmVersionTest();
}

TEST_F(BlazeUtilTest, TestSearchNullaryEmptyCase) {
  ASSERT_FALSE(SearchNullaryOption({}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnaryEmptyCase) {
  ASSERT_STREQ(nullptr, SearchUnaryOption({}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryForEmpty) {
  ASSERT_FALSE(SearchNullaryOption({"bazel", "build", ":target"}, ""));
}

TEST_F(BlazeUtilTest, TestSearchNullaryForFlagNotPresent) {
  ASSERT_FALSE(SearchNullaryOption({"bazel", "build", ":target"},
                                   "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryStartupOption) {
  ASSERT_TRUE(SearchNullaryOption({"bazel", "--flag", "build", ":target"},
                                  "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryStartupOptionWithEquals) {
  ASSERT_DEATH(SearchNullaryOption(
      {"bazel", "--flag=value", "build", ":target"}, "--flag"),
              "In argument '--flag=value': option "
              "'--flag' does not take a value");
}

TEST_F(BlazeUtilTest, TestSearchNullaryCommandOption) {
  ASSERT_TRUE(SearchNullaryOption({"bazel", "build", ":target", "--flag"},
                                  "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullarySkipsAfterDashDash) {
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "build", ":target", "--", "--flag"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullarySucceedsWithEqualsAndDashDash) {
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "build", ":target", "--", "--flag=value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereEmptyCase) {
  ASSERT_FALSE(SearchNullaryOptionEverywhere({}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereForEmpty) {
  ASSERT_FALSE(SearchNullaryOptionEverywhere(
      {"bazel", "build", ":target", "--"}, ""));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereForFlagNotPresent) {
  ASSERT_FALSE(SearchNullaryOptionEverywhere(
      {"bazel", "build", ":target", "--"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereStartupOption) {
  ASSERT_TRUE(SearchNullaryOptionEverywhere(
      {"bazel", "--flag", "build", ":target", "--"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereStartupOptionWithEquals) {
  ASSERT_DEATH(SearchNullaryOptionEverywhere(
      {"bazel", "--flag=value", "build", ":target", "--"}, "--flag"),
               "In argument '--flag=value': option "
                   "'--flag' does not take a value");
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereCommandOption) {
  ASSERT_TRUE(SearchNullaryOptionEverywhere(
      {"bazel", "build", ":target", "--flag"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereReadsAfterPositionalParams) {
  ASSERT_TRUE(SearchNullaryOptionEverywhere(
      {"bazel", "build", ":target", "--", "--flag"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchNullaryEverywhereFailsAfterPositionalParams) {
  ASSERT_DEATH(SearchNullaryOptionEverywhere(
      {"bazel", "build", ":target", "--", "--flag=value"}, "--flag"),
               "In argument '--flag=value': option "
                   "'--flag' does not take a value");
}

TEST_F(BlazeUtilTest, TestSearchUnaryForEmpty) {
  ASSERT_STREQ(nullptr, SearchUnaryOption({"bazel", "build", ":target"}, ""));
}

TEST_F(BlazeUtilTest, TestSearchUnaryFlagNotPresent) {
  ASSERT_STREQ(nullptr,
               SearchUnaryOption({"bazel", "build", ":target"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnaryStartupOptionWithEquals) {
  ASSERT_STREQ("value",
               SearchUnaryOption({"bazel", "--flag=value", "build", ":target"},
                                 "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnaryStartupOptionWithoutEquals) {
  ASSERT_STREQ("value",
               SearchUnaryOption(
                   {"bazel", "--flag", "value", "build", ":target"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnaryCommandOptionWithEquals) {
  ASSERT_STREQ("value",
               SearchUnaryOption(
                   {"bazel", "build", ":target", "--flag", "value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnaryCommandOptionWithoutEquals) {
  ASSERT_STREQ("value",
               SearchUnaryOption(
                   {"bazel", "build", ":target", "--flag=value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnarySkipsAfterDashDashWithEquals) {
  ASSERT_STREQ(nullptr,
               SearchUnaryOption(
                   {"bazel", "build", ":target", "--", "--flag", "value"},
                   "--flag"));
}

TEST_F(BlazeUtilTest, TestSearchUnarySkipsAfterDashDashWithoutEquals) {
  ASSERT_STREQ(nullptr,
               SearchUnaryOption(
                   {"bazel", "build", ":target", "--", "--flag=value"},
                   "--flag"));
}

TEST_F(BlazeUtilTest, MakeAbsolute) {
#if defined(WIN32)
  EXPECT_EQ(MakeAbsolute("C:\\foo\\bar"), "C:\\foo\\bar");
  EXPECT_EQ(MakeAbsolute("C:/foo/bar"), "C:\\foo\\bar");
  EXPECT_EQ(MakeAbsolute("C:\\foo\\bar\\"), "C:\\foo\\bar\\");
  EXPECT_EQ(MakeAbsolute("C:/foo/bar/"), "C:\\foo\\bar\\");
  EXPECT_EQ(MakeAbsolute("foo"), blaze_util::GetCwd() + "\\foo");
#else
  EXPECT_EQ(MakeAbsolute("/foo/bar"), "/foo/bar");
  EXPECT_EQ(MakeAbsolute("/foo/bar/"), "/foo/bar/");
  EXPECT_EQ(MakeAbsolute("foo"), blaze_util::GetCwd() + "/foo");
#endif
  EXPECT_EQ(MakeAbsolute(std::string()), blaze_util::GetCwd());
  EXPECT_EQ(MakeAbsolute("/dev/null"), "/dev/null");
}

}  // namespace blaze
