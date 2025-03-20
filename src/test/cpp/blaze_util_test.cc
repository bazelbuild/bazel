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

#include "src/main/cpp/blaze_util.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "src/main/cpp/util/file.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

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
};

TEST_F(BlazeUtilTest, ReadFrom) { ReadFromTest(); }

TEST_F(BlazeUtilTest, TestSearchNullaryEmptyCase) {
  ASSERT_FALSE(SearchNullaryOption({}, "flag", false));
  ASSERT_TRUE(SearchNullaryOption({}, "flag", true));
}

TEST_F(BlazeUtilTest, TestSearchNullaryForEmpty) {
  ASSERT_TRUE(SearchNullaryOption({"bazel", "build", ":target"}, "", true));
  ASSERT_FALSE(SearchNullaryOption({"bazel", "build", ":target"}, "", false));
}

TEST_F(BlazeUtilTest, TestSearchNullaryForFlagNotPresent) {
  ASSERT_FALSE(SearchNullaryOption({"bazel", "build", ":target"},
                                   "flag", false));
  ASSERT_TRUE(SearchNullaryOption({"bazel", "build", ":target"},
                                   "flag", true));
}

TEST_F(BlazeUtilTest, TestSearchNullaryStartupOption) {
  ASSERT_TRUE(SearchNullaryOption({"bazel", "--flag", "build", ":target"},
                                  "flag", false));
  ASSERT_TRUE(SearchNullaryOption({"bazel", "--flag", "build", ":target"},
                                  "flag", true));
}

TEST_F(BlazeUtilTest, TestSearchNullaryStartupOptionWithEquals) {
  ASSERT_DEATH(SearchNullaryOption(
      {"bazel", "--flag=value", "build", ":target"}, "flag", false),
              "In argument '--flag=value': option "
              "'--flag' does not take a value");
}

TEST_F(BlazeUtilTest, TestSearchNullaryCommandOption) {
  ASSERT_TRUE(SearchNullaryOption({"bazel", "build", ":target", "--flag"},
                                  "flag", false));
}

TEST_F(BlazeUtilTest, TestSearchNullarySkipsAfterDashDash) {
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "build", ":target", "--", "--flag"}, "flag", false));
}

TEST_F(BlazeUtilTest, TestSearchNullarySucceedsWithEqualsAndDashDash) {
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "build", ":target", "--", "--flag=value"}, "flag", false));
}

TEST_F(BlazeUtilTest, TestSearchNullaryLastFlagWins) {
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "--flag", "--noflag", "build"}, "flag", false));
  ASSERT_FALSE(SearchNullaryOption(
      {"bazel", "--flag", "--noflag", "build"}, "flag", true));
}

void assert_equal_vector_char_pointer(std::vector<std::string> expected,
                                      std::vector<std::string> actual) {
  ASSERT_THAT(actual, testing::ContainerEq(expected));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryForEmpty) {
  assert_equal_vector_char_pointer(
      {}, GetAllUnaryOptionValues({"bazel", "build", ":target"}, ""));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryFlagNotPresent) {
  assert_equal_vector_char_pointer(
      {}, GetAllUnaryOptionValues({"bazel", "build", ":target"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithEquals) {
  assert_equal_vector_char_pointer(
      {"value"}, GetAllUnaryOptionValues(
                     {"bazel", "--flag=value", "build", ":target"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithEquals2) {
  assert_equal_vector_char_pointer(
      {"value1", "value2"},
      GetAllUnaryOptionValues(
          {"bazel", "--flag=value1", "--flag=value2", "build", ":target"},
          "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithRepeatingFlag) {
  assert_equal_vector_char_pointer(
      {"--flag"}, GetAllUnaryOptionValues({"bazel", "--flag", "--flag",
                                           "value1", "build", ":target"},
                                          "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithRepeatingFlagOptions) {
  assert_equal_vector_char_pointer(
      {"--flag"}, GetAllUnaryOptionValues(
                      {"bazel", "--flag", "--flag", "value1"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionValuesWithEquals) {
  assert_equal_vector_char_pointer(
      {"--flag", "value1"},
      GetAllUnaryOptionValues({"bazel", "--flag=--flag", "--flag", "value1"},
                              "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithEquals3) {
  assert_equal_vector_char_pointer(
      {"value1", "value2", "value3"},
      GetAllUnaryOptionValues({"bazel", "--flag=value1", "--flag=value2",
                               "--flag=value3", "build", ":target"},
                              "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithoutEquals) {
  assert_equal_vector_char_pointer(
      {"value"},
      GetAllUnaryOptionValues({"bazel", "--flag", "value", "build", ":target"},
                              "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryOptionWithoutEquals2) {
  assert_equal_vector_char_pointer(
      {"value1", "value2"},
      GetAllUnaryOptionValues(
          {"bazel", "--flag", "value1", "--flag", "value2", "build", ":target"},
          "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithEquals) {
  assert_equal_vector_char_pointer(
      {"value"},
      GetAllUnaryOptionValues({"bazel", "build", ":target", "--flag", "value"},
                              "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithEquals2) {
  assert_equal_vector_char_pointer(
      {"value1", "value2"},
      GetAllUnaryOptionValues(
          {"bazel", "build", ":target", "--flag", "value1", "--flag", "value2"},
          "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithoutEquals) {
  assert_equal_vector_char_pointer(
      {"value"}, GetAllUnaryOptionValues(
                     {"bazel", "build", ":target", "--flag=value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithoutEquals2) {
  assert_equal_vector_char_pointer(
      {"value1", "value2"},
      GetAllUnaryOptionValues(
          {"bazel", "build", ":target", "--flag=value1", "--flag=value2"},
          "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnarySkipsAfterDashDashWithEquals) {
  assert_equal_vector_char_pointer(
      {},
      GetAllUnaryOptionValues(
          {"bazel", "build", ":target", "--", "--flag", "value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnarySkipsAfterDashDashWithoutEquals) {
  assert_equal_vector_char_pointer(
      {}, GetAllUnaryOptionValues(
              {"bazel", "build", ":target", "--", "--flag=value"}, "--flag"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithIgnoreAfter) {
  assert_equal_vector_char_pointer(
      {"value1", "/dev/null"},
      GetAllUnaryOptionValues({"bazel", "build", ":target", "--flag", "value1",
                               "--flag", "/dev/null", "--flag", "value3"},
                              "--flag", "/dev/null"));
}

TEST_F(BlazeUtilTest, TestGetAllUnaryCommandOptionWithIgnoreAfterDevNull) {
  assert_equal_vector_char_pointer(
      {"/dev/null"}, GetAllUnaryOptionValues(
                         {"bazel", "build", ":target", "--flag", "/dev/null",
                          "--flag", "value2", "--flag", "value3"},
                         "--flag", "/dev/null"));
}

}  // namespace blaze
