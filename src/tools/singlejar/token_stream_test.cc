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

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "src/tools/singlejar/test_util.h"
#include "src/tools/singlejar/token_stream.h"
#include "googletest/include/gtest/gtest.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

// Indirect command file contents (each string is a separate line):
static const char *lines[] = {
    "-cmd1 foo", "bar",       "'abcd'",    "\"efg\"", "hi'x'",
    "'\\jkl'",   "\"\\xyz\"", "\"\\\"0\"", "cont\\",  "inue x",
};

// Tokens that the tokenizer is expected to return from the
// indirect file above:
static const char *expected_tokens[] = {
    "-cmd1", "foo",   "bar", "abcd",     "efg", "hix",
    "\\jkl", "\\xyz", "\"0", "continue", "x",
};

// Simple '-foo -bar' command line.
TEST(TokenStreamTest, SimpleArgs) {
  const char *args[] = {"-foo", "-bar"};
  ArgTokenStream token_stream(ARRAY_SIZE(args), args);
  EXPECT_EQ("-foo", token_stream.token());
  bool flag_foo = false;
  EXPECT_FALSE(token_stream.MatchAndSet("-bar", &flag_foo));
  ASSERT_TRUE(token_stream.MatchAndSet("-foo", &flag_foo));
  EXPECT_TRUE(flag_foo);
  bool flag_bar = false;
  ASSERT_TRUE(token_stream.MatchAndSet("-bar", &flag_bar));
  EXPECT_TRUE(flag_bar);
  EXPECT_TRUE(token_stream.AtEnd());
}

// '-foo @commandfile -bar' command line.
TEST(TokenStreamTest, CommandFile) {
  const char *tempdir = getenv("TEST_TMPDIR");
  ASSERT_NE(nullptr, tempdir);
  std::string command_file_path = singlejar_test_util::OutputFilePath("tokens");
  FILE *fp = fopen(command_file_path.c_str(), "w");
  ASSERT_NE(nullptr, fp);
  for (size_t i = 0; i < ARRAY_SIZE(lines); ++i) {
    fprintf(fp, "%s\n", lines[i]);
  }
  fclose(fp);

  std::string command_file_arg = std::string("@") + command_file_path;
  const char *args[] = {"-before_file", "", "-after_file"};
  args[1] = command_file_arg.c_str();
  ArgTokenStream token_stream(ARRAY_SIZE(args), args);
  bool flag = false;
  ASSERT_TRUE(token_stream.MatchAndSet("-before_file", &flag));
  EXPECT_TRUE(flag);
  for (size_t i = 0; i < ARRAY_SIZE(expected_tokens); ++i) {
    flag = false;
    ASSERT_TRUE(token_stream.MatchAndSet(expected_tokens[i], &flag));
    EXPECT_TRUE(flag);
  }
  ASSERT_TRUE(token_stream.MatchAndSet("-after_file", &flag));
  EXPECT_TRUE(flag);
  EXPECT_TRUE(token_stream.AtEnd());
}

// '--arg1 optval1 --arg2' command line.
TEST(TokenStreamTest, OptargOne) {
  const char *args[] = {"--arg1", "optval1", "--arg2", "--arg3", "optval3"};
  ArgTokenStream token_stream(ARRAY_SIZE(args), args);
  std::string optval;
  EXPECT_FALSE(token_stream.MatchAndSet("--foo", &optval));
  ASSERT_TRUE(token_stream.MatchAndSet("--arg1", &optval));
  EXPECT_EQ("optval1", optval);
  bool flag = true;
  ASSERT_TRUE(token_stream.MatchAndSet("--arg2", &flag));
  EXPECT_TRUE(flag);
  ASSERT_TRUE(token_stream.MatchAndSet("--arg3", &optval));
  EXPECT_EQ("optval3", optval);
  EXPECT_TRUE(token_stream.AtEnd());
}

// '--arg1 value1 value2 --arg2' command line.
TEST(TokenStreamTest, OptargMulti) {
  const char *args[] = {"--arg1", "value11", "value12",
                        "--arg2", "value21", "value22"};
  ArgTokenStream token_stream(ARRAY_SIZE(args), args);
  std::vector<std::string> optvals1;
  EXPECT_FALSE(token_stream.MatchAndSet("--arg2", &optvals1));
  ASSERT_TRUE(token_stream.MatchAndSet("--arg1", &optvals1));
  ASSERT_EQ(2UL, optvals1.size());
  EXPECT_EQ("value11", optvals1[0]);
  EXPECT_EQ("value12", optvals1[1]);

  std::vector<std::string> optvals2;
  ASSERT_TRUE(token_stream.MatchAndSet("--arg2", &optvals2));
  ASSERT_EQ(2UL, optvals2.size());
  EXPECT_EQ("value21", optvals2[0]);
  EXPECT_EQ("value22", optvals2[1]);

  EXPECT_TRUE(token_stream.AtEnd());
}

// '--arg1 optval1,optsuff1 optval2,optstuff2 --arg2' command line.
TEST(TokenStreamTest, OptargMultiSplit) {
  const char *args[] = {"--arg1", "optval1,optsuff1", "optval2,optsuff2",
                        "optvalnosuff"};
  ArgTokenStream token_stream(ARRAY_SIZE(args), args);
  std::vector<std::pair<std::string, std::string> > optvals1;

  EXPECT_FALSE(token_stream.MatchAndSet("--foo", &optvals1));
  ASSERT_TRUE(token_stream.MatchAndSet("--arg1", &optvals1));

  ASSERT_EQ(3UL, optvals1.size());
  EXPECT_EQ("optval1", optvals1[0].first);
  EXPECT_EQ("optsuff1", optvals1[0].second);
  EXPECT_EQ("optval2", optvals1[1].first);
  EXPECT_EQ("optsuff2", optvals1[1].second);
  EXPECT_EQ("optvalnosuff", optvals1[2].first);
  EXPECT_EQ("", optvals1[2].second);
}
