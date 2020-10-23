// Copyright 2018 The Bazel Authors. All rights reserved.
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
#include "src/test/cpp/test_util.h"

#include "src/main/cpp/startup_options.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

// TODO(b/31290599): We should not only check that the options are registered,
// but also that they are parsed. StartupOptions* options would need to be
// non-const to call ProcessArgs and test that the value is recognized by the
// command line.
void ExpectValidNullaryOption(const StartupOptions* options,
                              const std::string& flag_name) {
  {
    bool result;
    std::string error;
    EXPECT_TRUE(
        options->MaybeCheckValidNullary("--" + flag_name, &result, &error));
    EXPECT_TRUE(result);
  }

  {
    bool result;
    std::string error;
    EXPECT_TRUE(
        options->MaybeCheckValidNullary("--no" + flag_name, &result, &error));
    EXPECT_TRUE(result);
  }

  {
    bool result;
    std::string error;
    EXPECT_TRUE(options->MaybeCheckValidNullary("--" + flag_name + "__invalid",
                                                &result, &error));
    EXPECT_FALSE(result);
  }

  {
    bool result;
    std::string error;
    EXPECT_FALSE(options->MaybeCheckValidNullary("--" + flag_name + "=foo",
                                                 &result, &error));
    EXPECT_EQ("In argument '--" + flag_name + "=foo': option '--" + flag_name +
                  "' does not take a value.",
              error);
  }

  {
    bool result;
    std::string error;
    EXPECT_FALSE(options->MaybeCheckValidNullary("--no" + flag_name + "=foo",
                                                 &result, &error));
    EXPECT_EQ("In argument '--no" + flag_name + "=foo': option '--no" +
                  flag_name + "' does not take a value.",
              error);
  }

  EXPECT_FALSE(options->IsUnary("--" + flag_name));
  EXPECT_FALSE(options->IsUnary("--no" + flag_name));
}

void ExpectIsUnaryOption(const StartupOptions* options,
                         const std::string& flag_name) {
  EXPECT_TRUE(options->IsUnary("--" + flag_name));
  EXPECT_TRUE(options->IsUnary("--" + flag_name + "="));
  EXPECT_TRUE(options->IsUnary("--" + flag_name + "=foo"));

  EXPECT_FALSE(options->IsUnary("--" + flag_name + "__invalid"));

  {
    bool result;
    std::string error;
    EXPECT_TRUE(
        options->MaybeCheckValidNullary("--" + flag_name, &result, &error));
    EXPECT_FALSE(result);
  }

  {
    bool result;
    std::string error;
    EXPECT_TRUE(
        options->MaybeCheckValidNullary("--no" + flag_name, &result, &error));
    EXPECT_FALSE(result);
  }
}

void ParseStartupOptionsAndExpectWarning(
    StartupOptions* startup_options,
    const std::vector<std::string>& options_to_parse,
    const std::string& expected_warning) {
  std::vector<RcStartupFlag> flags;
  flags.reserve(options_to_parse.size());
  for (const std::string& option : options_to_parse) {
    flags.push_back(RcStartupFlag("", option));
  }

  std::string error;
  EXPECT_EQ(blaze_exit_code::SUCCESS,
            startup_options->ProcessArgs(flags, &error));
  ASSERT_EQ("", error);

  testing::internal::CaptureStderr();
  startup_options->MaybeLogStartupOptionWarnings();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(expected_warning, output);
}

}  // namespace blaze
