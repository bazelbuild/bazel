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

#include <stdlib.h>

#include "src/main/cpp/startup_options.h"
#include "gtest/gtest.h"

namespace blaze {

class StartupOptionsTest : public ::testing::Test {
 protected:
  StartupOptionsTest() = default;
  ~StartupOptionsTest() = default;

  void SetUp() override {
    // This knowingly ignores the possibility of these environment variables
    // being unset because we expect our test runner to set them in all cases.
    // Otherwise, we'll crash here, but this keeps our code simpler.
    old_home_ = getenv("HOME");
    old_test_tmpdir_ = getenv("TEST_TMPDIR");
  }

  void TearDown() override {
    setenv("HOME", old_home_.c_str(), 1);
    setenv("TEST_TMPDIR", old_test_tmpdir_.c_str(), 1);
  }

 private:
  std::string old_home_;
  std::string old_test_tmpdir_;
};

TEST_F(StartupOptionsTest, ProductName) {
  blaze::StartupOptions startup_options;
  ASSERT_EQ("Bazel", startup_options.product_name);
}

TEST_F(StartupOptionsTest, OutputRootPreferTestTmpdirIfSet) {
  setenv("HOME", "/nonexistent/home", 1);
  setenv("TEST_TMPDIR", "/nonexistent/tmpdir", 1);

  blaze::StartupOptions startup_options;
  ASSERT_EQ("/nonexistent/tmpdir", startup_options.output_root);
}

TEST_F(StartupOptionsTest, OutputRootUseHomeDirectory) {
  setenv("HOME", "/nonexistent/home", 1);
  unsetenv("TEST_TMPDIR");

  blaze::StartupOptions startup_options;
  ASSERT_EQ("/nonexistent/home/.cache/bazel", startup_options.output_root);
}

TEST_F(StartupOptionsTest, OutputRootUseBuiltin) {
  // We cannot just unsetenv("HOME") because the logic to compute the output
  // root falls back to using the passwd database if HOME is null... and mocking
  // that out is hard.
  setenv("HOME", "", 1);
  unsetenv("TEST_TMPDIR");

  blaze::StartupOptions startup_options;
  ASSERT_EQ("/tmp", startup_options.output_root);
}

TEST_F(StartupOptionsTest, IsNullaryTest) {
  blaze::StartupOptions startup_options;
  ASSERT_TRUE(startup_options.IsNullary("--master_bazelrc"));
  ASSERT_TRUE(startup_options.IsNullary("--nomaster_bazelrc"));
  ASSERT_FALSE(startup_options.IsNullary(""));
  ASSERT_FALSE(startup_options.IsNullary("--"));
  ASSERT_FALSE(startup_options.IsNullary("--master_bazelrcascasc"));
  string error_msg = std::string("In argument '--master_bazelrc=foo': option ")
      + std::string("'--master_bazelrc' does not take a value");
  ASSERT_DEATH(startup_options.IsNullary("--master_bazelrc=foo"),
               error_msg.c_str());
}

TEST_F(StartupOptionsTest, IsUnaryTest) {
  blaze::StartupOptions startup_options;
  ASSERT_FALSE(startup_options.IsUnary("", ""));
  ASSERT_FALSE(startup_options.IsUnary("--", ""));

  ASSERT_TRUE(startup_options.IsUnary("--blazerc=foo", "--blah"));
  ASSERT_TRUE(startup_options.IsUnary("--blazerc", "foo"));
  ASSERT_TRUE(startup_options.IsUnary("--blazerc=", "--foo"));
  ASSERT_TRUE(startup_options.IsUnary("--blazerc", ""));
  ASSERT_FALSE(startup_options.IsUnary("--blazercfooblah", "foo"));
}

}  // namespace blaze
