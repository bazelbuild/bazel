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

#include "src/main/cpp/blaze_startup_options.h"
#include "gtest/gtest.h"

namespace blaze {

class BlazeStartupOptionsTest : public ::testing::Test {
 protected:
  BlazeStartupOptionsTest() = default;
  ~BlazeStartupOptionsTest() = default;

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

TEST_F(BlazeStartupOptionsTest, OutputRootPreferTestTmpdirIfSet) {
  setenv("HOME", "/nonexistent/home", 1);
  setenv("TEST_TMPDIR", "/nonexistent/tmpdir", 1);

  blaze::BlazeStartupOptions startup_options;
  ASSERT_EQ("/nonexistent/tmpdir", startup_options.output_root);
}

TEST_F(BlazeStartupOptionsTest, OutputRootUseHomeDirectory) {
  setenv("HOME", "/nonexistent/home", 1);
  unsetenv("TEST_TMPDIR");

  blaze::BlazeStartupOptions startup_options;
  ASSERT_EQ("/nonexistent/home/.cache/bazel", startup_options.output_root);
}

TEST_F(BlazeStartupOptionsTest, OutputRootUseBuiltin) {
  // We cannot just unsetenv("HOME") because the logic to compute the output
  // root falls back to using the passwd database if HOME is null... and mocking
  // that out is hard.
  setenv("HOME", "", 1);
  unsetenv("TEST_TMPDIR");

  blaze::BlazeStartupOptions startup_options;
  ASSERT_EQ("/tmp", startup_options.output_root);
}

}  // namespace blaze
