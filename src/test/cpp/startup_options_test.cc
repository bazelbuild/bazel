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

#include "src/main/cpp/startup_options.h"

#include <stdlib.h>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/workspace_layout.h"
#include "gtest/gtest.h"

namespace blaze {

class StartupOptionsTest : public ::testing::Test {
 protected:
  StartupOptionsTest() : workspace_layout_(new WorkspaceLayout()) {}
  ~StartupOptionsTest() = default;

  void SetUp() override {
    // This knowingly ignores the possibility of these environment variables
    // being unset because we expect our test runner to set them in all cases.
    // Otherwise, we'll crash here, but this keeps our code simpler.
    old_home_ = GetHomeDir();
    old_test_tmpdir_ = GetEnv("TEST_TMPDIR");

    ReinitStartupOptions();
  }

  void TearDown() override {
    SetEnv("HOME", old_home_);
    SetEnv("TEST_TMPDIR", old_test_tmpdir_);
  }

  // Recreates startup_options_ after changes to the environment.
  void ReinitStartupOptions() {
    startup_options_.reset(new StartupOptions(workspace_layout_.get()));
  }

  void SuccessfulIsNullaryTest(const std::string& flag_name) const {
    EXPECT_TRUE(startup_options_->IsNullary("--" + flag_name));
    EXPECT_TRUE(startup_options_->IsNullary("--no" + flag_name));

    EXPECT_FALSE(startup_options_->IsNullary("--" + flag_name + "__invalid"));

    EXPECT_DEATH(startup_options_->IsNullary("--" + flag_name + "=foo"),
                 ("In argument '--" + flag_name + "=foo': option "
                     "'--" + flag_name + "' does not take a value").c_str());

    EXPECT_DEATH(startup_options_->IsNullary("--no" + flag_name + "=foo"),
                 ("In argument '--no" + flag_name + "=foo': option "
                     "'--no" + flag_name + "' does not take a value").c_str());

    EXPECT_FALSE(startup_options_->IsUnary("--" + flag_name));
    EXPECT_FALSE(startup_options_->IsUnary("--no" + flag_name));
  }

  void SuccessfulIsUnaryTest(const std::string& flag_name) const {
    EXPECT_TRUE(startup_options_->IsUnary("--" + flag_name));
    EXPECT_TRUE(startup_options_->IsUnary("--" + flag_name + "="));
    EXPECT_TRUE(startup_options_->IsUnary("--" + flag_name + "=foo"));

    EXPECT_FALSE(startup_options_->IsUnary("--" + flag_name + "__invalid"));
    EXPECT_FALSE(startup_options_->IsNullary("--" + flag_name));
    EXPECT_FALSE(startup_options_->IsNullary("--no" + flag_name));
  }

 private:
  std::unique_ptr<WorkspaceLayout> workspace_layout_;

 protected:
  std::unique_ptr<StartupOptions> startup_options_;

 private:
  std::string old_home_;
  std::string old_test_tmpdir_;
};

TEST_F(StartupOptionsTest, ProductName) {
  ASSERT_EQ("Bazel", startup_options_->product_name);
}

TEST_F(StartupOptionsTest, JavaLoggingOptions) {
  ASSERT_EQ("com.google.devtools.build.lib.util.SingleLineFormatter",
      startup_options_->java_logging_formatter);
}

// TODO(bazel-team): remove the ifdef guard once the implementation of
// GetOutputRoot is stable among the different platforms.
#ifdef __linux
TEST_F(StartupOptionsTest, OutputRootPreferTestTmpdirIfSet) {
  SetEnv("HOME", "/nonexistent/home");
  SetEnv("TEST_TMPDIR", "/nonexistent/tmpdir");
  ReinitStartupOptions();

  ASSERT_EQ("/nonexistent/tmpdir", startup_options_->output_root);
}

TEST_F(StartupOptionsTest, OutputRootUseHomeDirectory) {
  SetEnv("HOME", "/nonexistent/home");
  UnsetEnv("TEST_TMPDIR");
  ReinitStartupOptions();

  ASSERT_EQ("/nonexistent/home/.cache/bazel", startup_options_->output_root);
}
#endif  // __linux

TEST_F(StartupOptionsTest, EmptyFlagsAreInvalidTest) {
  EXPECT_FALSE(startup_options_->IsNullary(""));
  EXPECT_FALSE(startup_options_->IsNullary("--"));
  EXPECT_FALSE(startup_options_->IsUnary(""));
  EXPECT_FALSE(startup_options_->IsUnary("--"));
}

TEST_F(StartupOptionsTest, ValidStartupFlagsTest) {
  // IMPORTANT: Before modifying this test, please contact a Bazel core team
  // member that knows the Google-internal procedure for adding/deprecating
  // startup flags.
  SuccessfulIsNullaryTest("allow_configurable_attributes");
  SuccessfulIsNullaryTest("batch");
  SuccessfulIsNullaryTest("batch_cpu_scheduling");
  SuccessfulIsNullaryTest("block_for_lock");
  SuccessfulIsNullaryTest("client_debug");
  SuccessfulIsNullaryTest("deep_execroot");
  SuccessfulIsNullaryTest("experimental_oom_more_eagerly");
  SuccessfulIsNullaryTest("fatal_event_bus_exceptions");
  SuccessfulIsNullaryTest("host_jvm_debug");
  SuccessfulIsNullaryTest("master_bazelrc");
  SuccessfulIsNullaryTest("master_blazerc");
  SuccessfulIsNullaryTest("watchfs");
  SuccessfulIsNullaryTest("write_command_log");
  SuccessfulIsUnaryTest("bazelrc");
  SuccessfulIsUnaryTest("blazerc");
  SuccessfulIsUnaryTest("command_port");
  SuccessfulIsUnaryTest("connect_timeout_secs");
  SuccessfulIsUnaryTest("experimental_oom_more_eagerly_threshold");
  SuccessfulIsUnaryTest("host_javabase");
  SuccessfulIsUnaryTest("host_jvm_args");
  SuccessfulIsUnaryTest("host_jvm_profile");
  SuccessfulIsUnaryTest("invocation_policy");
  SuccessfulIsUnaryTest("io_nice_level");
  SuccessfulIsUnaryTest("install_base");
  SuccessfulIsUnaryTest("max_idle_secs");
  SuccessfulIsUnaryTest("output_base");
  SuccessfulIsUnaryTest("output_user_root");
}

}  // namespace blaze
