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

#include "src/main/cpp/bazel_startup_options.h"

#include <stdlib.h>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/workspace_layout.h"
#include "src/test/cpp/test_util.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

class BazelStartupOptionsTest : public ::testing::Test {
 protected:
  BazelStartupOptionsTest() : workspace_layout_(new WorkspaceLayout()) {}
  ~BazelStartupOptionsTest() = default;

  void SetUp() override {
    // This knowingly ignores the possibility of these environment variables
    // being unset because we expect our test runner to set them in all cases.
    // Otherwise, we'll crash here, but this keeps our code simpler.
    old_test_tmpdir_ = GetEnv("TEST_TMPDIR");

    ReinitStartupOptions();
  }

  void TearDown() override { SetEnv("TEST_TMPDIR", old_test_tmpdir_); }

  // Recreates startup_options_ after changes to the environment.
  void ReinitStartupOptions() {
    startup_options_.reset(new BazelStartupOptions(workspace_layout_.get()));
  }

 private:
  std::unique_ptr<WorkspaceLayout> workspace_layout_;

 protected:
  std::unique_ptr<BazelStartupOptions> startup_options_;

 private:
  std::string old_test_tmpdir_;
};

TEST_F(BazelStartupOptionsTest, ProductName) {
  ASSERT_EQ("Bazel", startup_options_->product_name);
}

TEST_F(BazelStartupOptionsTest, JavaLoggingOptions) {
  ASSERT_EQ("com.google.devtools.build.lib.util.SingleLineFormatter",
            startup_options_->java_logging_formatter);
}

TEST_F(BazelStartupOptionsTest, EmptyFlagsAreInvalid) {
  EXPECT_FALSE(startup_options_->IsNullary(""));
  EXPECT_FALSE(startup_options_->IsNullary("--"));
  EXPECT_FALSE(startup_options_->IsUnary(""));
  EXPECT_FALSE(startup_options_->IsUnary("--"));
}

// TODO(#4502 related cleanup) This test serves as a catalog of the valid
// options - make this test check that the list is complete, that no options are
// missing.
TEST_F(BazelStartupOptionsTest, ValidStartupFlags) {
  // IMPORTANT: Before modifying this test, please contact a Bazel core team
  // member that knows the Google-internal procedure for adding/deprecating
  // startup flags.
  const StartupOptions* options = startup_options_.get();
  ExpectIsNullaryOption(options, "batch");
  ExpectIsNullaryOption(options, "batch_cpu_scheduling");
  ExpectIsNullaryOption(options, "block_for_lock");
  ExpectIsNullaryOption(options, "client_debug");
  ExpectIsNullaryOption(options, "deep_execroot");
  ExpectIsNullaryOption(options, "experimental_oom_more_eagerly");
  ExpectIsNullaryOption(options, "fatal_event_bus_exceptions");
  ExpectIsNullaryOption(options, "host_jvm_debug");
  ExpectIsNullaryOption(options, "master_bazelrc");
  ExpectIsNullaryOption(options, "watchfs");
  ExpectIsNullaryOption(options, "write_command_log");
  ExpectIsUnaryOption(options, "bazelrc");
  ExpectIsUnaryOption(options, "command_port");
  ExpectIsUnaryOption(options, "connect_timeout_secs");
  ExpectIsUnaryOption(options, "experimental_oom_more_eagerly_threshold");
  ExpectIsUnaryOption(options, "host_javabase");
  ExpectIsUnaryOption(options, "host_jvm_args");
  ExpectIsUnaryOption(options, "host_jvm_profile");
  ExpectIsUnaryOption(options, "invocation_policy");
  ExpectIsUnaryOption(options, "io_nice_level");
  ExpectIsUnaryOption(options, "install_base");
  ExpectIsUnaryOption(options, "max_idle_secs");
  ExpectIsUnaryOption(options, "output_base");
  ExpectIsUnaryOption(options, "output_user_root");
}

TEST_F(BazelStartupOptionsTest, BlazercFlagsAreNotAccepted) {
  EXPECT_FALSE(startup_options_->IsNullary("--master_blazerc"));
  EXPECT_FALSE(startup_options_->IsUnary("--master_blazerc"));
  EXPECT_FALSE(startup_options_->IsNullary("--blazerc"));
  EXPECT_FALSE(startup_options_->IsUnary("--blazerc"));
}

}  // namespace blaze
