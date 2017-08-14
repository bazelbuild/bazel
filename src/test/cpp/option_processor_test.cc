// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include "src/main/cpp/option_processor.h"

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor-internal.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/workspace_layout.h"
#include "gtest/gtest.h"
#include "re2/re2.h"

namespace blaze {

class OptionProcessorTest : public ::testing::Test {
 protected:
  OptionProcessorTest() :
      workspace_(
          blaze_util::JoinPath(blaze::GetEnv("TEST_TMPDIR"), "testdir")),
      cwd_("cwd"),
      workspace_layout_(new WorkspaceLayout()) {}

  ~OptionProcessorTest() override {}

  void SetUp() override {
    ASSERT_TRUE(blaze_util::MakeDirectories(workspace_, 0755));
    option_processor_.reset(new OptionProcessor(
        workspace_layout_.get(),
        std::unique_ptr<StartupOptions>(
            new StartupOptions(workspace_layout_.get()))));
  }

  void TearDown() override {
    // TODO(bazel-team): The code below deletes all the files in the workspace
    // but it intentionally skips directories. As a consequence, there may be
    // empty directories from test to test. Remove this once
    // blaze_util::DeleteDirectories(path) exists.
    std::vector<std::string> files_in_workspace;
    blaze_util::GetAllFilesUnder(workspace_, &files_in_workspace);
    for (const std::string& file : files_in_workspace) {
      blaze_util::UnlinkPath(file);
    }
  }

  void FailedSplitStartupOptionsTest(const std::vector<std::string>& args,
                                     const std::string& expected_error) const {
    std::string error;
    const std::unique_ptr<CommandLine> result =
        option_processor_->SplitCommandLine(args, &error);
    ASSERT_EQ(expected_error, error);
    ASSERT_EQ(nullptr, result);
  }

  void SuccessfulSplitStartupOptionsTest(const std::vector<std::string>& args,
                                         const CommandLine& expected) const {
    std::string error;
    const std::unique_ptr<CommandLine> result =
        option_processor_->SplitCommandLine(args, &error);

    ASSERT_EQ("", error);
    EXPECT_EQ(expected.path_to_binary, result->path_to_binary);
    EXPECT_EQ(expected.startup_args, result->startup_args);
    EXPECT_EQ(expected.command, result->command);
    EXPECT_EQ(expected.command_args, result->command_args);
  }

  const std::string workspace_;
  const std::string cwd_;
  const std::unique_ptr<WorkspaceLayout> workspace_layout_;
  std::unique_ptr<OptionProcessor> option_processor_;
};

TEST_F(OptionProcessorTest, CanParseOptions) {
  const std::vector<std::string> args =
      {"bazel",
       "--host_jvm_args=MyParam", "--nobatch",
       "command",
       "--flag", "//my:target", "--flag2=42"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;

  ASSERT_EQ("", error);
  ASSERT_EQ(1,
            option_processor_->GetParsedStartupOptions()->host_jvm_args.size());
  EXPECT_EQ("MyParam",
            option_processor_->GetParsedStartupOptions()->host_jvm_args[0]);
  EXPECT_FALSE(option_processor_->GetParsedStartupOptions()->batch);

  EXPECT_EQ("command", option_processor_->GetCommand());

  EXPECT_EQ(std::vector<std::string>({"--flag", "//my:target", "--flag2=42"}),
            option_processor_->GetExplicitCommandArguments());
}

TEST_F(OptionProcessorTest, CanParseHelpArgs) {
  const std::vector<std::string> args =
      {"bazel",
       "--host_jvm_args=MyParam", "--nobatch",
       "help",
       "--flag", "//my:target", "--flag2=42"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;

  ASSERT_EQ("", error);
  ASSERT_EQ(1,
            option_processor_->GetParsedStartupOptions()->host_jvm_args.size());
  EXPECT_EQ("MyParam",
            option_processor_->GetParsedStartupOptions()->host_jvm_args[0]);
  EXPECT_FALSE(option_processor_->GetParsedStartupOptions()->batch);

  EXPECT_EQ("help", option_processor_->GetCommand());

  EXPECT_EQ(std::vector<std::string>({"--flag", "//my:target", "--flag2=42"}),
            option_processor_->GetExplicitCommandArguments());
}

TEST_F(OptionProcessorTest, CanParseEmptyArgs) {
  const std::vector<std::string> args = {"bazel"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;
  ASSERT_EQ("", error);

  EXPECT_EQ("", option_processor_->GetCommand());

  EXPECT_EQ(std::vector<std::string>({}),
            option_processor_->GetExplicitCommandArguments());
}

TEST_F(OptionProcessorTest, CanParseDifferentStartupArgs) {
  const std::vector<std::string> args =
      {"bazel",
       "--nobatch", "--host_jvm_args=MyParam", "--host_jvm_args", "42"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;
  ASSERT_EQ("", error);

  ASSERT_EQ(2,
            option_processor_->GetParsedStartupOptions()->host_jvm_args.size());
  EXPECT_EQ("MyParam",
            option_processor_->GetParsedStartupOptions()->host_jvm_args[0]);
  EXPECT_EQ("42",
            option_processor_->GetParsedStartupOptions()->host_jvm_args[1]);

  EXPECT_EQ("", option_processor_->GetCommand());

  EXPECT_EQ(std::vector<std::string>({}),
            option_processor_->GetExplicitCommandArguments());
}

TEST_F(OptionProcessorTest, CommandLineBazelrcTest) {
  const std::string cmdline_rc_path =
      blaze_util::JoinPath(workspace_, "mybazelrc");
  ASSERT_TRUE(blaze_util::MakeDirectories(
      blaze_util::Dirname(cmdline_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --foo", cmdline_rc_path, 0755));

  const std::vector<std::string> args =
      {"bazel", "--bazelrc=" + cmdline_rc_path, "build"};
  const std::string expected_error =
      "Unknown startup option: '--foo'.\n"
          "  For more info, run 'bazel help startup_options'.";
  std::string error;
  ASSERT_NE(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;
  ASSERT_EQ(expected_error, error);

  // Check that the startup option option provenance message prints the correct
  // information for the incorrect flag, and does not print the command-line
  // provided startup flags.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_PRED1(
      [](std::string actualOutput) {
        return RE2::FullMatch(
            actualOutput,
            "INFO: Reading 'startup' options from .*mybazelrc: "
            "--foo\n");
      },
      output);
}

TEST_F(OptionProcessorTest, NoMasterBazelrcAndBazelrcWorkTogetherCorrectly) {
  const std::string cmdline_rc_path =
      blaze_util::JoinPath(workspace_, "mybazelrc");
  ASSERT_TRUE(blaze_util::MakeDirectories(
      blaze_util::Dirname(cmdline_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --max_idle_secs=123",
                                    cmdline_rc_path, 0755));

  const std::string master_rc_path =
      blaze_util::JoinPath(workspace_, "tools/bazel.rc");
  ASSERT_TRUE(blaze_util::MakeDirectories(
      blaze_util::Dirname(master_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --foo", master_rc_path, 0755));

  const std::vector<std::string> args =
      {"bazel",
       "--bazelrc=" + cmdline_rc_path, "--nomaster_bazelrc",
       "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
                << error;

  EXPECT_EQ(123, option_processor_->GetParsedStartupOptions()->max_idle_secs);

  // Check that the startup option option provenance message prints the correct
  // information for the provided rc, and prints nothing for the master bazelrc.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_PRED1(
      [](std::string actualOutput) {
        return RE2::FullMatch(
            actualOutput,
            "INFO: Reading 'startup' options from .*mybazelrc: "
            "--max_idle_secs=123\n");
      },
      output);
}

TEST_F(OptionProcessorTest, MultipleStartupArgsInMasterBazelrcWorksCorrectly) {
  // Add startup flags to the master bazelrc.
  const std::string master_rc_path =
      blaze_util::JoinPath(workspace_, "tools/bazel.rc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(master_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile(
      "startup --max_idle_secs=42\nstartup --io_nice_level=6", master_rc_path,
      0755));

  const std::vector<std::string> args = {"bazel", "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;

  EXPECT_EQ(42, option_processor_->GetParsedStartupOptions()->max_idle_secs);
  EXPECT_EQ(6, option_processor_->GetParsedStartupOptions()->io_nice_level);

  // Check that the startup options get grouped together properly in the output
  // message.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_PRED1(
      [](std::string actualOutput) {
        return RE2::FullMatch(
            actualOutput,
            "INFO: Reading 'startup' options from .*tools.*bazel.rc: "
            "--max_idle_secs=42 --io_nice_level=6\n");
      },
      output);
}

TEST_F(OptionProcessorTest, CustomBazelrcOverridesMasterBazelrc) {
  // Add startup flags to the master bazelrc.
  const std::string master_rc_path =
      blaze_util::JoinPath(workspace_, "tools/bazel.rc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(master_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile(
      "startup --max_idle_secs=42\nstartup --io_nice_level=6", master_rc_path,
      0755));

  // Override one of the master bazelrc's flags in the custom bazelrc.
  const std::string cmdline_rc_path =
      blaze_util::JoinPath(workspace_, "mybazelrc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(cmdline_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --max_idle_secs=123",
                                    cmdline_rc_path, 0755));
  const std::vector<std::string> args = {
      "bazel", "--bazelrc=" + cmdline_rc_path, "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;

  EXPECT_EQ(123, option_processor_->GetParsedStartupOptions()->max_idle_secs);
  EXPECT_EQ(6, option_processor_->GetParsedStartupOptions()->io_nice_level);

  // Check that the options are reported in the correct order in the provenance
  // message.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_PRED1(
      [](std::string actualOutput) {
        return RE2::FullMatch(
            actualOutput,
            "INFO: Reading 'startup' options from .*tools.*bazel.rc: "
            "--max_idle_secs=42 --io_nice_level=6\n"
            "INFO: Reading 'startup' options from .*mybazelrc: "
            "--max_idle_secs=123\n");
      },
      output);
}

TEST_F(OptionProcessorTest, BazelRcImportsMaintainsFlagOrdering) {
  // Override one of the master bazelrc's flags in the custom bazelrc.
  const std::string imported_rc_path =
      blaze_util::JoinPath(workspace_, "myimportedbazelrc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(imported_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile(
      "startup --max_idle_secs=123\nstartup --io_nice_level=4",
      imported_rc_path, 0755));

  // Add startup flags the imported bazelrc.
  const std::string master_rc_path =
      blaze_util::JoinPath(workspace_, "tools/bazel.rc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(master_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --max_idle_secs=42\nimport " +
                                        imported_rc_path +
                                        "\nstartup --io_nice_level=6",
                                    master_rc_path, 0755));

  const std::vector<std::string> args = {"bazel", "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;

  EXPECT_EQ(123, option_processor_->GetParsedStartupOptions()->max_idle_secs);
  EXPECT_EQ(6, option_processor_->GetParsedStartupOptions()->io_nice_level);

  // Check that the options are reported in the correct order in the provenance
  // message, the imported file between the two master flags
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_PRED1(
      [](std::string actualOutput) {
        return RE2::FullMatch(
            actualOutput,
            "INFO: Reading 'startup' options from .*tools.*bazel.rc: "
            "--max_idle_secs=42\n"
            "INFO: Reading 'startup' options from .*myimportedbazelrc: "
            "--max_idle_secs=123 --io_nice_level=4\n"
            "INFO: Reading 'startup' options from .*tools.*bazel.rc: "
            "--io_nice_level=6\n");
      },
      output);
}

TEST_F(OptionProcessorTest, SplitCommandLineWithEmptyArgs) {
  FailedSplitStartupOptionsTest(
      {},
      "Unable to split command line, args is empty");
}

TEST_F(OptionProcessorTest, SplitCommandLineWithAllParams) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--nomaster_bazelrc", "build", "--bar", ":mytarget"},
      CommandLine("bazel",
                  {"--nomaster_bazelrc"},
                  "build",
                  {"--bar", ":mytarget"}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithAbsolutePathToBinary) {
  SuccessfulSplitStartupOptionsTest(
      {"mybazel", "build", ":mytarget"},
      CommandLine("mybazel", {}, "build", {":mytarget"}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithUnaryStartupWithEquals) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--bazelrc=foo", "build", ":mytarget"},
      CommandLine("bazel", {"--bazelrc=foo"}, "build", {":mytarget"}));
}

TEST_F(OptionProcessorTest,
       SplitCommandLineWithUnaryStartupWithoutEquals) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--bazelrc", "foo", "build", ":mytarget"},
      CommandLine("bazel", {"--bazelrc=foo"}, "build", {":mytarget"}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithIncompleteUnaryOption) {
  FailedSplitStartupOptionsTest(
      {"bazel", "--bazelrc"},
      "Startup option '--bazelrc' expects a value.\n"
          "Usage: '--bazelrc=somevalue' or '--bazelrc somevalue'.\n"
          "  For more info, run 'bazel help startup_options'.");
}

TEST_F(OptionProcessorTest, SplitCommandLineWithMultipleStartup) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--bazelrc", "foo", "--nomaster_bazelrc", "build", ":mytarget"},
      CommandLine("bazel",
                  {"--bazelrc=foo", "--nomaster_bazelrc"},
                  "build",
                  {":mytarget"}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithNoStartupArgs) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "build", ":mytarget"},
      CommandLine("bazel", {}, "build", {":mytarget"}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithNoCommandArgs) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "build"},
      CommandLine("bazel", {}, "build", {}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithBlazeHelp) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "help"},
      CommandLine("bazel", {}, "help", {}));

  SuccessfulSplitStartupOptionsTest(
      {"bazel", "-h"},
      CommandLine("bazel", {}, "-h", {}));

  SuccessfulSplitStartupOptionsTest(
      {"bazel", "-help"},
      CommandLine("bazel", {}, "-help", {}));

  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--help"},
      CommandLine("bazel", {}, "--help", {}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithBlazeVersion) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "version"},
      CommandLine("bazel", {}, "version", {}));
}

TEST_F(OptionProcessorTest, SplitCommandLineWithMultipleCommandArgs) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "build", "--foo", "-s", ":mytarget"},
      CommandLine("bazel", {}, "build", {"--foo", "-s", ":mytarget"}));
}

TEST_F(OptionProcessorTest,
       SplitCommandLineFailsWithDashDashInStartupArgs) {
  FailedSplitStartupOptionsTest(
      {"bazel", "--"},
      "Unknown startup option: '--'.\n"
          "  For more info, run 'bazel help startup_options'.");
}

TEST_F(OptionProcessorTest, SplitCommandLineWithDashDash) {
  SuccessfulSplitStartupOptionsTest(
      {"bazel", "--nomaster_bazelrc", "build", "--b", "--", ":mytarget"},
      CommandLine("bazel",
                  {"--nomaster_bazelrc"},
                  "build",
                  {"--b", "--", ":mytarget"}));
}

TEST_F(OptionProcessorTest, TestDedupePathsOmitsInvalidPath) {
  std::vector<std::string> input = {"foo"};
  std::vector<std::string> expected = {};
  ASSERT_EQ(expected, internal::DedupeBlazercPaths(input));
}

TEST_F(OptionProcessorTest, TestDedupePathsWithDifferentFiles) {
  std::string foo_path = blaze_util::JoinPath(workspace_, "foo");
  std::string bar_path = blaze_util::JoinPath(workspace_, "bar");

  ASSERT_TRUE(blaze_util::WriteFile("foo", foo_path));
  ASSERT_TRUE(blaze_util::WriteFile("bar", bar_path));

  std::vector<std::string> input = {foo_path, bar_path};
  ASSERT_EQ(input, internal::DedupeBlazercPaths(input));
}

TEST_F(OptionProcessorTest, TestDedupePathsWithSameFile) {
  std::string foo_path = blaze_util::JoinPath(workspace_, "foo");

  ASSERT_TRUE(blaze_util::WriteFile("foo", foo_path));

  std::vector<std::string> input = {foo_path, foo_path};
  std::vector<std::string> expected = {foo_path};
  ASSERT_EQ(expected, internal::DedupeBlazercPaths(input));
}

TEST_F(OptionProcessorTest, TestDedupePathsWithRelativePath) {
  std::string dir(blaze_util::JoinPath(workspace_, "dir"));
  std::string foo_path(blaze_util::JoinPath(dir, "foo"));
  std::string relative_foo_path(blaze_util::JoinPath(dir, "../dir/foo"));

  ASSERT_TRUE(blaze_util::MakeDirectories(dir, 0755));
  ASSERT_TRUE(blaze_util::WriteFile("foo", foo_path));

  std::vector<std::string> input = {foo_path, relative_foo_path};
  std::vector<std::string> expected = {foo_path};
  ASSERT_EQ(expected, internal::DedupeBlazercPaths(input));
}

#if !defined(COMPILER_MSVC) && !defined(__CYGWIN__)
static bool Symlink(const std::string& old_path, const std::string& new_path) {
  return symlink(old_path.c_str(), new_path.c_str()) == 0;
}

TEST_F(OptionProcessorTest, TestDedupePathsWithSymbolicLink) {
  std::string foo_path = blaze_util::JoinPath(workspace_, "foo");
  std::string sym_foo_path = blaze_util::JoinPath(workspace_, "sym_foo");

  ASSERT_TRUE(blaze_util::WriteFile("foo", foo_path));
  ASSERT_TRUE(Symlink(foo_path, sym_foo_path));
  std::vector<std::string> input = {foo_path, sym_foo_path};
  std::vector<std::string> expected = {foo_path};
  ASSERT_EQ(expected, internal::DedupeBlazercPaths(input));
}
#endif  // !defined(COMPILER_MSVC) && !defined(__CYGWIN__)

}  // namespace blaze
