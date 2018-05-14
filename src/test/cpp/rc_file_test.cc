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

#include "src/main/cpp/bazel_startup_options.h"
#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor-internal.h"
#include "src/main/cpp/rc_file.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/workspace_layout.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
constexpr const char* kNullDevice = "nul";
#else  // Assume POSIX if not Windows.
constexpr const char* kNullDevice = "/dev/null";
#endif

class RcFileTest : public ::testing::Test {
 protected:
  RcFileTest()
      : workspace_(
            blaze_util::JoinPath(blaze::GetEnv("TEST_TMPDIR"), "testdir")),
        cwd_(blaze_util::JoinPath(blaze::GetEnv("TEST_TMPDIR"), "cwd")),
        binary_dir_(
            blaze_util::JoinPath(blaze::GetEnv("TEST_TMPDIR"), "bazeldir")),
        binary_path_(blaze_util::JoinPath(binary_dir_, "bazel")),
        workspace_layout_(new WorkspaceLayout()) {}

  void SetUp() override {
    ASSERT_TRUE(blaze_util::MakeDirectories(workspace_, 0755));
    ASSERT_TRUE(blaze_util::MakeDirectories(cwd_, 0755));
    ASSERT_TRUE(blaze_util::MakeDirectories(binary_dir_, 0755));
    option_processor_.reset(new OptionProcessor(
        workspace_layout_.get(),
        std::unique_ptr<StartupOptions>(
            new BazelStartupOptions(workspace_layout_.get()))));
  }

  void TearDown() override {
    // TODO(bazel-team): The code below deletes all the files in the workspace
    // and other rc-related directories, but it intentionally skips directories.
    // As a consequence, there may be empty directories from test to test.
    // Remove this once blaze_util::DeleteDirectories(path) exists.
    std::vector<std::string> files;
    blaze_util::GetAllFilesUnder(workspace_, &files);
    for (const std::string& file : files) {
      blaze_util::UnlinkPath(file);
    }
    blaze_util::GetAllFilesUnder(cwd_, &files);
    for (const std::string& file : files) {
      blaze_util::UnlinkPath(file);
    }
    blaze_util::GetAllFilesUnder(binary_dir_, &files);
    for (const std::string& file : files) {
      blaze_util::UnlinkPath(file);
    }
  }

  // We only test 2 of the 3 master bazelrc locations in this test. The third
  // masterrc that should be loaded is the system wide bazelrc path,
  // /etc/bazel.bazelrc, which we do not mock within this test because it is not
  // within the sandbox. It may or may not exist on the system running the test,
  // so we do not check for it.
  // TODO(#4502): Make the system-wide master bazelrc location configurable and
  // add test coverage for it.
  bool SetUpMasterRcFileInWorkspace(const std::string& contents,
                                    std::string* rcfile_path) const {
    const std::string tools_dir = blaze_util::JoinPath(workspace_, "tools");
    const std::string workspace_rc_path =
        blaze_util::JoinPath(tools_dir, "bazel.rc");
    if (blaze_util::MakeDirectories(tools_dir, 0755) &&
        blaze_util::WriteFile(contents, workspace_rc_path, 0755)) {
      *rcfile_path = workspace_rc_path;
      return true;
    }
    return false;
  }

  bool SetUpMasterRcFileAlongsideBinary(const std::string& contents,
                                        std::string* rcfile_path) const {
    const std::string binary_rc_path =
        blaze_util::JoinPath(binary_dir_, "bazel.bazelrc");
    if (blaze_util::WriteFile(contents, binary_rc_path, 0755)) {
      *rcfile_path = binary_rc_path;
      return true;
    }
    return false;
  }

  // This file is looked for if no --bazelrc is explicitly provided.
  bool SetUpUserRcFileInWorkspace(const std::string& contents,
                                  std::string* rcfile_path) const {
    const std::string workspace_user_rc_path =
        blaze_util::JoinPath(workspace_, ".bazelrc");
    if (blaze_util::WriteFile(contents, workspace_user_rc_path, 0755)) {
      *rcfile_path = workspace_user_rc_path;
      return true;
    }
    return false;
  }

  const std::string workspace_;
  const std::string cwd_;
  const std::string binary_dir_;
  const std::string binary_path_;
  const std::unique_ptr<WorkspaceLayout> workspace_layout_;
  std::unique_ptr<OptionProcessor> option_processor_;
};

using GetRcFileTest = RcFileTest;

TEST_F(GetRcFileTest, GetRcFilesLoadsAllMasterBazelrcs) {
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("", &workspace_rc));
  std::string binary_rc;
  ASSERT_TRUE(SetUpMasterRcFileAlongsideBinary("", &binary_rc));

  const CommandLine cmd_line =
      CommandLine(binary_path_, {"--bazelrc=/dev/null"}, "build", {});
  std::string error = "check that this string is not modified";
  std::vector<std::unique_ptr<RcFile>> parsed_rcs;
  const blaze_exit_code::ExitCode exit_code =
      option_processor_->GetRcFiles(workspace_layout_.get(), workspace_, cwd_,
                                    &cmd_line, &parsed_rcs, &error);
  EXPECT_EQ(blaze_exit_code::SUCCESS, exit_code);
  EXPECT_EQ("check that this string is not modified", error);

  // There should be 3-4 rc files, "/dev/null" does in some sense count as a
  // file, and there's an optional /etc/ file the test environment cannot
  // control. The first 2 rcs parsed should be the two rc files we expect, and
  // the last file is the user-provided /dev/null.
  ASSERT_LE(3, parsed_rcs.size());
  ASSERT_GE(4, parsed_rcs.size());
  const std::deque<std::string> expected_workspace_rc_que = {workspace_rc};
  const std::deque<std::string> expected_binary_rc_que = {binary_rc};
  const std::deque<std::string> expected_user_rc_que = {kNullDevice};
  EXPECT_EQ(expected_workspace_rc_que, parsed_rcs[0].get()->sources());
  EXPECT_EQ(expected_binary_rc_que, parsed_rcs[1].get()->sources());
  EXPECT_EQ(expected_user_rc_que,
            parsed_rcs[parsed_rcs.size() - 1].get()->sources());
}

TEST_F(GetRcFileTest, GetRcFilesRespectsNoMasterBazelrc) {
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("", &workspace_rc));
  std::string binary_rc;
  ASSERT_TRUE(SetUpMasterRcFileAlongsideBinary("", &binary_rc));

  const CommandLine cmd_line = CommandLine(
      binary_path_, {"--nomaster_bazelrc", "--bazelrc=/dev/null"}, "build", {});
  std::string error = "check that this string is not modified";
  std::vector<std::unique_ptr<RcFile>> parsed_rcs;
  const blaze_exit_code::ExitCode exit_code =
      option_processor_->GetRcFiles(workspace_layout_.get(), workspace_, cwd_,
                                    &cmd_line, &parsed_rcs, &error);
  EXPECT_EQ(blaze_exit_code::SUCCESS, exit_code);
  EXPECT_EQ("check that this string is not modified", error);

  // /dev/null is technically a file, but no master rcs should have been loaded.
  const std::deque<std::string> expected_user_rc_que = {kNullDevice};
  EXPECT_EQ(expected_user_rc_que, parsed_rcs[0].get()->sources());
}

TEST_F(GetRcFileTest, GetRcFilesReadsCommandLineUserRc) {
  const std::string cmdline_rc_path =
      blaze_util::JoinPath(workspace_, "mybazelrc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(cmdline_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("", cmdline_rc_path, 0755));

  const CommandLine cmd_line = CommandLine(
      binary_path_, {"--nomaster_bazelrc", "--bazelrc=" + cmdline_rc_path},
      "build", {});
  std::string error = "check that this string is not modified";
  std::vector<std::unique_ptr<RcFile>> parsed_rcs;
  const blaze_exit_code::ExitCode exit_code =
      option_processor_->GetRcFiles(workspace_layout_.get(), workspace_, cwd_,
                                    &cmd_line, &parsed_rcs, &error);
  EXPECT_EQ(blaze_exit_code::SUCCESS, exit_code);
  EXPECT_EQ("check that this string is not modified", error);

  // Because of the variety of path representations in windows, this
  // equality test does not attempt to check the entire path.
  ASSERT_EQ(1, parsed_rcs.size());
  ASSERT_EQ(1, parsed_rcs[0].get()->sources().size());
  EXPECT_THAT(parsed_rcs[0].get()->sources().front(), HasSubstr("mybazelrc"));
}

TEST_F(GetRcFileTest, GetRcFilesReadsUserRcInWorkspace) {
  // We expect the user rc to be read when from the workspace if no alternative
  // --bazelrc is provided.
  std::string user_workspace_rc;
  ASSERT_TRUE(SetUpUserRcFileInWorkspace("", &user_workspace_rc));

  const CommandLine cmd_line =
      CommandLine(binary_path_, {"--nomaster_bazelrc"}, "build", {});
  std::string error = "check that this string is not modified";
  std::vector<std::unique_ptr<RcFile>> parsed_rcs;
  const blaze_exit_code::ExitCode exit_code =
      option_processor_->GetRcFiles(workspace_layout_.get(), workspace_, cwd_,
                                    &cmd_line, &parsed_rcs, &error);
  EXPECT_EQ(blaze_exit_code::SUCCESS, exit_code);
  EXPECT_EQ("check that this string is not modified", error);

  const std::deque<std::string> expected_user_rc_que = {user_workspace_rc};
  ASSERT_EQ(1, parsed_rcs.size());
  EXPECT_EQ(expected_user_rc_que, parsed_rcs[0].get()->sources());
}

using ParseOptionsTest = RcFileTest;

TEST_F(ParseOptionsTest, CommandLineBazelrcHasUnknownOption) {
  const std::string cmdline_rc_path =
      blaze_util::JoinPath(workspace_, "mybazelrc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(cmdline_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile("startup --foo", cmdline_rc_path, 0755));

  const std::vector<std::string> args = {binary_path_, "--nomaster_bazelrc",
                                         "--bazelrc=" + cmdline_rc_path,
                                         "build"};
  const std::string expected_error =
      "Unknown startup option: '--foo'.\n"
      "  For more info, run 'bazel help startup_options'.";
  std::string error;
  ASSERT_NE(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;
  ASSERT_EQ(expected_error, error);

  // Check that the startup options' provenance message contains the correct
  // information for the incorrect flag, and does not print the command-line
  // provided startup flags.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output,
              MatchesRegex(
                  "INFO: Reading 'startup' options from .*mybazelrc: --foo\n"));
}

TEST_F(ParseOptionsTest, MasterBazelrcHasUnknownOption) {
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("startup --foo", &workspace_rc));

  const std::vector<std::string> args = {binary_path_, "build"};

  // Expect no error due to the incorrect --foo.
  std::string error;
  EXPECT_EQ(blaze_exit_code::BAD_ARGV,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error));
  ASSERT_EQ(
      "Unknown startup option: '--foo'.\n"
      "  For more info, run 'bazel help startup_options'.",
      error);

  // Check that the startup options' provenance message contains nothing for the
  // master bazelrc.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(
      output,
      MatchesRegex(
          "INFO: Reading 'startup' options from .*tools.bazel.rc: --foo\n"));
}

TEST_F(ParseOptionsTest,
       IncorrectMasterBazelrcIgnoredWhenNoMasterBazelrcIsPresent) {
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("startup --foo", &workspace_rc));

  const std::vector<std::string> args = {binary_path_, "--nomaster_bazelrc",
                                         "build"};

  // Expect no error due to the incorrect --foo.
  std::string error;
  EXPECT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error));
  ASSERT_EQ("", error);

  // Check that the startup options' provenance message contains nothing for the
  // master bazelrc.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(output, "");
}

TEST_F(ParseOptionsTest, UserBazelrcHasPriorityOverMasterBazelrc) {
  std::string user_rc;
  ASSERT_TRUE(
      SetUpUserRcFileInWorkspace("startup --max_idle_secs=123", &user_rc));
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("startup --max_idle_secs=42",
                                           &workspace_rc));

  const std::vector<std::string> args = {binary_path_, "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;

  EXPECT_EQ(123, option_processor_->GetParsedStartupOptions()->max_idle_secs);

  // Check that the startup options' provenance message contains the correct
  // information for the provided rc, and prints nothing for the master bazelrc.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  const std::string expected_message = "INFO: Reading 'startup' options from " +
                                       workspace_rc +
                                       ": --max_idle_secs=42\n"
                                       "INFO: Reading 'startup' options from " +
                                       user_rc + ": --max_idle_secs=123\n";
  EXPECT_EQ(output, expected_message);
}

TEST_F(ParseOptionsTest, MasterBazelrcOverridesNoMasterBazelrc) {
  std::string workspace_rc;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace("startup --max_idle_secs=123",
                                           &workspace_rc));

  const std::vector<std::string> args = {"bazel", "--nomaster_bazelrc",
                                         "--master_bazelrc", "build"};
  std::string error;
  ASSERT_EQ(blaze_exit_code::SUCCESS,
            option_processor_->ParseOptions(args, workspace_, cwd_, &error))
      << error;
  EXPECT_EQ(123, option_processor_->GetParsedStartupOptions()->max_idle_secs);

  // Check that the startup options' provenance message contains the correct
  // information for the master bazelrc.
  testing::internal::CaptureStderr();
  option_processor_->PrintStartupOptionsProvenanceMessage();
  const std::string& output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, MatchesRegex("INFO: Reading 'startup' options from "
                                   ".*tools.bazel.rc: --max_idle_secs=123\n"));
}

TEST_F(ParseOptionsTest, MultipleStartupArgsInMasterBazelrcWorksCorrectly) {
  // Add startup flags to the master bazelrc.
  std::string master_rc_path;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace(
      "startup --max_idle_secs=42\nstartup --io_nice_level=6",
      &master_rc_path));

  const std::vector<std::string> args = {binary_path_, "build"};
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

  EXPECT_THAT(
      output,
      MatchesRegex("INFO: Reading 'startup' options from .*tools.bazel.rc: "
                   "--max_idle_secs=42 --io_nice_level=6\n"));
}

TEST_F(ParseOptionsTest, CustomBazelrcOverridesMasterBazelrc) {
  // Add startup flags to the master bazelrc.
  std::string master_rc_path;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace(
      "startup --max_idle_secs=42\nstartup --io_nice_level=6",
      &master_rc_path));

  // Override one of the master bazelrc's flags in the commandline rc.
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

  EXPECT_THAT(
      output,
      MatchesRegex("INFO: Reading 'startup' options from .*tools.*bazel.rc: "
                   "--max_idle_secs=42 --io_nice_level=6\n"
                   "INFO: Reading 'startup' options from .*mybazelrc: "
                   "--max_idle_secs=123\n"));
}

TEST_F(ParseOptionsTest, BazelRcImportsMaintainsFlagOrdering) {
  // Override one of the master bazelrc's flags in the custom bazelrc.
  const std::string imported_rc_path =
      blaze_util::JoinPath(workspace_, "myimportedbazelrc");
  ASSERT_TRUE(
      blaze_util::MakeDirectories(blaze_util::Dirname(imported_rc_path), 0755));
  ASSERT_TRUE(blaze_util::WriteFile(
      "startup --max_idle_secs=123\nstartup --io_nice_level=4",
      imported_rc_path, 0755));

  // Add startup flags the imported bazelrc.
  std::string master_rc_path;
  ASSERT_TRUE(SetUpMasterRcFileInWorkspace(
      "startup --max_idle_secs=42\nimport " + imported_rc_path +
          "\nstartup --io_nice_level=6",
      &master_rc_path));

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

  EXPECT_THAT(
      output,
      MatchesRegex("INFO: Reading 'startup' options from .*tools.*bazel.rc: "
                   "--max_idle_secs=42\n"
                   "INFO: Reading 'startup' options from .*myimportedbazelrc: "
                   "--max_idle_secs=123 --io_nice_level=4\n"
                   "INFO: Reading 'startup' options from .*tools.*bazel.rc: "
                   "--io_nice_level=6\n"));
}

}  // namespace blaze
