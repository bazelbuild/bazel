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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/workspace_layout.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "src/main/cpp/rc_file.h"

namespace blaze {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::MatchesRegex;
using ::testing::MockFunction;
using ::testing::Pair;
using ::testing::Property;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::UnorderedElementsAre;

#if _WIN32
constexpr bool kIsWindows = true;
#else
constexpr bool kIsWindows = false;
#endif
const SemVer kTestSemVer = SemVer({"8", "4"});

class RcOptionsTest : public ::testing::Test {
 protected:
  RcOptionsTest()
      : test_file_dir_(blaze_util::JoinPath(
            blaze::GetPathEnv("TEST_TMPDIR"),
            ::testing::UnitTest::GetInstance()->current_test_info()->name())),
        workspace_layout_() {
  }

  const std::string test_file_dir_;
  const WorkspaceLayout workspace_layout_;

  void WriteRc(const std::string& filename, const std::string& contents) {
    bool success_mkdir = blaze_util::MakeDirectories(test_file_dir_, 0755);
    ASSERT_TRUE(success_mkdir) << "Failed to mkdir -p " << test_file_dir_;
    bool success_write = blaze_util::WriteFile(
        contents, blaze_util::JoinPath(test_file_dir_, filename));
    ASSERT_TRUE(success_write) << "Failed to write " << filename;
  }

  std::unique_ptr<RcFile> Parse(const std::string& filename,
                                RcFile::ParseError* error,
                                std::string* error_text) {
    return RcFile::Parse(
        blaze_util::JoinPath(test_file_dir_, filename),
        &workspace_layout_,
        // Set workspace to test_file_dir_ so importing %workspace%/foo works.
        test_file_dir_,
        error,
        error_text,
        kTestSemVer);
  }

  void SuccessfullyParseRcWithExpectedArgs(
      const std::string& filename,
      const absl::flat_hash_map<std::string, std::vector<std::string>>&
          expected_args_map) {
    RcFile::ParseError error;
    std::string error_text;
    std::unique_ptr<RcFile> rc = Parse(filename, &error, &error_text);
    EXPECT_EQ(error_text, "");
    ASSERT_EQ(error, RcFile::ParseError::NONE);

    // Test that exactly each command in the expected map was in the results,
    // and that for each of these, exactly the expected args are found, in the
    // correct order. Note that this is not just an exercise in rewriting map
    // equality - the results have type RcOption, and the expected values
    // are just strings. This is ignoring the source_path for convenience.
    RcFile::OptionMap result = rc->options();
    ASSERT_EQ(expected_args_map.size(), result.size());
    for (const auto& command_args_pair : expected_args_map) {
      const std::string& expected_command = command_args_pair.first;
      const std::vector<std::string>& expected_args = command_args_pair.second;
      const auto result_args_iter = result.find(expected_command);
      ASSERT_NE(result_args_iter, rc->options().end());
      const std::vector<RcOption>& result_args = result_args_iter->second;
      ASSERT_EQ(result_args.size(), expected_args.size());
      for (size_t i = 0; i < result_args.size(); ++i) {
        EXPECT_EQ(result_args[i].option, expected_args[i]);
      }
    }
  }
};

TEST_F(RcOptionsTest, Empty) {
  WriteRc("empty.bazelrc",
          "");
  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("empty.bazelrc", no_expected_args);
}

TEST_F(RcOptionsTest, Whitespace) {
  WriteRc("whitespace.bazelrc",
          "      \n\t      ");
  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("whitespace.bazelrc", no_expected_args);
}

TEST_F(RcOptionsTest, CommentedStartup) {
  WriteRc("commented_startup.bazelrc",
          "# startup foo");
  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("commented_startup.bazelrc",
                                      no_expected_args);
}

TEST_F(RcOptionsTest, EmptyStartupLine) {
  WriteRc("empty_startup_line.bazelrc",
          "startup");
  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("empty_startup_line.bazelrc",
                                      no_expected_args);
}

TEST_F(RcOptionsTest, StartupWithOnlyCommentedArg) {
  WriteRc("startup_with_comment.bazelrc",
          "startup # bar");
  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("startup_with_comment.bazelrc",
                                      no_expected_args);
}

TEST_F(RcOptionsTest, SingleStartupArg) {
  WriteRc("startup_foo.bazelrc",
          "startup foo");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_foo.bazelrc",
      {{"startup", {"foo"}}});
}

TEST_F(RcOptionsTest, SingleStartupArgWithComment) {
  WriteRc("startup_foo_and_comment.bazelrc",
          "startup foo # comment");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_foo_and_comment.bazelrc",
      {{"startup", {"foo"}}});
}

TEST_F(RcOptionsTest, TwoStartupArgsOnOneLine) {
  WriteRc("startup_foo_bar.bazelrc",
          "startup foo bar");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_foo_bar.bazelrc",
      {{"startup", {"foo", "bar"}}});
}

TEST_F(RcOptionsTest, TwoStartupArgsOnOneLineTabSeparated) {
  WriteRc("startup_with_tabs.bazelrc",
          "startup\tfoo\tbar");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_with_tabs.bazelrc",
      {{"startup", {"foo", "bar"}}});
}

TEST_F(RcOptionsTest, StartupOptWithSimpleValue) {
  WriteRc("startup_opt_with_simple_value.bazelrc",
          "startup --opt=foo");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_opt_with_simple_value.bazelrc",
      {{"startup", {"--opt=foo"}}});
}

TEST_F(RcOptionsTest, StartupQuotedArg) {
  WriteRc("startup_quoted_foo_bar.bazelrc",
          "startup \"foo bar\"");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_quoted_foo_bar.bazelrc",
      {{"startup", {"foo bar"}}});
}

TEST_F(RcOptionsTest, QuotedValueStartupArgAfterEquals) {
  WriteRc("startup_opt_quoted_arg.bazelrc",
          "startup --opt=\"foo bar\"");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_opt_quoted_arg.bazelrc",
      {{"startup", {"--opt=foo bar"}}});
}

TEST_F(RcOptionsTest, QuotedValueStartupArgAfterWhitespace) {
  WriteRc("startup_opt_quoted_arg_as_separate_token.bazelrc",
          "startup --opt \"foo bar\"");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_opt_quoted_arg_as_separate_token.bazelrc",
      {{"startup", {"--opt", "foo bar"}}});
}

TEST_F(RcOptionsTest, QuotedValueStartupArgOnNewLine) {
  WriteRc("startup_opt_quoted_arg_different_line.bazelrc",
          "startup --opt\n"
          "startup \"foo bar\"");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_opt_quoted_arg_different_line.bazelrc",
      {{"startup", {"--opt", "foo bar"}}});
}

TEST_F(RcOptionsTest, TwoOptStartup) {
  WriteRc("startup_two_options.bazelrc",
          "startup --opt1\n"
          "startup --opt2");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_two_options.bazelrc",
      {{"startup", {"--opt1", "--opt2"}}});
}

TEST_F(RcOptionsTest, WhitespaceBeforeStartup) {
  WriteRc("whitespace_before_command.bazelrc",
          "  startup foo\n"
          "        # indented comments\n"
          "startup bar\n"
          "\tstartup     \t baz");
  SuccessfullyParseRcWithExpectedArgs(
      "whitespace_before_command.bazelrc",
      {{"startup", {"foo", "bar", "baz"}}});
}

TEST_F(RcOptionsTest, StartupLineContinuation) {
  WriteRc("startup_line_continuation.bazelrc",
          "startup \\\n"
          "foo\n"
          "startup bar \\\n"
          "baz");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_line_continuation.bazelrc",
      {{"startup", {"foo", "bar", "baz"}}});
}

TEST_F(RcOptionsTest, ManyArgStartup) {
  WriteRc("startup_with_many_args.bazelrc",
          "# Many arguments\n"
          "startup foo # First argument has reasons.\n"
          "startup --opt1   --opt2 # These arguments are split wide\n"
          "#startup --this_is_not_an_arg\n"
          "\n\n\n # A few empty lines for good measure\n"
          "startup\t \"string input Value 123. \" --bar");
  SuccessfullyParseRcWithExpectedArgs(
      "startup_with_many_args.bazelrc",
      {{
          "startup",
          {"foo", "--opt1", "--opt2", "string input Value 123. ", "--bar"}
      }});
}

TEST_F(RcOptionsTest, MultipleCommands) {
  WriteRc("multiple_commands_intermixed.bazelrc",
          "startup foo\n"
          "build aaa\n"
          "startup bar baz\n"
          "build bbb\n"
          "build ccc\n");
  SuccessfullyParseRcWithExpectedArgs(
      "multiple_commands_intermixed.bazelrc",
      {{"startup", {"foo", "bar", "baz"}}, {"build", {"aaa", "bbb", "ccc"}}});
}

TEST_F(RcOptionsTest, SimpleImportFoo) {
  WriteRc("startup_foo.bazelrc",
          "startup foo");
  WriteRc("import_simple.bazelrc",
          "import %workspace%/startup_foo.bazelrc");
  SuccessfullyParseRcWithExpectedArgs(
      "import_simple.bazelrc",
      {{"startup", {"foo"}}});
}

TEST_F(RcOptionsTest, ImportFooThenAddBar) {
  WriteRc("startup_foo.bazelrc",
          "startup foo");
  WriteRc("import_foo_then_bar.bazelrc",
          "import %workspace%/startup_foo.bazelrc\n"
          "startup bar");
  SuccessfullyParseRcWithExpectedArgs(
      "import_foo_then_bar.bazelrc",
      {{"startup", {"foo", "bar"}}});
}

TEST_F(RcOptionsTest, StartupBarThenImportFoo) {
  WriteRc("startup_foo.bazelrc",
          "startup foo");
  WriteRc("bar_then_import_foo.bazelrc",
          "startup bar\n"
          "import %workspace%/startup_foo.bazelrc");
  SuccessfullyParseRcWithExpectedArgs(
      "bar_then_import_foo.bazelrc",
      {{"startup", {"bar", "foo"}}});
}

TEST_F(RcOptionsTest, SimpleTryImportFoo) {
  WriteRc("startup_foo.bazelrc", "startup foo");
  WriteRc("import_simple.bazelrc",
          "try-import %workspace%/startup_foo.bazelrc");
  SuccessfullyParseRcWithExpectedArgs("import_simple.bazelrc",
                                      {{"startup", {"foo"}}});
}

TEST_F(RcOptionsTest, ImportTryFooThenAddBar) {
  WriteRc("startup_foo.bazelrc", "startup foo");
  WriteRc("import_foo_then_bar.bazelrc",
          "try-import %workspace%/startup_foo.bazelrc\n"
          "startup bar");
  SuccessfullyParseRcWithExpectedArgs("import_foo_then_bar.bazelrc",
                                      {{"startup", {"foo", "bar"}}});
}

TEST_F(RcOptionsTest, StartupBarThenTryImportFoo) {
  WriteRc("startup_foo.bazelrc", "startup foo");
  WriteRc("bar_then_import_foo.bazelrc",
          "startup bar\n"
          "try-import %workspace%/startup_foo.bazelrc");
  SuccessfullyParseRcWithExpectedArgs("bar_then_import_foo.bazelrc",
                                      {{"startup", {"bar", "foo"}}});
}

// Most likely, import diamonds like this are unintended, and they might lead
// to surprising doubled values for allow_multiple options. This causes a
// warning in option_processor, which checks for duplicates across multiple rc
// files.
TEST_F(RcOptionsTest, ImportDiamond) {
  WriteRc("startup_foo.bazelrc",
          "startup foo");
  WriteRc("import_foo_then_bar.bazelrc",
          "import %workspace%/startup_foo.bazelrc\n"
          "startup bar");
  WriteRc("bar_then_import_foo.bazelrc",
          "startup bar\n"
          "import %workspace%/startup_foo.bazelrc");
  WriteRc("import_diamond.bazelrc",
          "import %workspace%/import_foo_then_bar.bazelrc\n"
          "import %workspace%/bar_then_import_foo.bazelrc");
  SuccessfullyParseRcWithExpectedArgs(
      "import_diamond.bazelrc",
      {{"startup", {"foo", "bar", "bar", "foo"}}});
}


TEST_F(RcOptionsTest, ImportCycleFails) {
  WriteRc("import_cycle_1.bazelrc",
          "import %workspace%/import_cycle_2.bazelrc");
  WriteRc("import_cycle_2.bazelrc",
          "import %workspace%/import_cycle_1.bazelrc");

  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc =
      Parse("import_cycle_1.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::IMPORT_LOOP);
  ASSERT_THAT(
      error_text,
      MatchesRegex("Import loop detected:\n"
                   "  .*import_cycle_1.bazelrc\n"
                   "  .*import_cycle_2.bazelrc\n"
                   "  .*import_cycle_1.bazelrc\n"));
}

TEST_F(RcOptionsTest, LongImportCycleFails) {
  WriteRc("chain_to_cycle_1.bazelrc",
          "import %workspace%/chain_to_cycle_2.bazelrc");
  WriteRc("chain_to_cycle_2.bazelrc",
          "import %workspace%/chain_to_cycle_3.bazelrc");
  WriteRc("chain_to_cycle_3.bazelrc",
          "import %workspace%/chain_to_cycle_4.bazelrc");
  WriteRc("chain_to_cycle_4.bazelrc",
          "import %workspace%/import_cycle_1.bazelrc");
  WriteRc("import_cycle_1.bazelrc",
          "import %workspace%/import_cycle_2.bazelrc");
  WriteRc("import_cycle_2.bazelrc",
          "import %workspace%/import_cycle_1.bazelrc");

  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc =
      Parse("chain_to_cycle_1.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::IMPORT_LOOP);
  ASSERT_THAT(
      error_text,
      MatchesRegex("Import loop detected:\n"
                   "  .*chain_to_cycle_1.bazelrc\n"
                   "  .*chain_to_cycle_2.bazelrc\n"
                   "  .*chain_to_cycle_3.bazelrc\n"
                   "  .*chain_to_cycle_4.bazelrc\n"
                   "  .*import_cycle_1.bazelrc\n"
                   "  .*import_cycle_2.bazelrc\n"
                   "  .*import_cycle_1.bazelrc\n"));
}

TEST_F(RcOptionsTest, FileDoesNotExist) {
  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc = Parse("not_a_file.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::UNREADABLE_FILE);
  ASSERT_THAT(
      error_text,
      MatchesRegex(kIsWindows ? "Unexpected error reading config file "
                                "'.*not_a_file\\.bazelrc':.*"
                              : "Unexpected error reading config file "
                                "'.*not_a_file\\.bazelrc': "
                                "\\(error: 2\\): No such file or directory"));
}

TEST_F(RcOptionsTest, ImportedFileDoesNotExist) {
  WriteRc("import_fake_file.bazelrc",
          "import somefile");

  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc =
      Parse("import_fake_file.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::UNREADABLE_FILE);
  if (kIsWindows) {
    ASSERT_THAT(
        error_text,
        MatchesRegex("Unexpected error reading config file 'somefile':.*"));
  } else {
    ASSERT_EQ(
        error_text,
        "Unexpected error reading config file 'somefile': (error: 2): No such "
        "file or directory");
  }
}

TEST_F(RcOptionsTest, TryImportedFileDoesNotExist) {
  WriteRc("try_import_fake_file.bazelrc", "try-import somefile");

  absl::flat_hash_map<std::string, std::vector<std::string>> no_expected_args;
  SuccessfullyParseRcWithExpectedArgs("try_import_fake_file.bazelrc",
                                      no_expected_args);
}

TEST_F(RcOptionsTest, ImportHasTooManyArgs) {
  WriteRc("bad_import.bazelrc",
          "import somefile bar");

  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc = Parse("bad_import.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::INVALID_FORMAT);
  ASSERT_THAT(error_text,
              MatchesRegex("Invalid import declaration in config file "
                           "'.*bad_import.bazelrc': 'import somefile bar'"));
}

TEST_F(RcOptionsTest, TryImportHasTooManyArgs) {
  WriteRc("bad_import.bazelrc", "try-import somefile bar");

  RcFile::ParseError error;
  std::string error_text;
  std::unique_ptr<RcFile> rc = Parse("bad_import.bazelrc", &error, &error_text);
  EXPECT_EQ(error, RcFile::ParseError::INVALID_FORMAT);
  ASSERT_THAT(
      error_text,
      MatchesRegex("Invalid import declaration in config file "
                   "'.*bad_import.bazelrc': 'try-import somefile bar'"));
}

// TODO(b/34811299) The tests below identify ways that '\' used as a line
// continuation is broken. This is on top of user-reported cases where an
// unintentional '\' made the command on the following line show up as
// an argument, which lead to cryptic messages. There is no value added by '\',
// since the following line could just repeat the command, so it might be best
// to remove this feature entirely.
//
// For now, these tests serve as documentation of the brokenness, and to keep
// broken behavior consistent before we get around to fixing it.

TEST_F(RcOptionsTest, BadStartupLineContinuation_HasWhitespaceAfterSlash) {
  WriteRc("bad_startup_line_continuation.bazelrc",
          "startup foo \\ \n"
          "bar");
  SuccessfullyParseRcWithExpectedArgs(
      "bad_startup_line_continuation.bazelrc",
      {{"startup", {"foo"}}});  // Does not contain "bar" from the next line.
}

TEST_F(RcOptionsTest, BadStartupLineContinuation_HasErroneousSlash) {
  WriteRc("bad_startup_line_continuation.bazelrc",
          "startup foo \\ bar");
  SuccessfullyParseRcWithExpectedArgs(
      "bad_startup_line_continuation.bazelrc",
      // Whitespace between the slash and bar gets counted as part of the token.
      {{"startup", {"foo", " bar"}}});
}

TEST_F(RcOptionsTest, BadStartupLineContinuation_HasCommentAfterSlash) {
  WriteRc("bad_startup_line_continuation.bazelrc",
          "startup foo \\ # comment\n"
          "bar");
  SuccessfullyParseRcWithExpectedArgs(
      "bad_startup_line_continuation.bazelrc",
      // Whitespace between the slash and comment gets counted as a new token,
      // and the bar on the next line is ignored (it's an argumentless command).
      {{"startup", {"foo", " "}}});
}

TEST_F(RcOptionsTest, BadStartupLineContinuation_InterpretsNextLineAsNewline) {
  WriteRc("bad_startup_line_continuation.bazelrc",
          "startup foo \\ #comment\n"
          "bar baz");
  SuccessfullyParseRcWithExpectedArgs(
      "bad_startup_line_continuation.bazelrc",
      // Whitespace between the slash and comment gets counted as a new token,
      // and the bar on the next line treated as its own command, instead of as
      // a "startup" args.
      {{"startup", {"foo", " "}}, {"bar", {"baz"}}});
}

TEST(RemoteFileTest, ParsingRemoteFiles) {
  WorkspaceLayout workspace_layout;
  RcFile::ParseError error;
  std::string error_text;

  MockFunction<bool(const std::string&, std::string*, std::string*)> read_file;
  EXPECT_CALL(read_file, Call("the base file", _, _))
      .WillOnce(DoAll(SetArgPointee<1>("import second.bazelrc"), Return(true)));
  EXPECT_CALL(read_file, Call("second.bazelrc", _, _))
      .WillOnce(
          DoAll(SetArgPointee<1>("startup --max_idle_secs=123"), Return(true)));

  MockFunction<std::string(const std::string&)> canonicalize_path;
  EXPECT_CALL(canonicalize_path, Call("the base file"))
      .WillOnce(Return("the base file"));
  EXPECT_CALL(canonicalize_path, Call("second.bazelrc"))
      .WillOnce(Return("second.bazelrc"));

  std::unique_ptr<RcFile> rcfile = RcFile::Parse(
      "the base file", &workspace_layout, "my workspace", &error, &error_text,
      kTestSemVer, read_file.AsStdFunction(), canonicalize_path.AsStdFunction());
  EXPECT_THAT(error_text, IsEmpty());
  ASSERT_EQ(error, RcFile::ParseError::NONE);
  EXPECT_THAT(rcfile, Pointee(Property(
                          &RcFile::canonical_source_paths,
                          ElementsAre("the base file", "second.bazelrc"))));
  EXPECT_THAT(rcfile,
              Pointee(Property(
                  &RcFile::options,
                  UnorderedElementsAre(Pair(
                      "startup", ElementsAre(Field(&RcOption::option,
                                                   "--max_idle_secs=123")))))));
}

}  // namespace
}  // namespace blaze
