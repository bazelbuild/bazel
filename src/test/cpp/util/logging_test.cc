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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/bazel_log_handler.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze_util {
using ::testing::HasSubstr;
using ::testing::Not;
// Note: gmock uses different regex syntax on different platforms. MatchesRegex
// is still useful since the '.' wildcard can help match formatted log lines
// like `[bazel INFO filename:134] message`
// but should not be used for more fine grained testing.
using ::testing::MatchesRegex;
using ::testing::ContainsRegex;

class LoggingTest : public ::testing::Test {
 protected:
  void SetUp() {
    // Set the value of $TMP first, because CaptureStderr retrieves a temp
    // directory path and on Windows, the corresponding function (GetTempPathA)
    // reads $TMP.
    blaze::SetEnv("TMP", blaze::GetPathEnv("TEST_TMPDIR"));
  }
  void TearDown() { blaze_util::SetLogHandler(nullptr); }
};

TEST(LoggingTest, LogLevelNamesMatch) {
  EXPECT_STREQ("INFO", LogLevelName(LOGLEVEL_INFO));
  EXPECT_STREQ("USER", LogLevelName(LOGLEVEL_USER));
  EXPECT_STREQ("WARNING", LogLevelName(LOGLEVEL_WARNING));
  EXPECT_STREQ("ERROR", LogLevelName(LOGLEVEL_ERROR));
  EXPECT_STREQ("FATAL", LogLevelName(LOGLEVEL_FATAL));
}

// Tests for when no log handler is set.

TEST(LoggingTest, NoHandler_InfoLogsIgnored) {
  testing::internal::CaptureStderr();
  blaze_util::SetLogHandler(nullptr);

  // Log something.
  std::string teststring = "test that the info log messages get ignored";
  BAZEL_LOG(INFO) << teststring;

  // Check that stderr does not receive the message.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, NoHandler_UserLogsPrinted) {
  testing::internal::CaptureStderr();
  blaze_util::SetLogHandler(nullptr);

  // Log something.
  std::string teststring = "test that the user log messages are not ignored";
  BAZEL_LOG(USER) << teststring;

  // Check that stderr receives the message.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(teststring));
}

TEST(LoggingTest, NoHandler_WarningsPrinted) {
  testing::internal::CaptureStderr();
  blaze_util::SetLogHandler(nullptr);

  // Log something.
  BAZEL_LOG(WARNING) << "test that warnings are printed";
  std::string expectedString = "WARNING: test that warnings are printed";

  // Check that stderr receives the message.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(expectedString));
}

TEST(LoggingTest, NoHandler_ErrorsPrinted) {
  testing::internal::CaptureStderr();
  blaze_util::SetLogHandler(nullptr);

  // Log something.
  BAZEL_LOG(ERROR) << "test that errors are printed";
  std::string expectedError = "ERROR: test that errors are printed";

  // Check that stderr receives the message.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(expectedError));
}

// Tests for the BazelLogHandler, with no call to SetLoggingOutputStream.

TEST(LoggingTest, BazelLogHandler_DumpsToCerrAtDestruction) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something.
  std::string teststring = "test that log messages get dumped to stderr";
  BAZEL_LOG(INFO) << teststring;

  // Check that stderr isn't getting anything yet.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
  testing::internal::CaptureStderr();

  // Destruct the log handler and get the stderr remains.
  blaze_util::SetLogHandler(nullptr);
  stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(teststring));
}

// Tests for the BazelLogHandler's buffer after SetLoggingOutputStream(nullptr).

TEST(LoggingTest, BazelLogHandler_DoesNotDumpToStderrIfOuputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something.
  std::string teststring = "test that this log message is lost.";
  BAZEL_LOG(INFO) << teststring;
  blaze_util::SetLoggingOutputStream(nullptr);

  // Destruct the log handler and check if stderr got anything.
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_DoesNotPrintInfoLogsIfOuputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));
  blaze_util::SetLoggingOutputStream(nullptr);

  std::string teststring = "test that the log message is lost.";
  BAZEL_LOG(INFO) << teststring;

  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_PrintsUserLogsEvenIfOuputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));
  blaze_util::SetLoggingOutputStream(nullptr);

  std::string teststring = "some user message";
  BAZEL_LOG(USER) << teststring;

  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(teststring));
}

TEST(LoggingTest, BazelLogHandler_PrintsWarningsEvenIfOuputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));
  blaze_util::SetLoggingOutputStream(nullptr);

  BAZEL_LOG(WARNING) << "this is a warning";
  std::string expectedWarning = "WARNING: this is a warning";

  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(expectedWarning));
}

TEST(LoggingTest, BazelLogHandler_PrintsErrorsEvenIfOuputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));
  blaze_util::SetLoggingOutputStream(nullptr);

  BAZEL_LOG(ERROR) << "this is an error, alert!";
  std::string expectedError = "ERROR: this is an error, alert!";

  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(expectedError));
}

TEST(LoggingTest,
     BazelLogHandler_BufferedInfoLogsGetLostEvenIfOutputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something before telling the loghandler where to send it.
  std::string teststring = "this message should be lost.";
  BAZEL_LOG(INFO) << teststring;

  // Ask that the debug logs not be kept.
  blaze_util::SetLoggingOutputStream(nullptr);

  // Set a null log handler, which causes the BazelLogHandler to be destructed.
  // This prompts its logs to be flushed, so we can capture them.
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest,
     BazelLogHandler_BufferedWarningLogsRedirectedAfterOutputStreamSetToNull) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something before telling the loghandler where to send it.
  std::string teststring = "test that this message gets directed to cerr";
  BAZEL_LOG(WARNING) << teststring;
  std::string expectedWarning =
      "WARNING: test that this message gets directed to cerr";

  // Ask that the debug logs not be kept.
  blaze_util::SetLoggingOutputStream(nullptr);

  // Set a null log handler, which causes the BazelLogHandler to be destructed.
  // This prompts its logs to be flushed, so we can capture them.
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(expectedWarning));
}

// Tests for the BazelLogHandler & SetLoggingOutputStream

TEST(LoggingTest, BazelLogHandler_DirectingLogsToBufferStreamWorks) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Ask that the logs get output to a string buffer (keep a ptr to it so we can
  // check its contents)
  std::unique_ptr<std::stringstream> stringbuf(new std::stringstream());
  std::stringstream* stringbuf_ptr = stringbuf.get();
  blaze_util::SetLoggingOutputStream(std::move(stringbuf));

  std::string teststring = "testing log getting directed to a stringbuffer.";
  BAZEL_LOG(INFO) << teststring;

  // Check that output went to the buffer.
  std::string output(stringbuf_ptr->str());
  EXPECT_THAT(output, HasSubstr(teststring));

  // Check that the output never went to stderr.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_BufferedLogsSentToSpecifiedStream) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  std::string teststring =
      "test sending logs to the buffer before setting the output stream";
  BAZEL_LOG(INFO) << teststring;

  // Check that stderr isn't getting anything.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
  testing::internal::CaptureStderr();

  // Ask that the logs get output to a string buffer (keep a ptr to it so we can
  // check its contents)
  std::unique_ptr<std::stringstream> stringbuf(new std::stringstream());
  std::stringstream* stringbuf_ptr = stringbuf.get();
  blaze_util::SetLoggingOutputStream(std::move(stringbuf));

  // Check that the buffered logs were sent.
  std::string output(stringbuf_ptr->str());
  EXPECT_THAT(output,
              MatchesRegex(".bazel INFO.* test sending logs to the buffer "
                           "before setting the output stream\n"));

  // Check that the output did not go to stderr.
  stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_WarningsSentToBufferStream) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Ask that the logs get output to a string buffer (keep a ptr to it so we can
  // check its contents)
  std::unique_ptr<std::stringstream> stringbuf(new std::stringstream());
  std::stringstream* stringbuf_ptr = stringbuf.get();
  blaze_util::SetLoggingOutputStream(std::move(stringbuf));

  std::string teststring = "test warning";
  BAZEL_LOG(WARNING) << teststring;

  // Check that output went to the buffer.
  std::string output(stringbuf_ptr->str());
  EXPECT_THAT(output, MatchesRegex(".bazel WARNING.* test warning\n"));

  // Check that the output never went to stderr.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_ErrorsSentToBufferStream) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Ask that the logs get output to a string buffer (keep a ptr to it so we can
  // check its contents)
  std::unique_ptr<std::stringstream> stringbuf(new std::stringstream());
  std::stringstream* stringbuf_ptr = stringbuf.get();
  blaze_util::SetLoggingOutputStream(std::move(stringbuf));

  std::string teststring = "test error";
  BAZEL_LOG(ERROR) << teststring;

  // Check that output went to the buffer.
  std::string output(stringbuf_ptr->str());
  EXPECT_THAT(output, MatchesRegex(".bazel ERROR.* test error\n"));

  // Check that the output never went to stderr.
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, Not(HasSubstr(teststring)));
}

TEST(LoggingTest, BazelLogHandler_ImpossibleFile) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Deliberately try to log to an impossible location, check that we error out.
  std::unique_ptr<std::ofstream> bad_logfile_stream_(
      new std::ofstream("/this/doesnt/exist.log", std::fstream::out));
  blaze_util::SetLoggingOutputStream(std::move(bad_logfile_stream_));

  // Set a null log handler, which causes the BazelLogHandler to be destructed.
  // This prompts its logs to be flushed, so we can capture them..
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output,
              MatchesRegex(".bazel ERROR.* Provided stream failed.\n"));
}

// Tests for the BazelLogHandler & SetLoggingOutputStreamToStderr

TEST(LoggingTest, BazelLogHandler_DirectingLogsToCerrWorks) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Ask that the logs get output to stderr
  blaze_util::SetLoggingOutputStreamToStderr();

  // Log something.
  std::string teststring = "test that the log messages get directed to cerr";
  BAZEL_LOG(INFO) << teststring;

  // Set a null log handler, which causes the BazelLogHandler to be destructed.
  // This prompts its logs to be flushed, so we can capture them.
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(teststring));
}

TEST(LoggingTest, BazelLogHandler_BufferedLogsGetDirectedToCerr) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something before telling the loghandler where to send it.
  std::string teststring = "test that this message gets directed to cerr";
  BAZEL_LOG(INFO) << teststring;

  // Ask that the logs get output to stderr
  blaze_util::SetLoggingOutputStreamToStderr();

  // Set a null log handler, which causes the BazelLogHandler to be destructed.
  // This prompts its logs to be flushed, so we can capture them.
  blaze_util::SetLogHandler(nullptr);
  std::string stderr_output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(stderr_output, HasSubstr(teststring));
}

// We use the LoggingDeathTest test case to make sure that the death tests are
// run in a single threaded environment, where it is safe to fork. These tests
// are run before the other tests, which can be run in parallel.
#if GTEST_HAS_DEATH_TEST
using LoggingDeathTest = LoggingTest;

TEST(LoggingDeathTest, NoHandler_FatalStatementUsesInternalErrorCode) {
  // When no handler is specified, we still expect fatal messages to get
  // printed to stderr.
  ASSERT_EXIT({ BAZEL_LOG(FATAL) << "something's wrong!"; },
              ::testing::ExitedWithCode(37), "FATAL: something's wrong!");
}

TEST(LoggingDeathTest,
     BazelLogHandler_UnsetOutputStream_FatalStatementUsesInternalErrorCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        BAZEL_LOG(FATAL) << "something's wrong!";
      },
      ::testing::ExitedWithCode(37), "\\[bazel FATAL .*\\] something's wrong!");
}

TEST(LoggingDeathTest,
     BazelLogHandler_Deactivated_FatalStatementUsesInternalErrorCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        blaze_util::SetLoggingOutputStream(nullptr);

        BAZEL_LOG(FATAL) << "something's wrong!";
      },
      ::testing::ExitedWithCode(37), "FATAL: something's wrong!");
}

TEST(LoggingDeathTest,
     BazelLogHandler_Stderr_FatalStatementUsesInternalErrorCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        blaze_util::SetLoggingOutputStreamToStderr();
        BAZEL_LOG(FATAL) << "something's wrong!";
      },
      ::testing::ExitedWithCode(37), "\\[bazel FATAL .*\\] something's wrong!");
}

TEST(LoggingDeathTest, NoHandler_BazelDieDiesWithCustomExitCode) {
  ASSERT_EXIT({ BAZEL_DIE(42) << "dying with exit code 42."; },
              ::testing::ExitedWithCode(42), "FATAL: dying with exit code 42.");
}

TEST(LoggingDeathTest,
     BazelLogHandler_UnsetOutputStream_BazelDieDiesWithCustomExitCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        BAZEL_DIE(42) << "dying with exit code 42.";
      },
      ::testing::ExitedWithCode(42),
      "\\[bazel FATAL .*\\] dying with exit code 42.");
}

TEST(LoggingDeathTest,
     BazelLogHandler_Deactivated_BazelDieDiesWithCustomExitCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        blaze_util::SetLoggingOutputStream(nullptr);
        BAZEL_DIE(42) << "dying with exit code 42.";
      },
      ::testing::ExitedWithCode(42), "FATAL: dying with exit code 42.");
}

TEST(LoggingDeathTest, BazelLogHandler_Stderr_BazelDieDiesWithCustomExitCode) {
  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));
        blaze_util::SetLoggingOutputStreamToStderr();
        BAZEL_DIE(42) << "dying with exit code 42.";
      },
      ::testing::ExitedWithCode(42),
      "\\[bazel FATAL .*\\] dying with exit code 42.");
}

TEST(LoggingDeathTest,
     BazelLogHandler_CustomStream_BazelDiePrintsToStderrAndCustomStream) {
  std::string logfile =
      blaze_util::JoinPath(blaze::GetPathEnv("TEST_TMPDIR"), "logfile");

  ASSERT_EXIT(
      {
        std::unique_ptr<blaze_util::BazelLogHandler> handler(
            new blaze_util::BazelLogHandler());
        blaze_util::SetLogHandler(std::move(handler));

        // Ask that the logs get output to a file (the string buffer setup used
        // in the non-death tests doesn't work here.)
        std::unique_ptr<std::ofstream> logfile_stream_(
            new std::ofstream(logfile, std::fstream::out));
        blaze_util::SetLoggingOutputStream(std::move(logfile_stream_));

        BAZEL_DIE(42) << "dying with exit code 42.";
      },
      ::testing::ExitedWithCode(42), "FATAL: dying with exit code 42.");
  // Check that the error is also in the custom stream.
  std::string output;
  ASSERT_TRUE(blaze_util::ReadFile(logfile, &output));
  // Unlike in earlier tests, this string is read from a file, and since Windows
  // uses the newline '\r\n', compared to the linux \n, we prefer to keep the
  // test simple and not test the end of the line explicitly.
  EXPECT_THAT(output,
              ContainsRegex("\\[bazel FATAL .*\\] dying with exit code 42."));
}

#endif  // GTEST_HAS_DEATH_TEST
}  // namespace blaze_util
