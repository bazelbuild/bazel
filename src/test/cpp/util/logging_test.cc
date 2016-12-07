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
#include <iostream>
#include <memory>
#include <string>

#include "src/main/cpp/util/bazel_log_handler.h"
#include "src/main/cpp/util/logging.h"
#include "gtest/gtest.h"

namespace blaze_util {

TEST(LoggingTest, BazelLogHandlerDumpsToCerrAtFail) {
  // Set up logging and be prepared to capture stderr at destruction.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Log something.
  std::string teststring = "test that the log messages get dumped to stderr";
  BAZEL_LOG(INFO) << teststring;

  // Check that stderr isn't getting anything yet.
  std::string nothing = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(nothing.find(teststring) == std::string::npos);
  testing::internal::CaptureStderr();

  // Destruct the log handler and get the stderr remains.
  blaze_util::SetLogHandler(nullptr);
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(output.find(teststring) != std::string::npos);
}

TEST(LoggingTest, LogLevelNamesMatch) {
  EXPECT_STREQ("INFO", LogLevelName(LOGLEVEL_INFO));
  EXPECT_STREQ("WARNING", LogLevelName(LOGLEVEL_WARNING));
  EXPECT_STREQ("ERROR", LogLevelName(LOGLEVEL_ERROR));
  EXPECT_STREQ("FATAL", LogLevelName(LOGLEVEL_FATAL));
}

TEST(LoggingTest, ImpossibleFile) {
  // Set up to capture logging to stderr.
  testing::internal::CaptureStderr();
  std::unique_ptr<blaze_util::BazelLogHandler> handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(handler));

  // Deliberately try to log to an impossible location, check that we error out.
  blaze_util::SetLogfileDirectory("/this/doesnt/exist");

  // Cause the logs to be flushed, and capture them.
  blaze_util::SetLogHandler(nullptr);
  std::string output = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(output.find("ERROR") != std::string::npos);
  ASSERT_TRUE(output.find("/this/doesnt/exist") != std::string::npos);
}

}  // namespace blaze_util
