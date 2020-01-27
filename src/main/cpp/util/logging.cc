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

// This file is based off the logging work by the protobuf team
#include "src/main/cpp/util/logging.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

const char* LogLevelName(LogLevel level) {
  static const char* level_names[] = {"INFO", "USER", "WARNING", "ERROR",
                                      "FATAL"};
  BAZEL_CHECK(static_cast<int>(level) < 5)
      << "LogLevelName: level out of range, there are only 5 levels.";
  return level_names[level];
}

namespace internal {

static std::unique_ptr<LogHandler> log_handler_(nullptr);

LogMessage::LogMessage(LogLevel level, const std::string& filename, int line)
    : level_(level),
      filename_(filename),
      line_(line),
      exit_code_(blaze_exit_code::INTERNAL_ERROR) {}

LogMessage::LogMessage(LogLevel level, const std::string& filename, int line,
                       int exit_code)
    : level_(level), filename_(filename), line_(line), exit_code_(exit_code) {}

#undef DECLARE_STREAM_OPERATOR
#define DECLARE_STREAM_OPERATOR(TYPE)              \
  LogMessage& LogMessage::operator<<(TYPE value) { \
    message_ << value;                             \
    return *this;                                  \
  }

DECLARE_STREAM_OPERATOR(const std::string&)
DECLARE_STREAM_OPERATOR(const char*)
DECLARE_STREAM_OPERATOR(char)
DECLARE_STREAM_OPERATOR(bool)
DECLARE_STREAM_OPERATOR(short)
DECLARE_STREAM_OPERATOR(int)
DECLARE_STREAM_OPERATOR(unsigned int)
DECLARE_STREAM_OPERATOR(long)
DECLARE_STREAM_OPERATOR(unsigned long)
DECLARE_STREAM_OPERATOR(long long)
DECLARE_STREAM_OPERATOR(unsigned long long)
DECLARE_STREAM_OPERATOR(float)
DECLARE_STREAM_OPERATOR(double)
DECLARE_STREAM_OPERATOR(long double)
DECLARE_STREAM_OPERATOR(void*)

#undef DECLARE_STREAM_OPERATOR

void LogMessage::Finish() {
  std::string message(message_.str());
  if (log_handler_ != nullptr) {
    log_handler_->HandleMessage(level_, filename_, line_, message, exit_code_);
  } else {
    // If no custom handler was provided, never print INFO messages,
    // but USER should always be printed, as should warnings and errors.
    if (level_ == LOGLEVEL_USER) {
      std::cerr << message << std::endl;
    } else if (level_ > LOGLEVEL_USER) {
      std::cerr << LogLevelName(level_) << ": " << message << std::endl;
    }

    if (level_ == LOGLEVEL_FATAL) {
      // Exit for fatal calls after handling the message.
      std::exit(exit_code_);
    }
  }
}

void LogFinisher::operator=(LogMessage& other) { other.Finish(); }

}  // namespace internal

void SetLogHandler(std::unique_ptr<LogHandler> new_handler) {
  internal::log_handler_ = std::move(new_handler);
}

void SetLoggingOutputStream(std::unique_ptr<std::ostream> output_stream) {
  if (internal::log_handler_ != nullptr) {
    internal::log_handler_->SetOutputStream(std::move(output_stream));
  }
}
void SetLoggingOutputStreamToStderr() {
  if (internal::log_handler_ != nullptr) {
    internal::log_handler_->SetOutputStreamToStderr();
  }
}

}  // namespace blaze_util
