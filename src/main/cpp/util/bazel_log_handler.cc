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

#include "src/main/cpp/util/bazel_log_handler.h"

#include <chrono>  // NOLINT -- for windows portability
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>

#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {

BazelLogHandler::BazelLogHandler()
    : output_stream_set_(false),
      logging_deactivated_(false),
      user_buffer_stream_(new std::stringstream()),
      debug_buffer_stream_(new std::stringstream()),
      output_stream_(),
      owned_output_stream_() {}

BazelLogHandler::~BazelLogHandler() {
  if (!logging_deactivated_) {
    // If SetLoggingOutputStream was never called, dump the buffer to stderr,
    // otherwise, flush the stream.
    if (output_stream_ != nullptr) {
      output_stream_->flush();
    } else if (debug_buffer_stream_ != nullptr) {
      std::cerr << debug_buffer_stream_->rdbuf();
    } else {
      std::cerr << "Illegal state - neither a logfile nor a logbuffer "
                << "existed at program end." << std::endl;
    }
  }
}

// Messages intended for the user (level USER, along with WARNINGs an ERRORs)
// should be printed even if debug level logging was not requested.
void PrintUserLevelMessageToStream(std::ostream* stream, LogLevel level,
                                   const std::string& message) {
  if (level == LOGLEVEL_USER) {
    (*stream) << message << std::endl;
  } else if (level > LOGLEVEL_USER) {
    (*stream) << LogLevelName(level) << ": " << message << std::endl;
  }
  // If level < USER, this is an INFO message. It's useful for debugging but
  // should not be printed to the user unless the user has asked for debugging
  // output. We ignore it here.
}

static std::string Timestamp() {
  auto now = std::chrono::system_clock::now();
  time_t s = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch());
  struct tm tmbuf = {};
#ifdef _WIN32
  tmbuf = *localtime(&s);  // NOLINT -- threadsafe on windows
#else
  localtime_r(&s, &tmbuf);
#endif
  char buf[16];
  int r = strftime(buf, sizeof buf - 5, "%H:%M:%S", &tmbuf);
  r += snprintf(buf + r, +5, ".%03d", static_cast<int>(ms.count() % 1000));
  return std::string(buf, r);
}

// For debug logs, print all logs, both debug logging and USER logs and above,
// along with information about where the log message came from.
void PrintDebugLevelMessageToStream(std::ostream* stream,
                                    const std::string& filename, int line,
                                    LogLevel level,
                                    const std::string& message) {
  (*stream) << "[" << LogLevelName(level) << " " << Timestamp() << " "
            << filename << ":" << line << "] " << message << std::endl;
}

void BazelLogHandler::HandleMessage(LogLevel level, const std::string& filename,
                                    int line, const std::string& message,
                                    int exit_code) {
  if (logging_deactivated_) {
    // If the output stream was explicitly deactivated, never print INFO
    // messages, but messages of level USER and above should always be printed,
    // as should warnings and errors. Omit the debug-level file and line number
    // information, though.
    PrintUserLevelMessageToStream(&std::cerr, level, message);
    if (level == LOGLEVEL_FATAL) {
      std::exit(exit_code);
    }
    return;
  }
  if (output_stream_ == nullptr) {
    // If we haven't decided whether messages should be logged to debug levels
    // or not, buffer each version. This is redundant for USER levels and above,
    // but is to make sure we can provide the right output to the user once we
    // know that they do or do not want debug level information.
    PrintUserLevelMessageToStream(user_buffer_stream_.get(), level, message);
    PrintDebugLevelMessageToStream(debug_buffer_stream_.get(), filename, line,
                                   level, message);
  } else {
    // If an output stream has been specifically set, it is for the full suite
    // of log messages. We don't print the user messages separately here as they
    // are included.
    PrintDebugLevelMessageToStream(output_stream_, filename, line, level,
                                   message);
  }

  // If we have a fatal message, exit with the provided error code.
  if (level == LOGLEVEL_FATAL) {
    if (owned_output_stream_ != nullptr) {
      // If this is is not being printed to stderr but to a custom stream,
      // also print the error message to stderr.
      PrintUserLevelMessageToStream(&std::cerr, level, message);
    }
    std::exit(exit_code);
  }
}

void BazelLogHandler::SetOutputStreamToStderr() {
  // Disallow second calls to this, we only intend to support setting the output
  // once, otherwise the buffering will not work as intended and the log will be
  // fragmented.
  BAZEL_CHECK(!output_stream_set_) << "Tried to set log output a second time";
  output_stream_set_ = true;

  FlushBufferToNewStreamAndSet(debug_buffer_stream_.get(), &std::cerr);
  debug_buffer_stream_ = nullptr;
  // The user asked for debug level information, which includes the user
  // messages. We can discard the separate buffer at this point.
  user_buffer_stream_ = nullptr;
}

void BazelLogHandler::SetOutputStream(
    std::unique_ptr<std::ostream> new_output_stream) {
  // Disallow second calls to this, we only intend to support setting the output
  // once, otherwise the buffering will not work as intended and the log will be
  // fragmented.
  BAZEL_CHECK(!output_stream_set_) << "Tried to set log output a second time";
  output_stream_set_ = true;

  if (new_output_stream == nullptr) {
    logging_deactivated_ = true;
    // Flush the buffered user-level messages to stderr - these are messages
    // that are meant for the user even when debug logging is not set.
    FlushBufferToNewStreamAndSet(user_buffer_stream_.get(), &std::cerr);

    user_buffer_stream_ = nullptr;
    // We discard the debug level logs, the user level ones were enough to
    // inform the user and debug logging was not requested.
    debug_buffer_stream_ = nullptr;
    return;
  }
  owned_output_stream_ = std::move(new_output_stream);
  if (owned_output_stream_->fail()) {
    // If opening the stream failed, continue buffering and have the logs
    // dump to stderr at shutdown.
    BAZEL_LOG(ERROR) << "Provided stream failed.";
    return;
  }
  FlushBufferToNewStreamAndSet(debug_buffer_stream_.get(),
                               owned_output_stream_.get());
  debug_buffer_stream_ = nullptr;
  // The user asked for debug level information, which includes the user
  // messages. We can discard the separate buffer at this point.
  user_buffer_stream_ = nullptr;
}

void BazelLogHandler::FlushBufferToNewStreamAndSet(
    std::stringstream* buffer, std::ostream* new_output_stream) {
  // Flush the buffer to the new stream, and print new log lines to it.
  output_stream_ = new_output_stream;
    // Transfer the contents of the buffer to the new stream, then remove the
    // buffer.
  (*output_stream_) << buffer->str();
  output_stream_->flush();
}

}  // namespace blaze_util
