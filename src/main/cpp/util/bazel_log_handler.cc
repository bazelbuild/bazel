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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {

BazelLogHandler::BazelLogHandler()
    : output_stream_set_(false),
      logging_deactivated_(false),
      buffer_stream_(new std::stringstream()),
      output_stream_(),
      owned_output_stream_() {}

BazelLogHandler::~BazelLogHandler() {
  if (!logging_deactivated_) {
    // If SetLoggingOutputStream was never called, dump the buffer to stderr,
    // otherwise, flush the stream.
    if (output_stream_ != nullptr) {
      output_stream_->flush();
    } else if (buffer_stream_ != nullptr) {
      std::cerr << buffer_stream_->rdbuf();
    } else {
      std::cerr << "Illegal state - neither a logfile nor a logbuffer "
                << "existed at program end." << std::endl;
    }
  }
}

void BazelLogHandler::HandleMessage(LogLevel level, const std::string& filename,
                                    int line, const std::string& message,
                                    int exit_code) {
  // Select the appropriate stream to log to.
  std::ostream* log_stream;
  if (logging_deactivated_) {
    // If the output stream was explicitly deactivated, never print INFO
    // messages, but messages of level USER and above should always be printed,
    // as should warnings and errors. Omit the debug-level file and line number
    // information, though.
    if (level == LOGLEVEL_USER) {
      std::cerr << message << std::endl;
    } else if (level > LOGLEVEL_USER) {
      std::cerr << LogLevelName(level) << ": " << message << std::endl;
    }

    if (level == LOGLEVEL_FATAL) {
      std::exit(exit_code);
    }
    return;
  } else if (output_stream_ == nullptr) {
    log_stream = buffer_stream_.get();
  } else {
    log_stream = output_stream_;
  }
  (*log_stream) << "[bazel " << LogLevelName(level) << " " << filename << ":"
                << line << "] " << message << std::endl;

  // If we have a fatal message, exit with the provided error code.
  if (level == LOGLEVEL_FATAL) {
    if (owned_output_stream_ != nullptr) {
      // If this is is not being printed to stderr but to a custom stream,
      // also print the error message to stderr.
      std::cerr << "[bazel " << LogLevelName(level) << " " << filename << ":"
                << line << "] " << message << std::endl;
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

  FlushBufferToNewStreamAndSet(&std::cerr);
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
    buffer_stream_ = nullptr;
    return;
  }
  owned_output_stream_ = std::move(new_output_stream);
  FlushBufferToNewStreamAndSet(owned_output_stream_.get());
}

void BazelLogHandler::FlushBufferToNewStreamAndSet(
    std::ostream* new_output_stream) {
  // Flush the buffer to the new stream, and print new log lines to it.
  output_stream_ = new_output_stream;
  if (output_stream_->fail()) {
    // If opening the stream failed, continue buffering and have the logs
    // dump to stderr at shutdown.
    output_stream_ = nullptr;
    BAZEL_LOG(ERROR) << "Provided stream failed.";
  } else {
    // Transfer the contents of the buffer to the new stream, then remove the
    // buffer.
    (*output_stream_) << buffer_stream_->str();
    buffer_stream_ = nullptr;
    output_stream_->flush();
  }
}

}  // namespace blaze_util
