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
#include <fstream>
#include <iostream>
#include <sstream>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"

namespace blaze_util {

BazelLogHandler::BazelLogHandler()
    : output_dir_set_attempted_(false),
      buffer_stream_(new std::stringstream()),
      logfile_stream_(nullptr) {}

BazelLogHandler::~BazelLogHandler() {
  // If we never wrote the logs to a file, dump the buffer to stderr,
  // otherwise, flush the stream.
  if (logfile_stream_ != nullptr) {
    logfile_stream_->flush();
  } else if (buffer_stream_ != nullptr) {
    std::cerr << buffer_stream_->rdbuf();
  } else {
    std::cerr << "Illegal state - neither a logfile nor a logbuffer "
              << "existed at program end." << std::endl;
  }
}

void BazelLogHandler::HandleMessage(LogLevel level, const std::string& filename,
                                    int line, const std::string& message) {
  // Select the appropriate stream to log to.
  std::ostream* log_stream;
  if (logfile_stream_ != nullptr) {
    log_stream = logfile_stream_.get();
  } else {
    log_stream = buffer_stream_.get();
  }
  *log_stream << "[bazel " << LogLevelName(level) << " " << filename << ":"
              << line << "] " << message << "\n";

  // If we have a fatal message, we should abort and leave a stack trace -
  // normal exit behavior will be lost, so print this log message out to
  // stderr and avoid loosing the information.
  if (level == LOGLEVEL_FATAL) {
    std::cerr << "[bazel " << LogLevelName(level) << " " << filename << ":"
              << line << "] " << message << "\n";
    std::abort();
  }
}

void BazelLogHandler::SetOutputDir(const std::string& new_output_dir) {
  // Disallow second calls to this, we only intend this to support setting
  // output_base once it is created, not changing the log location.
  BAZEL_CHECK(!output_dir_set_attempted_)
      << "Tried to SetOutputDir a second time, to " << new_output_dir;
  output_dir_set_attempted_ = true;

  // Create a log file in the newly available directory, and flush the
  // buffer to it.
  const std::string logfile = JoinPath(new_output_dir, "bazel_client.log");
  logfile_stream_ = std::unique_ptr<std::ofstream>(
      new std::ofstream(logfile, std::fstream::out));
  if (logfile_stream_->fail()) {
    // If opening the stream failed, continue buffering and have the logs
    // dump to stderr at shutdown.
    logfile_stream_ = nullptr;
    BAZEL_LOG(ERROR) << "Opening the log file failed, in directory "
                     << new_output_dir;
  } else {
    // Transfer the contents of the buffer to the logfile's stream before
    // replacing it.
    *logfile_stream_ << buffer_stream_->rdbuf();
    buffer_stream_ = nullptr;
    logfile_stream_->flush();
  }
}

}  // namespace blaze_util
