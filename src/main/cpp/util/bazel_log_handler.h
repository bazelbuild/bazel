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

#ifndef BAZEL_SRC_MAIN_CPP_BAZEL_LOG_HANDLER_H_
#define BAZEL_SRC_MAIN_CPP_BAZEL_LOG_HANDLER_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/util/logging.h"

namespace blaze_util {

// Handles logging for the Bazel client.
// In order to control where the log output goes based on the flags received at
// startup, logs are buffered until SetOutputStream is called. At that point,
// all past log statements are dumped in the appropriate stream, and all
// following statements are logged directly.
class BazelLogHandler : public blaze_util::LogHandler {
 public:
  BazelLogHandler();
  ~BazelLogHandler() override;

  void HandleMessage(blaze_util::LogLevel level, const std::string& filename,
                     int line, const std::string& message,
                     int exit_code) override;
  void SetLoggingDetail(blaze_util::LoggingDetail detail,
                        std::ostream* stream) override;
  void Close() override;

 private:
  void FlushBufferToNewStreamAndSet(std::stringstream* buffer,
                                    std::ostream* new_output_stream);

  // The stream to which debug logs are sent (if logging detail is not
  // LOGGINGDETAIL_DEBUG, everything goes to stderr)
  std::ostream* debug_stream_;
  bool debug_stream_set_;
  LoggingDetail detail_;

  // Buffers for messages received before the logging detail was determined.
  // non-debug messages are buffered alongside their log level so that we can
  // use the log level to filter them based on the eventual logging detail,
  // debug messages are simply buffered as a stream.
  std::vector<std::pair<blaze_util::LogLevel, std::string>> user_messages_;
  std::unique_ptr<std::stringstream> debug_buffer_stream_;
};
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_BAZEL_LOG_HANDLER_H_
