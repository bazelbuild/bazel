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
  void SetOutputStream(
      std::unique_ptr<std::ostream> new_output_stream) override;
  void SetOutputStreamToStderr() override;

 private:
  void FlushBufferToNewStreamAndSet(std::stringstream* buffer,
                                    std::ostream* new_output_stream);
  bool output_stream_set_;
  bool logging_deactivated_;
  std::unique_ptr<std::stringstream> user_buffer_stream_;
  std::unique_ptr<std::stringstream> debug_buffer_stream_;
  // The actual output_stream to which all logs will be sent.
  std::ostream* output_stream_;
  // A unique pts to the output_stream, if we need to keep ownership of the
  // stream. In the case of stderr logging, this is null.
  std::unique_ptr<std::ostream> owned_output_stream_;
};
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_BAZEL_LOG_HANDLER_H_
