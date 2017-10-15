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
// In order to have the logfile in output_base, which does not exist or is
// unknown at the time of the client's creation, logs are buffered until
// SetOutputDir is called. At that point, all past log statements are dumped
// in the appropriate file, and all following statements are logged directly.
class BazelLogHandler : public blaze_util::LogHandler {
 public:
  BazelLogHandler();
  ~BazelLogHandler() override;

  void HandleMessage(blaze_util::LogLevel level, const std::string& filename,
                     int line, const std::string& message) override;

  // Sets the output directory of the logfile.
  // Can only be called once - all logs before this call will be buffered and
  // dumped to the logfile once this is called. If this is never called, or if
  // creating the logfile failed, the buffered logs will be dumped to stderr at
  // destruction.
  void SetOutputDir(const std::string& new_output_dir) override;

 private:
  bool output_dir_set_attempted_;
  std::unique_ptr<std::stringstream> buffer_stream_;
  std::unique_ptr<std::ofstream> logfile_stream_;
};
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_BAZEL_LOG_HANDLER_H_
