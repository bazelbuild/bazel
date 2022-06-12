// Copyright 2021 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_MAIN_NATIVE_DARWIN_JNI_UTIL_H_
#define BAZEL_SRC_MAIN_NATIVE_DARWIN_JNI_UTIL_H_

#include <dispatch/dispatch.h>
#include <os/log.h>

namespace bazel {
namespace darwin {

// Queue used for all of our anomaly tracking.
dispatch_queue_t JniDispatchQueue();

// Log used for all of our anomaly logging.
// Logging can be traced using:
// `log stream -level debug --predicate '(subsystem == "build.bazel")'`
//
// This may return NULL if `os_log_create` is not supported on this version of
// macOS. Use `log_if_possible` to log when supported.
os_log_t JniOSLog();

}  // namespace darwin
}  // namespace bazel

// The macOS implementation asserts that `msg` be a string literal (not just a
// const char*), so we cannot use a function.
#define log_if_possible(msg...)                    \
  do {                                             \
    os_log_t log = bazel::darwin::JniOSLog();      \
    if (log != nullptr) {                          \
      os_log_debug(log, msg);                      \
    }                                              \
  } while (0);

#endif  // BAZEL_SRC_MAIN_NATIVE_DARWIN_JNI_UTIL_H_

