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

#include "src/main/native/darwin/util.h"

#include "src/main/native/macros.h"

namespace bazel {
namespace darwin {

dispatch_queue_t JniDispatchQueue() {
  static dispatch_once_t once_token;
  static dispatch_queue_t queue;
  dispatch_once(&once_token, ^{
    queue = dispatch_queue_create("build.bazel.jni", DISPATCH_QUEUE_SERIAL);
    CHECK(queue);
  });
  return queue;
}

os_log_t JniOSLog() {
  static dispatch_once_t once_token;
  static os_log_t log = nullptr;
  // On macOS < 10.12, os_log_create is not available. Since we target 10.10,
  // this will be weakly linked and can be checked for availability at run
  // time.
  if (&os_log_create != nullptr) {
    dispatch_once(&once_token, ^{
      log = os_log_create("build.bazel", "jni");
      CHECK(log);
    });
  }
  return log;
}

}  // namespace darwin
}  // namespace bazel
