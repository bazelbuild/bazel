// Copyright 2014 The Bazel Authors. All rights reserved.
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

#include "src/main/cpp/util/port.h"

#ifdef __linux
#include <sys/syscall.h>
#include <unistd.h>
#endif  // __linux

namespace blaze_util {

#ifdef __linux

int sys_ioprio_set(int which, int who, int ioprio) {
  return syscall(SYS_ioprio_set, which, who, ioprio);
}

#else  // Not Linux.

int sys_ioprio_set(int which, int who, int ioprio) {
  return 0;
}

#endif  // __linux

}  // namespace blaze_util
