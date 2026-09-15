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

#include "src/main/cpp/server_process_info.h"

#include "src/main/cpp/util/path.h"

namespace blaze {

static blaze_util::Path GetJvmOutFile(const blaze_util::Path &output_base,
                                      const blaze_util::Path &server_jvm_out) {
  if (!server_jvm_out.IsEmpty()) {
    return server_jvm_out;
  } else {
    return output_base.GetRelative("server/jvm.out");
  }
}

ServerProcessInfo::ServerProcessInfo(const blaze_util::Path &output_base,
                                     const blaze_util::Path &server_jvm_out)
    : jvm_log_file_(GetJvmOutFile(output_base, server_jvm_out)),
      jvm_log_file_append_(!server_jvm_out.IsEmpty()),
      // TODO(b/69972303): Formalize the "no server" magic value or rm it.
      server_pid_(-1) {}

}  // namespace blaze
