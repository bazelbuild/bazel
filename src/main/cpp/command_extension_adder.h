// Copyright 2025 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_MAIN_CPP_COMMAND_EXTENSION_ADDER_H_
#define THIRD_PARTY_BAZEL_SRC_MAIN_CPP_COMMAND_EXTENSION_ADDER_H_

#include "src/main/protobuf/command_server.pb.h"

namespace blaze {

// Interface for adding command extensions to a RunRequest.
class CommandExtensionAdder {
 public:
  CommandExtensionAdder() = default;
  virtual ~CommandExtensionAdder() = default;
  virtual void MaybeAddCommandExtensions(
      command_server::RunRequest& run_request) = 0;
};

}  // namespace blaze

#endif  // THIRD_PARTY_BAZEL_SRC_MAIN_CPP_COMMAND_EXTENSION_ADDER_H_
