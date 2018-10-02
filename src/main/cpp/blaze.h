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
#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_H_

#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/workspace_layout.h"
#include "src/main/cpp/util/logging.h"

namespace blaze {

int Main(int argc, const char* argv[], WorkspaceLayout* workspace_layout,
         OptionProcessor* option_processor, uint64_t start_time);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_H_
