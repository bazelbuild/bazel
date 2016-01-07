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
#include "src/main/cpp/blaze_abrupt_exit.h"
#include "src/main/cpp/util/exit_code.h"

namespace blaze {

int GetExitCodeForAbruptExit(const GlobalVariables& globals) {
  return blaze_exit_code::INTERNAL_ERROR;
}

}  // namespace blaze
