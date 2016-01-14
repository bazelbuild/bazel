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
//
//
// blaze_abrupt_exit.h: Deals with abrupt exits of the Blaze server.
//
#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_ABRUPT_EXIT_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_ABRUPT_EXIT_H_

namespace blaze {

struct GlobalVariables;

// Returns the exit code to use for when the Blaze server exits abruptly.
int GetExitCodeForAbruptExit(const GlobalVariables& globals);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_ABRUPT_EXIT_H_
