// Copyright 2018 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_TEST_CPP_TEST_UTIL_H
#define BAZEL_SRC_TEST_CPP_TEST_UTIL_H

#include "src/main/cpp/startup_options.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

void ExpectIsNullaryOption(const StartupOptions* options,
                           const std::string& flag_name);
void ExpectIsUnaryOption(const StartupOptions* options,
                         const std::string& flag_name);
void ParseStartupOptionsAndExpectWarning(
    StartupOptions* startup_options,
    const std::vector<std::string>& options_to_parse,
    const std::string& expected_warning);

}  // namespace blaze

#endif  // BAZEL_SRC_TEST_CPP_TEST_UTIL_H
