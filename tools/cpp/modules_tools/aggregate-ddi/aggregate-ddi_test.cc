// Copyright 2024 The Bazel Authors. All rights reserved.
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

#include "aggregate-ddi.h"
#include <gtest/gtest.h>

TEST(WriteOutputTest, BasicFunctionality) {
  Cpp20ModulesInfo info;
  info.modules["module1"] = "/path/to/module1";
  info.modules["module2"] = "/path/to/module2";
  info.usages["module1"].push_back("module2");

  std::ostringstream output_stream;
  write_output(output_stream, info);

  std::string expected_output =
      R"({"modules":{"module1":"/path/to/module1","module2":"/path/to/module2"},"usages":{"module1":["module2"]}})";
  EXPECT_EQ(output_stream.str(), expected_output);
}
