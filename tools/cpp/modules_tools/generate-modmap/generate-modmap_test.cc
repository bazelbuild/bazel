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

#include "tools/cpp/modules_tools/generate-modmap/generate-modmap.h"

#include <gtest/gtest.h>

#include <sstream>

TEST(ModmapTest, EmptyInput) {
  ModuleDep dep{};
  Cpp20ModulesInfo info{};
  auto modmap = process(dep, info);

  std::set<ModmapItem> expected_modmap;
  EXPECT_EQ(modmap, expected_modmap);
}

TEST(ModmapTest, BasicFunctionality) {
  ModuleDep dep{};
  Cpp20ModulesInfo info{};

  dep.require_list.push_back("module1");
  dep.require_list.push_back("module2");

  info.modules["module1"] = "/path/to/module1";
  info.modules["module2"] = "/path/to/module2";

  info.usages["module1"].push_back("module2");

  auto modmap = process(dep, info);

  std::set<ModmapItem> expected_modmap = {{"module1", "/path/to/module1"},
                                          {"module2", "/path/to/module2"}};
  EXPECT_EQ(modmap, expected_modmap);
}

TEST(ModmapTest, BasicFunctionality2) {
  ModuleDep dep{};
  Cpp20ModulesInfo info{};

  dep.require_list.push_back("module1");

  info.modules["module1"] = "/path/to/module1";
  info.modules["module2"] = "/path/to/module2";
  info.modules["module3"] = "/path/to/module3";

  info.usages["module1"].push_back("module2");
  info.usages["module2"].push_back("module3");

  auto modmap = process(dep, info);

  std::set<ModmapItem> expected_modmap = {{"module1", "/path/to/module1"},
                                          {"module2", "/path/to/module2"},
                                          {"module3", "/path/to/module3"}};
  EXPECT_EQ(modmap, expected_modmap);
}

TEST(ModmapTest, BasicFunctionality3) {
  ModuleDep dep{};
  Cpp20ModulesInfo info{};

  dep.require_list.push_back("module1");
  dep.require_list.push_back("module4");

  info.modules["module1"] = "/path/to/module1";
  info.modules["module2"] = "/path/to/module2";
  info.modules["module3"] = "/path/to/module3";
  info.modules["module4"] = "/path/to/module4";

  info.usages["module1"].push_back("module2");

  auto modmap = process(dep, info);

  std::set<ModmapItem> expected_modmap = {{"module1", "/path/to/module1"},
                                          {"module2", "/path/to/module2"},
                                          {"module4", "/path/to/module4"}};
  EXPECT_EQ(modmap, expected_modmap);
}

// if use std::unordered_set, the order of iteration is not deterministic
// therefore, use std::set for ModmapItem
TEST(ModmapTest, DeterministicIterationOrderOfWriteModmap) {
  std::stringstream modmap_file_stream1;
  std::stringstream modmap_file_dot_input_stream1;
  write_modmap(modmap_file_stream1, modmap_file_dot_input_stream1,
               {
                   {"module1", "/path/to/module1"},
                   {"module2", "/path/to/module2"},
                   {"module3", "/path/to/module3"},
               },
               "clang", std::nullopt);
  std::stringstream modmap_file_stream2;
  std::stringstream modmap_file_dot_input_stream2;
  write_modmap(modmap_file_stream2, modmap_file_dot_input_stream2,
               {
                   {"module2", "/path/to/module2"},
                   {"module1", "/path/to/module1"},
                   {"module3", "/path/to/module3"},
               },
               "clang", std::nullopt);
  EXPECT_EQ(modmap_file_stream1.str(), modmap_file_stream2.str());
  EXPECT_EQ(modmap_file_dot_input_stream1.str(),
            modmap_file_dot_input_stream2.str());
}
