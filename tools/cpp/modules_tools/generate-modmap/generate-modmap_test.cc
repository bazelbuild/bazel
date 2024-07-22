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

#include "generate-modmap.h"
#include <gtest/gtest.h>

TEST(ModmapTest, EmptyInput) {
  ModuleDep dep{};
  Cpp20ModulesInfo info{};
  auto modmap = process(dep, info);

  std::unordered_set<ModmapItem> expected_modmap;
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

  std::unordered_set<ModmapItem> expected_modmap = {
      {"module1", "/path/to/module1"}, {"module2", "/path/to/module2"}};
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

  std::unordered_set<ModmapItem> expected_modmap = {
      {"module1", "/path/to/module1"},
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

  std::unordered_set<ModmapItem> expected_modmap = {
      {"module1", "/path/to/module1"},
      {"module2", "/path/to/module2"},
      {"module4", "/path/to/module4"}};
  EXPECT_EQ(modmap, expected_modmap);
}
