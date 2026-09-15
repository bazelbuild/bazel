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

#include "tools/cpp/modules_tools/common/common.h"

#include <gtest/gtest.h>

TEST(Cpp20ModulesInfoTest, BasicFunctionality) {
  std::string info_content = R"({
        "modules": {
            "module1": "/path/to/module1",
            "module2": "/path/to/module2"
        },
        "usages": {
            "module1": ["module2"],
            "module2": []
        }
    })";

  std::istringstream info_stream(info_content);
  Cpp20ModulesInfo full_info = parse_info(info_stream);

  EXPECT_EQ(full_info.modules["module1"], "/path/to/module1");
  EXPECT_EQ(full_info.modules["module2"], "/path/to/module2");
  EXPECT_EQ(full_info.usages["module1"].size(), 1);
  EXPECT_EQ(full_info.usages["module1"][0], "module2");
  EXPECT_EQ(full_info.usages["module2"].size(), 0);
}

TEST(Cpp20ModulesInfoTest, BasicFunctionalityWithTwoFile) {
  std::string info_content = R"({
        "modules": {
            "module1": "/path/to/module1",
            "module2": "/path/to/module2"
        },
        "usages": {
            "module1": ["module2"],
            "module2": []
        }
    })";

  std::string info_content2 = R"({
        "modules": {
            "foo": "/path/to/foo",
            "bar": "/path/to/bar"
        },
        "usages": {
            "foo": [],
            "bar": ["foo"]
        }
    })";

  std::istringstream info_stream(info_content);
  std::istringstream info_stream2(info_content2);

  Cpp20ModulesInfo full_info{};
  auto info1 = parse_info(info_stream);
  auto info2 = parse_info(info_stream2);
  full_info.merge(info1);
  full_info.merge(info2);

  EXPECT_EQ(full_info.modules["module1"], "/path/to/module1");
  EXPECT_EQ(full_info.modules["module2"], "/path/to/module2");
  EXPECT_EQ(full_info.modules["foo"], "/path/to/foo");
  EXPECT_EQ(full_info.modules["bar"], "/path/to/bar");
  EXPECT_EQ(full_info.usages["module1"].size(), 1);
  EXPECT_EQ(full_info.usages["module1"][0], "module2");
  EXPECT_EQ(full_info.usages["module2"].size(), 0);
  EXPECT_EQ(full_info.usages["bar"].size(), 1);
  EXPECT_EQ(full_info.usages["bar"][0], "foo");
  EXPECT_EQ(full_info.usages["foo"].size(), 0);
}

TEST(DdiTest, BasicFunctionality) {
  Cpp20ModulesInfo full_info;
  std::string ddi_content = R"({
        "rules": [{
            "provides": [{
                "logical-name": "foo"
            }],
            "requires": [{
                "logical-name": "bar"
            }]
        }]
    })";

  std::istringstream ddi_stream(ddi_content);
  auto ddi = parse_ddi(ddi_stream);
  EXPECT_EQ(ddi.name, "foo");
  EXPECT_EQ(ddi.gen_bmi, true);
  EXPECT_EQ(ddi.require_list, std::vector<std::string>{"bar"});
}

TEST(DdiTest, BasicEmpty) {
  std::string ddi_content = R"(
  {
    "revision": 0,
    "rules": [
      {
        "primary-output": "main.ddi"
      }
    ],
    "version": 1
  })";
  std::istringstream ddi_stream(ddi_content);
  auto ddi = parse_ddi(ddi_stream);
  EXPECT_EQ(ddi.name, "");
  EXPECT_EQ(ddi.gen_bmi, false);
  EXPECT_TRUE(ddi.require_list.empty());
}

TEST(DdiTest, EmptyRequires) {
  Cpp20ModulesInfo full_info;
  std::string ddi_content = R"({
        "rules": [{
            "provides": [{
                "logical-name": "foo"
            }]
        }]
    })";

  std::istringstream ddi_stream(ddi_content);
  auto ddi = parse_ddi(ddi_stream);
  EXPECT_EQ(ddi.name, "foo");
  EXPECT_EQ(ddi.gen_bmi, true);
  EXPECT_TRUE(ddi.require_list.empty());
}

TEST(DdiTest, EmptyProvides) {
  Cpp20ModulesInfo full_info;
  std::string ddi_content = R"({
        "rules": [{
            "requires": [{
                "logical-name": "bar"
            }]
        }]
    })";

  std::istringstream ddi_stream(ddi_content);
  auto ddi = parse_ddi(ddi_stream);
  EXPECT_EQ(ddi.name, "");
  EXPECT_EQ(ddi.gen_bmi, false);
  EXPECT_EQ(ddi.require_list, std::vector<std::string>{"bar"});
}
