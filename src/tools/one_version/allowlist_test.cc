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

#include "src/tools/one_version/allowlist.h"

#include <string>

#include "googletest/include/gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "src/tools/one_version/duplicate_class_collector.h"

namespace one_version {

class AllowlistTest : public ::testing::Test {};

TEST_F(AllowlistTest, Allowlisting) {
  DuplicateClassCollector vc;
  vc.Add("com/google/Foo", 1, Label("//hello:foo", "hello/libfoo.jar"));
  vc.Add("com/google/Foo", 2, Label("//hello:bar", "hello/libbar.jar"));
  vc.Add("com/google/Foo", 2, Label("//hello:baz", "hello/libbaz.jar"));
  std::string expected =
      "  com/google/Foo has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:foo [new]\n"
      "      via hello/libfoo.jar\n"
      "    crc32=2\n"
      "      //hello:bar [new]\n"
      "      via hello/libbar.jar\n"
      "      //hello:baz [new]\n"
      "      via hello/libbaz.jar\n";
  EXPECT_EQ(expected, DuplicateClassCollector::Report(vc.Violations()));

  MapAllowlist allowlist(
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>(
          {{"com/google", {"//hello:bar"}}}));
  expected =
      "  com/google/Foo has incompatible definitions in:\n"
      "    crc32=1\n"
      "      //hello:foo [new]\n"
      "      via hello/libfoo.jar\n"
      "    crc32=2\n"
      "      //hello:bar [allowlisted]\n"
      "      via hello/libbar.jar\n"
      "      //hello:baz [new]\n"
      "      via hello/libbaz.jar\n";
  EXPECT_EQ(expected,
            DuplicateClassCollector::Report(allowlist.Apply(vc.Violations())));

  allowlist = MapAllowlist(
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>(
          {{"com/google", {"//hello:bar", "//hello:foo"}}}));
  EXPECT_TRUE(allowlist.Apply(vc.Violations()).empty());

  allowlist = MapAllowlist(
      absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>(
          {{"com/google", {"//hello:foo"}}}));
  EXPECT_TRUE(allowlist.Apply(vc.Violations()).empty());
}

}  // namespace one_version
