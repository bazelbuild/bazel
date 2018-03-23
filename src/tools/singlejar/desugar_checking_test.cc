// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include "src/tools/singlejar/desugar_checking.h"

#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/zip_headers.h"
#include "src/tools/singlejar/zlib_interface.h"
#include "googletest/include/gtest/gtest.h"

// A test fixture is used because friend access to class under test is needed.
// Tests are instance methods to avoid gUnit dep in .h file.
class Java8DesugarDepsCheckerTest : public ::testing::Test {
 protected:
  static void TestHasDefaultMethods() {
    Java8DesugarDepsChecker checker([](const std::string &) { return false; },
                                    /*verbose=*/false);
    checker.has_default_methods_["a"] = true;
    checker.extended_interfaces_["c"] = {"b", "a"};

    // Induce cycle (shouldn't happen but make sure we don't crash)
    checker.extended_interfaces_["d"] = {"e"};
    checker.extended_interfaces_["e"] = {"d", "a"};

    EXPECT_TRUE(checker.HasDefaultMethods("a"));
    EXPECT_FALSE(checker.HasDefaultMethods("b"));
    EXPECT_TRUE(checker.HasDefaultMethods("c"));  // Transitivly through a
    EXPECT_TRUE(checker.HasDefaultMethods("d"));  // Transitivly through a
    EXPECT_FALSE(checker.error_);
  }

  static void TestOutputEntry() {
    bool checkedA = false;
    Java8DesugarDepsChecker checker(
        [&checkedA](const std::string &binary_name) {
          checkedA = true;
          return binary_name == "a$$CC.class";
        },
        /*verbose=*/false);
    checker.has_default_methods_["a"] = true;
    checker.extended_interfaces_["b"] = {"c", "d"};
    checker.extended_interfaces_["c"] = {"e"};
    checker.needed_deps_["a$$CC.class"] = "f";
    checker.missing_interfaces_["b"] = "g";
    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(checkedA);

    // Make sure we checked b and its extended interfaces for default methods
    EXPECT_FALSE(checker.has_default_methods_.at("b"));  // should be cached
    EXPECT_FALSE(checker.has_default_methods_.at("c"));  // should be cached
    EXPECT_FALSE(checker.has_default_methods_.at("d"));  // should be cached
    EXPECT_FALSE(checker.has_default_methods_.at("e"));  // should be cached
    EXPECT_FALSE(checker.error_);
  }

  static void TestNeededDepMissing() {
    Java8DesugarDepsChecker checker([](const std::string &) { return false; },
                                    /*verbose=*/false,
                                    /*fail_on_error=*/false);
    checker.needed_deps_["a$$CC.class"] = "b";
    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(checker.error_);
  }

  static void TestMissedDefaultMethods() {
    Java8DesugarDepsChecker checker([](const std::string &) { return true; },
                                    /*verbose=*/false,
                                    /*fail_on_error=*/false);
    checker.has_default_methods_["b"] = true;
    checker.extended_interfaces_["a"] = {"b", "a"};
    checker.missing_interfaces_["a"] = "g";
    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(checker.error_);
  }
};

TEST_F(Java8DesugarDepsCheckerTest, HasDefaultMethods) {
  TestHasDefaultMethods();
}

TEST_F(Java8DesugarDepsCheckerTest, OutputEntry) {
  TestOutputEntry();
}

TEST_F(Java8DesugarDepsCheckerTest, NeededDepMissing) {
  TestNeededDepMissing();
}

TEST_F(Java8DesugarDepsCheckerTest, MissingDefaultMethods) {
  TestMissedDefaultMethods();
}
