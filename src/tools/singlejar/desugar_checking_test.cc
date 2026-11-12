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

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/tools/singlejar/combiners.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/zip_headers.h"
#include "src/tools/singlejar/zlib_interface.h"
#include "googletest/include/gtest/gtest.h"

// A test fixture is used because friend access to class under test is needed.
// Tests are instance methods to avoid gUnit dep in .h file.
class Java8DesugarDepsCheckerTest : public ::testing::Test {
 protected:
  static bool MergeContents(Java8DesugarDepsChecker* checker,
                            const std::string& contents) {
    Concatenator concatenator("META-INF/desugar_deps");
    concatenator.Append(contents.data(), contents.size());
    LH* lh =
        reinterpret_cast<LH*>(concatenator.OutputEntry(/*compress=*/false));
    uint8_t cdh_bytes[256] = {};
    CDH* cdh = reinterpret_cast<CDH*>(cdh_bytes);
    cdh->signature();
    cdh->bit_flag(0);
    bool ok = checker->Merge(cdh, lh);
    free(lh);
    return ok;
  }

  static void TestHasDefaultMethods() {
    Java8DesugarDepsChecker checker([](const std::string&) { return false; },
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
        [&checkedA](const std::string& binary_name) {
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
    Java8DesugarDepsChecker checker([](const std::string&) { return false; },
                                    /*verbose=*/false,
                                    /*fail_on_error=*/false);
    checker.needed_deps_["a$$CC.class"] = "b";
    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(checker.error_);
  }

  static void TestMissedDefaultMethods() {
    Java8DesugarDepsChecker checker([](const std::string&) { return true; },
                                    /*verbose=*/false,
                                    /*fail_on_error=*/false);
    checker.has_default_methods_["b"] = true;
    checker.extended_interfaces_["a"] = {"b", "a"};
    checker.missing_interfaces_["a"] = "g";
    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(checker.error_);
  }

  static void TestMergeJson() {
    Java8DesugarDepsChecker checker(
        [](const std::string& binary_name) {
          return binary_name == "a/I$$CC.class";
        },
        /*verbose=*/false, /*fail_on_error=*/false, DesugarDepsFormat::kJson);
    const std::string json1 = R"({
      "assume_present": [{"origin": "a/A", "target": "a/I$$CC"}],
      "missing_interface": [{"origin": "a/C", "target": "b/K"}],
      "interface_with_supertypes": [
        {"origin": "b/K", "extended_interface": ["b/L", "b/M"]}
      ],
      "interface_with_companion": [
        {"origin": "a/I", "num_default_methods": 2},
        {"origin": "b/L", "num_default_methods": 0}
      ]
    })";
    EXPECT_TRUE(MergeContents(&checker, json1));
    EXPECT_EQ("a/A", checker.needed_deps_.at("a/I$$CC.class"));
    EXPECT_EQ("a/C", checker.missing_interfaces_.at("b/K"));
    EXPECT_EQ((std::vector<std::string>{"b/L", "b/M"}),
              checker.extended_interfaces_.at("b/K"));
    EXPECT_TRUE(checker.has_default_methods_.at("a/I"));
    EXPECT_EQ(0u, checker.has_default_methods_.count("b/L"));

    EXPECT_EQ(nullptr, checker.OutputEntry(/*compress=*/true));
    EXPECT_FALSE(checker.error_);

    // Now merge a second JSON entry that marks b/M as having default methods;
    // since b/K extends b/M and b/K is missing, OutputEntry should flag an
    // error.
    Java8DesugarDepsChecker failing_checker(
        [](const std::string&) { return true; },
        /*verbose=*/false, /*fail_on_error=*/false, DesugarDepsFormat::kJson);
    EXPECT_TRUE(MergeContents(&failing_checker, json1));
    const std::string json2 = R"({
      "interface_with_companion": [
        {"origin": "b/M", "num_default_methods": 1}
      ]
    })";
    EXPECT_TRUE(MergeContents(&failing_checker, json2));
    EXPECT_EQ(nullptr, failing_checker.OutputEntry(/*compress=*/true));
    EXPECT_TRUE(failing_checker.error_);
  }

  static void TestMergeJsonInvalid() {
    Java8DesugarDepsChecker checker([](const std::string&) { return true; },
                                    /*verbose=*/false, /*fail_on_error=*/false,
                                    DesugarDepsFormat::kJson);
    const std::string bad_json = "{not valid json";
    EXPECT_FALSE(MergeContents(&checker, bad_json));
    EXPECT_TRUE(checker.error_);

    Java8DesugarDepsChecker fatal_checker(
        [](const std::string&) { return true; },
        /*verbose=*/false, /*fail_on_error=*/true, DesugarDepsFormat::kJson);
    EXPECT_EXIT(MergeContents(&fatal_checker, bad_json),
                testing::ExitedWithCode(2),
                "META-INF/desugar_deps: unable to parse JSON");
  }
};

TEST_F(Java8DesugarDepsCheckerTest, HasDefaultMethods) {
  TestHasDefaultMethods();
}

TEST_F(Java8DesugarDepsCheckerTest, OutputEntry) { TestOutputEntry(); }

TEST_F(Java8DesugarDepsCheckerTest, NeededDepMissing) {
  TestNeededDepMissing();
}

TEST_F(Java8DesugarDepsCheckerTest, MissingDefaultMethods) {
  TestMissedDefaultMethods();
}

TEST_F(Java8DesugarDepsCheckerTest, MergeJson) { TestMergeJson(); }

TEST_F(Java8DesugarDepsCheckerTest, MergeJsonInvalid) {
  TestMergeJsonInvalid();
}
