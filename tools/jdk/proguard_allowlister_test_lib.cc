// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include "tools/jdk/proguard_allowlister_test_lib.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "tools/jdk/proguard_allowlister_lib.h"

namespace proguard_allowlister {

std::string GetTestTmpDir() {
  const char* tmpdir = std::getenv("TEST_TMPDIR");
  if (tmpdir && tmpdir[0] != '\0') {
    return tmpdir;
  }
  return "/tmp";
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Cannot read file: " << path << "\n";
    std::abort();
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

void WriteFile(const std::string& path, const std::string& content) {
  std::ofstream file(path, std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "Cannot write file: " << path << "\n";
    std::abort();
  }
  file << content;
}

std::string ResolveRunfile(const std::string& rlocation_path) {
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  if (!runfiles) {
    std::cerr << "Failed to initialize runfiles: " << error << "\n";
    std::abort();
  }
  std::string path = runfiles->Rlocation(rlocation_path);
  if (path.empty()) {
    std::cerr << "Could not find runfile: " << rlocation_path << "\n";
    std::abort();
  }
  return path;
}

std::string ResolveRunfileFromEnv(const char* env_var) {
  const char* rlocation = std::getenv(env_var);
  if (!rlocation || rlocation[0] == '\0') {
    std::cerr << "Environment variable '" << env_var << "' not set.\n";
    std::abort();
  }
  return ResolveRunfile(rlocation);
}

ProguardConfigValidatorTest::ProguardConfigValidatorTest(
    std::string test_input_path)
    : test_input_path_(std::move(test_input_path)) {}

std::unique_ptr<ProguardConfigValidator>
ProguardConfigValidatorTest::CreateValidator(
    const std::string& input_path, const std::string& output_path) const {
  return std::make_unique<ProguardConfigValidator>(input_path, output_path);
}

void ProguardConfigValidatorTest::TestValidConfig() const {
  std::string tmpdir = GetTestTmpDir();
  std::string output_path = tmpdir + "/proguard_allowlister_test_output.pgcfg";

  std::unique_ptr<ProguardConfigValidator> validator =
      CreateValidator(test_input_path_, output_path);
  std::string error;
  if (!validator->ValidateAndWriteOutput(&error)) {
    std::cerr << "Validation failed unexpectedly: " << error << "\n";
    std::abort();
  }

  std::string output = ReadFile(output_path);
  std::string expected_prefix = "# Merged from " + test_input_path_;
  if (output.find(expected_prefix) == std::string::npos) {
    std::cerr << "Expected '" << expected_prefix << "' in output, but got:\n"
              << output << "\n";
    std::abort();
  }
}

void ProguardConfigValidatorTest::TestValidConfigString(
    const std::string& config) const {
  std::string tmpdir = GetTestTmpDir();
  std::string input_path = tmpdir + "/proguard_allowlister_test_input.pgcfg";
  std::string output_path = tmpdir + "/proguard_allowlister_test_output.pgcfg";

  WriteFile(input_path, config);

  std::unique_ptr<ProguardConfigValidator> validator =
      CreateValidator(input_path, output_path);
  std::string error;
  if (!validator->ValidateAndWriteOutput(&error)) {
    std::cerr << "Validation failed unexpectedly: " << error << "\n";
    std::abort();
  }

  std::string output = ReadFile(output_path);
  std::string expected_prefix = "# Merged from " + input_path;
  if (output.find(expected_prefix) == std::string::npos) {
    std::cerr << "Expected '" << expected_prefix << "' in output, but got:\n"
              << output << "\n";
    std::abort();
  }
}

void ProguardConfigValidatorTest::TestValidConfigWithLeadingWhitespace() const {
  TestValidConfigString(
      " # Leading space before comment\n"
      " -if @com.example.Serializable class **\n"
      " -keep, allowshrinking class <1>\n"
      "  -keepclassmembers class * {\n"
      "    static int x;\n"
      "  }\n"
      "\t-dontwarn com.example.**\n");
}

void ProguardConfigValidatorTest::TestInvalidConfig(
    const std::vector<std::string>& invalid_args,
    const std::string& config) const {
  std::string tmpdir = GetTestTmpDir();
  std::string input_path = tmpdir + "/proguard_allowlister_test_input.pgcfg";
  std::string output_path = tmpdir + "/proguard_allowlister_test_output.pgcfg";

  WriteFile(input_path, config);

  std::unique_ptr<ProguardConfigValidator> validator =
      CreateValidator(input_path, output_path);
  std::string error;
  bool success = validator->ValidateAndWriteOutput(&error);
  if (success) {
    std::cerr << "Expected failure for config:\n" << config << "\n";
    std::abort();
  }
  for (const auto& invalid_arg : invalid_args) {
    if (error.find(invalid_arg) == std::string::npos) {
      std::cerr << "Expected '" << invalid_arg
                << "' in error message: " << error << "\n";
      std::abort();
    }
  }
}

void ProguardConfigValidatorTest::TestInvalidConfigWithLeadingWhitespace()
    const {
  TestInvalidConfig({"-dontnote"}, "  -dontnote\n");
  TestInvalidConfig(
      {"-optimizations"},
      "  # We don't want libraries disabling global optimizations.\n"
      "  -optimizations !class/merging/*,!code/allocation/variable\n");
}

void ProguardConfigValidatorTest::TestInvalidNoteConfig() const {
  TestInvalidConfig({"-dontnote"},
                    "# We don't want libraries disabling notes globally.\n"
                    "-dontnote\n");
}

void ProguardConfigValidatorTest::TestInvalidWarnConfig() const {
  TestInvalidConfig({"-dontwarn"},
                    "# We don't want libraries disabling warnings globally.\n"
                    "-dontwarn\n");
}

void ProguardConfigValidatorTest::TestInvalidOptimizationConfig() const {
  TestInvalidConfig(
      {"-optimizations"},
      "# We don't want libraries disabling global optimizations.\n"
      "-optimizations !class/merging/*,!code/allocation/variable\n");
}

void ProguardConfigValidatorTest::TestMultipleInvalidArgs() const {
  TestInvalidConfig(
      {"-optimizations", "-dontnote"},
      "# We don't want libraries disabling global optimizations.\n"
      "-optimizations !class/merging/*,!code/allocation/variable\n"
      "-dontnote\n");
}

void ProguardConfigValidatorTest::RunAllTests() const {
  TestValidConfig();
  TestValidConfigWithLeadingWhitespace();
  TestInvalidNoteConfig();
  TestInvalidWarnConfig();
  TestInvalidOptimizationConfig();
  TestMultipleInvalidArgs();
  TestInvalidConfigWithLeadingWhitespace();
}

}  // namespace proguard_allowlister
