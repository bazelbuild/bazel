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

#ifndef THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_TEST_LIB_H_
#define THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_TEST_LIB_H_

#include <memory>
#include <string>
#include <vector>

#include "tools/jdk/proguard_allowlister_lib.h"

namespace proguard_allowlister {

std::string GetTestTmpDir();
std::string ReadFile(const std::string& path);
void WriteFile(const std::string& path, const std::string& content);
std::string ResolveRunfile(const std::string& rlocation_path);
std::string ResolveRunfileFromEnv(const char* env_var);

class ProguardConfigValidatorTest {
 public:
  explicit ProguardConfigValidatorTest(std::string test_input_path);
  virtual ~ProguardConfigValidatorTest() = default;

  virtual std::unique_ptr<ProguardConfigValidator> CreateValidator(
      const std::string& input_path, const std::string& output_path) const;

  virtual void TestValidConfig() const;
  void TestValidConfigString(const std::string& config) const;
  void TestValidConfigWithLeadingWhitespace() const;
  void TestInvalidConfig(const std::vector<std::string>& invalid_args,
                         const std::string& config) const;
  void TestInvalidConfigWithLeadingWhitespace() const;
  void TestInvalidNoteConfig() const;
  void TestInvalidWarnConfig() const;
  void TestInvalidOptimizationConfig() const;
  void TestMultipleInvalidArgs() const;

  virtual void RunAllTests() const;

 protected:
  std::string test_input_path_;
};

}  // namespace proguard_allowlister

#endif  // THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_TEST_LIB_H_
