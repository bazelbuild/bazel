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

#ifndef THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_LIB_H_
#define THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_LIB_H_

#include <string>
#include <vector>

namespace proguard_allowlister {

class ProguardConfigValidator {
 public:
  ProguardConfigValidator(std::string config_path, std::string outconfig_path);
  virtual ~ProguardConfigValidator() = default;

  bool ValidateAndWriteOutput() { return ValidateAndWriteOutput(nullptr); }
  virtual bool ValidateAndWriteOutput(std::string* error);
  virtual std::vector<std::string> Validate(const std::string& config) const;
  virtual bool ValidateArg(const std::string& arg) const;
  bool ValidateConfigFileExtension() const {
    return ValidateConfigFileExtension(nullptr);
  }
  virtual bool ValidateConfigFileExtension(std::string* error) const;

  const std::string& config_path() const { return config_path_; }
  const std::string& outconfig_path() const { return outconfig_path_; }

  static std::string StripComments(const std::string& config);
  static std::vector<std::string> SplitArgs(const std::string& config);
  static bool ParseCommandLine(int argc, char** argv, std::string* path,
                               std::string* output);

 protected:
  std::string config_path_;
  std::string outconfig_path_;
};

}  // namespace proguard_allowlister

#endif  // THIRD_PARTY_BAZEL_TOOLS_JDK_PROGUARD_ALLOWLISTER_LIB_H_
