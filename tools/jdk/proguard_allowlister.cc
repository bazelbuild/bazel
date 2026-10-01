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

#include <iostream>
#include <string>

#include "tools/jdk/proguard_allowlister_lib.h"

int main(int argc, char** argv) {
  std::string path;
  std::string output;
  if (!proguard_allowlister::ProguardConfigValidator::ParseCommandLine(
          argc, argv, &path, &output)) {
    std::cerr << "Usage: " << argv[0] << " --path <path> --output <output>\n";
    return 1;
  }

  proguard_allowlister::ProguardConfigValidator validator(path, output);
  std::string error;
  if (!validator.ValidateAndWriteOutput(&error)) {
    std::cerr << error << std::endl;
    return 1;
  }

  return 0;
}
