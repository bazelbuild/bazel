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

#include "tools/jdk/proguard_allowlister_test_lib.h"

int main() {
  std::string test_input =
      proguard_allowlister::ResolveRunfileFromEnv("TEST_INPUT");
  proguard_allowlister::ProguardConfigValidatorTest test(test_input);
  test.RunAllTests();
  std::cout << "All proguard_allowlister tests passed.\n";
  return 0;
}
