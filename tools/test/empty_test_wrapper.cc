// Copyright 2018 The Bazel Authors. All rights reserved.
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

// Empty implementation of the test wrapper.
//
// As of 2018-08-10, every platform uses //tools/test:test-setup.sh as the test
// wrapper, and we are working on introducing a C++ test wrapper for Windows.
// See
// https://github.com/laszlocsomor/proposals/blob/win-test-runner/designs/2018-07-18-windows-native-test-runner.md

#include <stdio.h>

int main(int, char**) {
  fprintf(stderr,
          __FILE__ ": The C++ test wrapper is not used on this platform.\n");
  return 1;
}
