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

#include <iostream>

int main(int argc, char** argv) {
  // TODO(bazel-team): decide whether we need build-runfiles at all on Windows.
  // Implement this program if so; make sure we don't run it on Windows if not.
  std::cout << "ERROR: build-runfiles is not (yet?) implemented on Windows."
            << std::endl
            << "Called with args:" << std::endl;
  for (int i = 0; i < argc; ++i) {
    std::cout << "argv[" << i << "]=(" << argv[i] << ")" << std::endl;
  }
  return 1;
}
