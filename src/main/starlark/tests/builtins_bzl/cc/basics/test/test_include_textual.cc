// Copyright 2025 The Bazel Authors. All rights reserved.
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

// NOLINTBEGIN(build/include)
// That the following two includes don't have a directory is the point of this
// test.
#include "lib.h"
#include "not_nested.h"
// NOLINTEND(build/include)

int main() {
  if (foo() != 42) {
    return 1;
  }
  if (NOT_NESTED != 42) {
    return 2;
  }
  return 0;
}
