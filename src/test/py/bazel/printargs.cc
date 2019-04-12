// Copyright 2019 The Bazel Authors. All rights reserved.
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

// Prints the command line arguments, one on each line, as arg=(<ARG>).
// This program aids testing the flags Bazel passes on the command line.
#include <stdio.h>
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    printf("arg=(%s)\n", argv[i]);
  }
  return 0;
}
