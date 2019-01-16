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

#include <stdio.h>
#include <stdint.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "ERROR(%s:%d): usage %s <path>\n", __FILE__, __LINE__, argv[0]);
    return 1;
  }
  FILE* f = fopen(argv[1], "rt");
  if (f == NULL) {
    fprintf(stderr, "ERROR(%s:%d): cannot open \"%s\"\n", __FILE__, __LINE__, argv[1]);
    return 1;
  }
  static constexpr size_t kBufSize = 0x10000;
  uint8_t buf[kBufSize];
  size_t read = 0;
  while ((read = fread(buf, 1, kBufSize, f)) > 0) {
    fwrite(buf, 1, read, stdout);
  }
  fclose(f);
  return 0;
}
