// Copyright 2022 The Bazel Authors. All rights reserved.
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

#include <fstream>
#include <string>

//  This is a replacement for
//  third_party/bazel/src/main/java/com/google/devtools/build/lib/analysis/actions/LauncherFileWriteAction.java
//
//  It takes exactly 3 arguments:
//    1) The path to the actual launcher executable
//    2) The multi-line .params file containing the launcher info data
//    3) The path of the output executable
//
//  The program copies the launcher executable as is to the output, and then
//  appends each line of the launch info as a null-terminated string. At the
//  end, the size of the launch data written is appended as a long value (8
//  bytes).
int main(int argc, char** argv) {
  char* launcher_path = argv[1];
  char* info_params = argv[2];
  char* output_path = argv[3];

  std::ifstream src(launcher_path, std::ios::binary);
  std::ofstream dst(output_path, std::ios::binary);
  dst << src.rdbuf();

  std::ifstream info_file(info_params);
  int64_t bytes = 0;
  std::string line;
  while (std::getline(info_file, line)) {
    dst << line;
    bytes += line.length();
    dst << '\0';
    bytes++;
  }

  dst.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
  return 0;
}
