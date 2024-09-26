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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#include "src/main/cpp/util/path_platform.h"
#endif  // _WIN32

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

#ifdef _WIN32

#define STRING_TYPE std::wstring
#define STRING_FORMAT "%ls"
std::wstring convert_path(char* path) {
  std::string error;
  std::wstring wpath;
  if (!blaze_util::AsAbsoluteWindowsPath(path, &wpath, &error)) {
    fprintf(stderr, "Failed to make absolute path for %s: %s\n", path,
            error.c_str());
    exit(1);
  }
  return wpath;
}

#else  // _WIN32

#define STRING_TYPE std::string
#define STRING_FORMAT "%s"
std::string convert_path(char* path) { return path; }

#endif  // _WIN32

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Expected 3 arguments, got %d\n", argc);
    return 1;
  }

  STRING_TYPE launcher_path = convert_path(argv[1]);
  STRING_TYPE info_params = convert_path(argv[2]);
  STRING_TYPE output_path = convert_path(argv[3]);

  std::ifstream src(launcher_path.c_str(), std::ios::binary);
  if (!src.good()) {
    fprintf(stderr, "Failed to open " STRING_FORMAT ": %s\n",
            launcher_path.c_str(), strerror(errno));
    return 1;
  }
  std::ofstream dst(output_path.c_str(), std::ios::binary);
  if (!dst.good()) {
    fprintf(stderr, "Failed to create " STRING_FORMAT ": %s\n",
            output_path.c_str(), strerror(errno));
    return 1;
  }
  dst << src.rdbuf();

  std::ifstream info_file(info_params.c_str());
  if (!info_file.good()) {
    fprintf(stderr, "Failed to open " STRING_FORMAT ": %s\n",
            info_params.c_str(), strerror(errno));
    return 1;
  }
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
