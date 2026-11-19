// Copyright 2015 The Bazel Authors. All rights reserved.
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

// A wrapper around the futimens system call which mutates both the 'last
// accessed' and 'last modified' times of a file (we both for the sake
// of simplicity). 'futimens' is used so we have nanosecond precision.
//
// Intended to be used for tests only.

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>

int main(int argc, char** argv) {
  std::string file;
  int64_t seconds = 0;
  int64_t nanoseconds = 0;
  bool has_file = false;
  bool has_seconds = false;
  bool has_nanoseconds = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--file=", 0) == 0) {
      file = arg.substr(7);
      has_file = true;
    } else if (arg.rfind("--seconds=", 0) == 0) {
      seconds = std::stoll(arg.substr(10));
      has_seconds = true;
    } else if (arg.rfind("--nanoseconds=", 0) == 0) {
      nanoseconds = std::stoll(arg.substr(14));
      has_nanoseconds = true;
    } else {
      fprintf(stderr, "Unknown flag: %s\n", arg.c_str());
      return 1;
    }
  }

  if (!has_file || !has_seconds || !has_nanoseconds) {
    fprintf(stderr,
            "Usage: %s --file=<path> --seconds=<sec> --nanoseconds=<nsec>\n",
            argv[0]);
    return 1;
  }

  timespec times[2];
  timespec time;
  time.tv_sec = seconds;
  time.tv_nsec = nanoseconds;
  // Last accessed time.
  times[0] = time;
  // Last modified time.
  times[1] = time;
  FILE* fp = fopen(file.c_str(), "r");
  if (fp == nullptr) {
    fprintf(stderr, "fopen(%s) failed: %s\n", file.c_str(), strerror(errno));
    return 1;
  }
  if (futimens(fileno(fp), times) != 0) {
    fprintf(stderr, "futimens failed: %s\n", strerror(errno));
    fclose(fp);
    return 1;
  }
  if (fclose(fp) != 0) {
    fprintf(stderr, "fclose failed: %s\n", strerror(errno));
    return 1;
  }
  return 0;
}
