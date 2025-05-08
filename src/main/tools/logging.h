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

#ifndef SRC_MAIN_TOOLS_LOGGING_H_
#define SRC_MAIN_TOOLS_LOGGING_H_

// See https://stackoverflow.com/a/8132440 .
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// see
// http://stackoverflow.com/questions/5641427/how-to-make-preprocessor-generate-a-string-for-line-keyword
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)

#define DIE(...)                                                \
  {                                                             \
    fprintf(stderr, __FILE__ ":" S__LINE__ ": \"" __VA_ARGS__); \
    fprintf(stderr, "\": ");                                    \
    perror(nullptr);                                            \
    exit(EXIT_FAILURE);                                         \
  }

#define PRINT_DEBUG(fmt, ...)                                       \
  do {                                                              \
    if (global_debug) {                                             \
      struct timespec ts;                                           \
      clock_gettime(CLOCK_REALTIME, &ts);                           \
                                                                    \
      fprintf(global_debug, "%" PRId64 ".%09ld: %s:%d: " fmt "\n",  \
              ((int64_t)ts.tv_sec), ts.tv_nsec, __FILE__, __LINE__, \
              ##__VA_ARGS__);                                       \
                                                                    \
      /* Minimize probability of losing output if we're killed. */  \
      fflush(global_debug);                                         \
    }                                                               \
  } while (0)

// The file PRINT_DEBUG writes to.
// Initialized to null (no debug logging) in logging.cc.
// Set in the linux-sandbox.cc main() if the -D command line option is set.
extern FILE* global_debug;

#endif  // SRC_MAIN_TOOLS_LOGGING_H_
