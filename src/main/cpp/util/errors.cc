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

#include "src/main/cpp/util/errors.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace blaze_util {

void die(const int exit_status, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(exit_status);
}

void pdie(const int exit_status, const char *format, ...) {
  const char *errormsg = GetLastErrorString().c_str();
  fprintf(stderr, "Error: ");
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fprintf(stderr, ": %s\n", errormsg);
  exit(exit_status);
}

void PrintError(const char *format, ...) {
  const char *errormsg = GetLastErrorString().c_str();
  fprintf(stderr, "ERROR: ");
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fprintf(stderr, ": %s\n", errormsg);
}

void PrintWarning(const char *format, ...) {
  va_list args;

  va_start(args, format);
  fputs("WARNING: ", stderr);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
  va_end(args);
}

}  // namespace blaze_util
