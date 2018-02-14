// Copyright 2016 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_DIAG_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_DIAG_H_ 1

/*
 * Various useful diagnostics functions from Linux err.h file, wrapped
 * for portability.
 */
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)

#include <err.h>
#define diag_err(...) err(__VA_ARGS__)
#define diag_errx(...) errx(__VA_ARGS__)
#define diag_warn(...) warn(__VA_ARGS__)
#define diag_warnx(...) warnx(__VA_ARGS__)

#elif defined(COMPILER_MSVC)

#include <stdio.h>
#include <string.h>
#include <windows.h>
#define _diag_msg(prefix, msg, ...) \
  { fprintf(stderr, prefix " [" __FILE__ ":%d]" msg, __LINE__, __VA_ARGS__); }
#define _diag_msgx(eval, prefix, msg, ...) \
  {                                        \
    _diag_msg(prefix, msg, __VA_ARGS__);   \
    ::ExitProcess(eval);                   \
  }
#define diag_err(eval, fmt, ...) _diag_msgx(eval, "ERROR", fmt, __VA_ARGS__)
#define diag_errx(eval, ...) _diag_msgx(eval, "ERROR", "", __VA_ARGS__)
#define diag_warn(...) _diag_msg("WARNING", __VA_ARGS__)
#define diag_warnx(...) _diag_msg("WARNING")

#else
#error Unknown platform
#endif

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_DIAG_H_
