// Copyright 2018 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_PORT_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_PORT_H_ 1

#ifdef _WIN32
// This macro is required to tell MSVC headers to also define POSIX names
// without "_" prefix (such as "open" for "_open").
#define _CRT_DECLARE_NONSTDC_NAMES 1
#include <fcntl.h>
#include <io.h>
#endif  // _WIN32

#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#if defined(__APPLE__)
typedef off_t off64_t;
#elif defined(_WIN32)
typedef __int64 off64_t;
#elif defined(__OpenBSD__)
typedef int64_t off64_t;
#endif
static_assert(sizeof(off64_t) == 8, "File offset type must be 64-bit");

#ifdef _WIN32

#define F_OK 0

#ifdef _WIN64
// MSVC by default defines stat and related functions to a version with 32-bit
// st_size even for Win64. We want 64-bit st_size instead so that we can handle
// large file.
#undef stat
#undef fstat
#define stat _stat64
#define fstat _fstat64
#endif  // _WIN64

typedef ptrdiff_t ssize_t;

// Various MSVC polyfills for POSIX functions.

inline tm* localtime_r(const time_t* tin, tm* tout) {
  if (!localtime_s(tout, tin)) return tout;

  return nullptr;
}

#endif  // _WIN32

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_PORT_H_
