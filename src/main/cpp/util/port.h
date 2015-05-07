// Copyright 2014 Google Inc. All rights reserved.
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
#ifndef DEVTOOLS_BLAZE_MAIN_UTIL_PORT_H_
#define DEVTOOLS_BLAZE_MAIN_UTIL_PORT_H_

#include <stddef.h>  // For size_t

// GCC-specific features
#if (defined(COMPILER_GCC3) || defined(__APPLE__)) && !defined(SWIG)

//
// Tell the compiler to do printf format string checking if the
// compiler supports it; see the 'format' attribute in
// <http://gcc.gnu.org/onlinedocs/gcc-4.3.0/gcc/Function-Attributes.html>.
//
// N.B.: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
//
#define PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__ \
    (__printf__, string_index, first_to_check)))

//
// Tell the compiler that a given function never returns
//
#define ATTRIBUTE_NORETURN __attribute__((noreturn))
#define ATTRIBUTE_UNUSED __attribute__ ((unused))

#else  // Not GCC

#define PRINTF_ATTRIBUTE(string_index, first_to_check)
#define ATTRIBUTE_NORETURN
#define ATTRIBUTE_UNUSED

#endif  // GCC

// Linux I/O priorities support is available only in later versions of glibc.
// Therefore, we include some of the needed definitions here.  May need to
// be removed once we switch to a new version of glibc
// (As of 10/24/08 it is unclear when glibc support will become available.)
enum IOPriorityClass {
  // No I/O priority value has yet been set.  The kernel may assign I/O
  // priority based on the process nice value.
  IOPRIO_CLASS_NONE,

  // Real-time, highest priority. Given first access to the disk at
  // every opportunity. Use with care: one such process can STARVE
  // THE ENTIRE SYSTEM. Has 8 priority levels (0-7).
  IOPRIO_CLASS_RT,

  // Best-effort, default for any process. Has 8 priority levels (0-7).
  IOPRIO_CLASS_BE,

  // Idle, lowest priority. Processes running at this level only get
  // I/O time when no one else needs the disk, and MAY BECOME
  // STARVED if higher priority processes are constantly accessing
  // the disk.  With the "anticipatory" I/O scheduler, mapped to
  // IOPRIO_CLASS_BE, level 3.
  IOPRIO_CLASS_IDLE,
};

enum {
  IOPRIO_WHO_PROCESS = 1,
  IOPRIO_WHO_PGRP,
  IOPRIO_WHO_USER,
};

#ifndef IOPRIO_CLASS_SHIFT
#define IOPRIO_CLASS_SHIFT 13
#endif

#ifndef IOPRIO_PRIO_VALUE
#define IOPRIO_PRIO_VALUE(class, data) (((class) << IOPRIO_CLASS_SHIFT) | data)
#endif

namespace blaze_util {

int sys_ioprio_set(int which, int who, int ioprio);

}  // namespace blaze_util

// The arraysize(arr) macro returns the # of elements in an array arr.
// The expression is a compile-time constant, and therefore can be
// used in defining new arrays, for example.  If you use arraysize on
// a pointer by mistake, you will get a compile-time error.

// This template function declaration is used in defining arraysize.
// Note that the function doesn't need an implementation, as we only
// use its type.
template <typename T, size_t N>
char (&ArraySizeHelper(T (&array)[N]))[N];

// That gcc wants both of these prototypes seems mysterious. VC, for
// its part, can't decide which to use (another mystery). Matching of
// template overloads: the final frontier.
template <typename T, size_t N>
char (&ArraySizeHelper(const T (&array)[N]))[N];

#define arraysize(array) (sizeof(ArraySizeHelper(array)))

#endif  // DEVTOOLS_BLAZE_MAIN_UTIL_PORT_H_
