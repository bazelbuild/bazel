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

#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <err.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Computes the time that passed, in millis, since the previous timestamp.
static uint64_t ElapsedCpuMillisSince(const clock_t before) {
  const clock_t now = clock();
  return 1000 * (now - before) / CLOCKS_PER_SEC;
}

// Computes the time that passed, in millis, since the previous timestamp.
static uint64_t ElapsedWallTimeMillisSince(const struct timeval* before) {
  struct timeval now;
  gettimeofday(&now, NULL);
  return (now.tv_sec * 1000 + now.tv_usec / 1000) -
      (before->tv_sec * 1000 + before->tv_usec / 1000);
}

// Spends CPU time for about the requested number of milliseconds.
//
// This function should not invoke any system calls, but as this is very hard to
// do in a portable way, the number of such invocations should be kept to a
// minimum so that their cost is not noticeable.
//
// This function does not guarantee that the used CPU time is above the given
// millis. The caller needs to check this and, if not yet achieved, call this
// function again with the remainder.
static void WasteUserTime(const uint64_t millis) {
  const clock_t before = clock();
  while (ElapsedCpuMillisSince(before) < millis) {
    // The body of this loop is supposed to consume enough CPU time to make the
    // actual calls to clock() insignificant. This means that if this loop gets
    // optimized, or if the CPU becomes fast enough to run this "too fast", this
    // function may consume more system time than user time and cause tests to
    // fail.
    volatile uint64_t counter = 0;
    while (counter < 1000000) {
      counter++;
    }
  }
}

// Spends system time for about the requested number of milliseconds.
//
// This function does not guarantee that the used system time is above the given
// millis. The caller needs to check this and, if not yet achieved, call this
// function again with the remainder.
static void WasteSystemTime(const uint64_t millis) {
  char current_dir_path[MAXPATHLEN];
  if (getcwd(current_dir_path, sizeof(current_dir_path)) == NULL) {
    err(EXIT_FAILURE, "getcwd() failed");
  }

  struct timeval before;
  gettimeofday(&before, NULL);
  while (ElapsedWallTimeMillisSince(&before) < millis) {
    // Arbitrary syscall to waste system time.
    if (chdir(current_dir_path) != 0) {
      err(EXIT_FAILURE, "chdir() failed");
    }
  }
}

static void GetResourceUsage(struct rusage *rusage) {
  if (getrusage(RUSAGE_SELF, rusage) != 0) {
    err(EXIT_FAILURE, "getrusage() failed");
  }
}

static uint64_t GetUsedUserTimeMillis() {
  struct rusage my_rusage;
  GetResourceUsage(&my_rusage);
  return my_rusage.ru_utime.tv_sec * 1000 + my_rusage.ru_utime.tv_usec / 1000;
}

static uint64_t GetUsedSystemTimeMillis() {
  struct rusage my_rusage;
  GetResourceUsage(&my_rusage);
  return my_rusage.ru_stime.tv_sec * 1000 + my_rusage.ru_stime.tv_usec / 1000;
}

// Subtracts subtrahend from minuend, or returns zero if the subtrahend is
// larger than the minuend.
static uint64_t SubtractOrZero(const uint64_t minuend,
                               const uint64_t subtrahend) {
  if (subtrahend > minuend) {
    return 0;
  } else {
    return minuend - subtrahend;
  }
}

// This program just wastes (at least) the desired amount of CPU time, by
// checking its own resource usage (rusage) while running.
int main(int argc, char **argv) {
  // Parse command-line arguments.
  const char *progname = argv[0] ? argv[0] : "spend_cpu_time";
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <user_time_seconds> <system_time_seconds>\n",
            progname);
    exit(EXIT_FAILURE);
  }

  int requested_user_time_seconds \
      = atoi(argv[1]);  // NOLINT(runtime/deprecated_fn)
  int requested_system_time_seconds \
      = atoi(argv[2]);  // NOLINT(runtime/deprecated_fn)

  // Waste system time first, because this also wastes some user time.
  if (requested_system_time_seconds > 0) {
    const uint64_t requested_millis = requested_system_time_seconds * 1000;
    for (;;) {
      const uint64_t remaining_millis =
          SubtractOrZero(requested_millis, GetUsedSystemTimeMillis());
      if (remaining_millis == 0) {
        break;
      }
      WasteSystemTime(remaining_millis);
    }
  }

  // Waste user time if we haven't already wasted enough.
  if (requested_user_time_seconds > 0) {
    const uint64_t requested_millis = requested_user_time_seconds * 1000;
    for (;;) {
      const uint64_t remaining_millis =
          SubtractOrZero(requested_millis, GetUsedUserTimeMillis());
      if (remaining_millis == 0) {
        break;
      }
      WasteUserTime(remaining_millis);
    }
  }

  printf("Total user time wasted: %" PRIu64 " ms\n",
         GetUsedUserTimeMillis());
  printf("Total system time wasted: %" PRIu64 " ms\n",
         GetUsedSystemTimeMillis());

  exit(EXIT_SUCCESS);
}
