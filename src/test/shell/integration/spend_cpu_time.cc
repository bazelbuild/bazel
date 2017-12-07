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
#include <sys/types.h>
#include <sys/resource.h>
#include <err.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Marked as volatile to force the C compiler to calculate it.
volatile uint64_t volatile_counter;

static void WasteUserTime() {
  volatile_counter = 0;
  while (true) {
    volatile_counter++;
    if (volatile_counter == 10000000) {
      break;
    }
  }
}

static void WasteSystemTime() {
  char current_dir_path[MAXPATHLEN];
  if (getcwd(current_dir_path, sizeof(current_dir_path)) == NULL) {
    err(EXIT_FAILURE, "getcwd() failed");
  }

  volatile_counter = 0;
  while (true) {
    // Arbitrary syscall to waste system time.
    if (chdir(current_dir_path) != 0) {
      err(EXIT_FAILURE, "chdir() failed");
    }
    volatile_counter++;
    if (volatile_counter == 100000) {
      break;
    }
  }
}

static void GetResourceUsage(struct rusage *rusage) {
  if (getrusage(RUSAGE_SELF, rusage) != 0) {
    err(EXIT_FAILURE, "getrusage() failed");
  }
}

static int GetUsedUserTimeSeconds() {
  struct rusage my_rusage;
  GetResourceUsage(&my_rusage);
  return my_rusage.ru_utime.tv_sec;
}

static int GetUsedSystemTimeSeconds() {
  struct rusage my_rusage;
  GetResourceUsage(&my_rusage);
  return my_rusage.ru_stime.tv_sec;
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
    int spent_system_time_seconds = 0;
    while (spent_system_time_seconds < requested_system_time_seconds) {
      WasteSystemTime();
      spent_system_time_seconds = GetUsedSystemTimeSeconds();
    }
  }

  // Waste user time if we haven't already wasted enough.
  if (requested_user_time_seconds > 0) {
    int spent_user_time_seconds = 0;
    while (spent_user_time_seconds < requested_user_time_seconds) {
      WasteUserTime();
      spent_user_time_seconds = GetUsedUserTimeSeconds();
    }
  }

  int spent_user_time_seconds = GetUsedUserTimeSeconds();
  int spent_system_time_seconds = GetUsedSystemTimeSeconds();
  printf("Total user time wasted: %d seconds\n", spent_user_time_seconds);
  printf("Total system time wasted: %d seconds\n", spent_system_time_seconds);

  exit(EXIT_SUCCESS);
}
