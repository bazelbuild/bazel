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

#include "src/main/tools/process-wrapper-options.h"

#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <vector>

#include "src/main/tools/logging.h"

struct Options opt;

// Print out a usage error. argc and argv are the argument counter and vector,
// fmt is a format, string for the error message to print.
static void Usage(char *program_name, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\nUsage: %s -- command arg1 @args\n", program_name);
  fprintf(
      stderr,
      "\nPossible arguments:\n"
      "  -t/--timeout <timeout>  timeout after which the child process will be "
      "terminated with SIGTERM\n"
      "  -k/--kill_delay <timeout>  in case timeout occurs, how long to wait "
      "before killing the child with SIGKILL\n"
      "  -o/--stdout <file>  redirect stdout to a file\n"
      "  -e/--stderr <file>  redirect stderr to a file\n"
      "  -s/--stats <file>  if set, write stats in protobuf format to a file\n"
      "  -d/--debug  if set, debug info will be printed\n"
      "  --  command to run inside sandbox, followed by arguments\n");
  exit(EXIT_FAILURE);
}

// Parses command line flags from an argv array and puts the results into the
// global `opt` struct.
static void ParseCommandLine(const std::vector<char *> &args) {
  static struct option long_options[] = {
      {"timeout", required_argument, 0, 't'},
      {"kill_delay", required_argument, 0, 'k'},
      {"stdout", required_argument, 0, 'o'},
      {"stderr", required_argument, 0, 'e'},
      {"stats", required_argument, 0, 's'},
      {"debug", no_argument, 0, 'd'},
      {0, 0, 0, 0}};
  extern char *optarg;
  extern int optind, optopt;
  int c;

  while ((c = getopt_long(args.size(), args.data(), "+:t:k:o:e:s:d",
                          long_options, nullptr)) != -1) {
    switch (c) {
      case 't':
        if (sscanf(optarg, "%lf", &opt.timeout_secs) != 1) {
          Usage(args.front(), "Invalid timeout (-t) value: %s", optarg);
        }
        break;
      case 'k':
        if (sscanf(optarg, "%lf", &opt.kill_delay_secs) != 1) {
          Usage(args.front(), "Invalid kill delay (-k) value: %s", optarg);
        }
        break;
      case 'o':
        if (opt.stdout_path.empty()) {
          opt.stdout_path.assign(optarg);
        } else {
          Usage(args.front(),
                "Cannot redirect stdout (-o) to more than one destination.");
        }
        break;
      case 'e':
        if (opt.stderr_path.empty()) {
          opt.stderr_path.assign(optarg);
        } else {
          Usage(args.front(),
                "Cannot redirect stderr (-e) to more than one destination.");
        }
        break;
      case 's':
        if (opt.stats_path.empty()) {
          opt.stats_path.assign(optarg);
        } else {
          Usage(args.front(),
                "Cannot write stats (-s) to more than one destination.");
        }
        break;
      case 'd':
        opt.debug = true;
        break;
      case '?':
        Usage(args.front(), "Unrecognized argument: -%c (%d)", optopt, optind);
        break;
      case ':':
        Usage(args.front(), "Flag -%c requires an argument", optopt);
        break;
    }
  }

  if (optind < static_cast<int>(args.size())) {
    opt.args.assign(args.begin() + optind, args.end());
  }
}

void ParseOptions(int argc, char *argv[]) {
  std::vector<char *> args(argv, argv + argc);

  ParseCommandLine(args);

  if (opt.args.empty()) {
    Usage(args.front(), "No command specified.");
  }

  // argv[] passed to execve() must be a null-terminated array.
  opt.args.push_back(nullptr);
}
