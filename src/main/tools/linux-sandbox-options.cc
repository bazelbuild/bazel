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

#include "src/main/tools/linux-sandbox-options.h"

#include <errno.h>
#include <sched.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/main/tools/logging.h"
#include "src/main/tools/process-tools.h"

using std::ifstream;
using std::unique_ptr;
using std::vector;

struct Options opt;

// Print out a usage error. argc and argv are the argument counter and vector,
// fmt is a format, string for the error message to print.
static void Usage(char *program_name, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\nUsage: %s -- command arg1 @args\n", program_name);
  fprintf(stderr,
          "\nPossible arguments:\n"
          "  -W <working-dir>  working directory (uses current directory if "
          "not specified)\n"
          "  -T <timeout>  timeout after which the child process will be "
          "terminated with SIGTERM\n"
          "  -t <timeout>  in case timeout occurs, how long to wait before "
          "killing the child with SIGKILL\n"
          "  -i  on receipt of a SIGINT, forward it to the child process as a "
          "SIGTERM first and then as a SIGKILL after the -T timeout\n"
          "  -l <file>  redirect stdout to a file\n"
          "  -L <file>  redirect stderr to a file\n"
          "  -w <file>  make a file or directory writable for the sandboxed "
          "process\n"
          "  -e <dir>  mount an empty tmpfs on a directory\n"
          "  -M/-m <source/target>  directory to mount inside the sandbox\n"
          "    Multiple directories can be specified and each of them will be "
          "mounted readonly.\n"
          "    The -M option specifies which directory to mount, the -m option "
          "specifies where to\n"
          "  -S <file>  if set, write stats in protobuf format to a file\n"
          "  -H  if set, make hostname in the sandbox equal to 'localhost'\n"
          "  -n  if set, create a new network namespace\n"
          "  -N  if set, create a new network namespace with loopback\n"
          "        Only one of -n and -N may be specified.\n"
          "  -R  if set, make the uid/gid be root\n"
          "  -U  if set, make the uid/gid be nobody\n"
          "  -P  if set, make the gid be tty and make /dev/pts writable\n"
          "  -D <debug-file> if set, debug info will be printed to this file\n"
          "  -p  if set, the process is persistent and ignores parent thread "
          "death signals\n"
          "  -C <dir> if set, put all subprocesses inside this cgroup.\n"
          "  -h <sandbox-dir>  if set, chroot to sandbox-dir and only "
          " mount whats been specified with -M/-m for improved hermeticity. "
          " The working-dir should be a folder inside the sandbox-dir\n"
          "  @FILE  read newline-separated arguments from FILE\n"
          "  --  command to run inside sandbox, followed by arguments\n");
  exit(EXIT_FAILURE);
}

static void ValidateIsAbsolutePath(char *path, char *program_name, char flag) {
  if (path[0] != '/') {
    Usage(program_name, "The -%c option must be used with absolute paths only.",
          flag);
  }
}

// Parses command line flags from an argv array and puts the results into an
// Options structure passed in as an argument.
static void ParseCommandLine(unique_ptr<vector<char *>> args) {
  extern char *optarg;
  extern int optind, optopt;
  int c;
  bool source_specified = false;
  while ((c = getopt(args->size(), args->data(),
                     ":W:T:t:il:L:w:e:M:m:S:h:pC:HnNRUPD:")) != -1) {
    if (c != 'M' && c != 'm') source_specified = false;
    switch (c) {
      case 'W':
        if (opt.working_dir.empty()) {
          ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
          opt.working_dir.assign(optarg);
        } else {
          Usage(args->front(),
                "Multiple working directories (-W) specified, expected one.");
        }
        break;
      case 'T':
        if (sscanf(optarg, "%d", &opt.timeout_secs) != 1 ||
            opt.timeout_secs < 0) {
          Usage(args->front(), "Invalid timeout (-T) value: %s", optarg);
        }
        break;
      case 't':
        if (sscanf(optarg, "%d", &opt.kill_delay_secs) != 1 ||
            opt.kill_delay_secs < 0) {
          Usage(args->front(), "Invalid kill delay (-t) value: %s", optarg);
        }
        break;
      case 'i':
        opt.sigint_sends_sigterm = true;
        break;
      case 'p':
        opt.persistent_process = true;
        break;
      case 'l':
        if (opt.stdout_path.empty()) {
          opt.stdout_path.assign(optarg);
        } else {
          Usage(args->front(),
                "Cannot redirect stdout to more than one destination.");
        }
        break;
      case 'L':
        if (opt.stderr_path.empty()) {
          opt.stderr_path.assign(optarg);
        } else {
          Usage(args->front(),
                "Cannot redirect stderr to more than one destination.");
        }
        break;
      case 'w':
        ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
        opt.writable_files.emplace_back(optarg);
        break;
      case 'e':
        ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
        opt.tmpfs_dirs.emplace_back(optarg);
        break;
      case 'M':
        ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
        // Add the current source path to both source and target lists
        opt.bind_mount_sources.emplace_back(optarg);
        opt.bind_mount_targets.emplace_back(optarg);
        source_specified = true;
        break;
      case 'm':
        ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
        if (!source_specified) {
          Usage(args->front(),
                "The -m option must be strictly preceded by an -M option.");
        }
        opt.bind_mount_targets.pop_back();
        opt.bind_mount_targets.emplace_back(optarg);
        source_specified = false;
        break;
      case 'S':
        if (opt.stats_path.empty()) {
          opt.stats_path.assign(optarg);
        } else {
          Usage(args->front(),
                "Cannot write stats to more than one destination.");
        }
        break;
      case 'h':
        opt.hermetic = true;
        if (opt.sandbox_root.empty()) {
          std::string sandbox_root(optarg);
          // Make sure that the sandbox_root path has no trailing slash.
          if (sandbox_root.back() == '/') {
            ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
            opt.sandbox_root.assign(sandbox_root, 0, sandbox_root.length() - 1);
            if (opt.sandbox_root.back() == '/') {
              Usage(args->front(),
                    "Sandbox root path should not have trailing slashes");
            }
          } else {
            opt.sandbox_root.assign(sandbox_root);
          }
        } else {
          Usage(args->front(),
                "Multiple sandbox roots (-s) specified, expected one.");
        }
        break;
      case 'H':
        opt.fake_hostname = true;
        break;
      case 'n':
        if (opt.create_netns == NETNS_WITH_LOOPBACK) {
          Usage(args->front(), "Only one of -n and -N may be specified.");
        }
        opt.create_netns = NETNS;
        break;
      case 'N':
        if (opt.create_netns == NETNS) {
          Usage(args->front(), "Only one of -n and -N may be specified.");
        }
        opt.create_netns = NETNS_WITH_LOOPBACK;
        break;
      case 'R':
        if (opt.fake_username) {
          Usage(args->front(),
                "The -R option cannot be used at the same time us the -U "
                "option.");
        }
        opt.fake_root = true;
        break;
      case 'U':
        if (opt.fake_root) {
          Usage(args->front(),
                "The -U option cannot be used at the same time us the -R "
                "option.");
        }
        opt.fake_username = true;
        break;
      case 'C':
        ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
        opt.cgroups_dir.assign(optarg);
        break;
      case 'P':
        opt.enable_pty = true;
        break;
      case 'D':
        if (opt.debug_path.empty()) {
          ValidateIsAbsolutePath(optarg, args->front(), static_cast<char>(c));
          opt.debug_path.assign(optarg);
        } else {
          Usage(args->front(),
                "Cannot write debug output to more than one file.");
        }
        break;
      case '?':
        Usage(args->front(), "Unrecognized argument: -%c (%d)", optopt, optind);
        break;
      case ':':
        Usage(args->front(), "Flag -%c requires an argument", optopt);
        break;
    }
  }

  if (!opt.working_dir.empty() && !opt.sandbox_root.empty() &&
      opt.working_dir.find(opt.sandbox_root) == std::string::npos) {
    Usage(args->front(),
          "working-dir %s (-W) should be a "
          "subdirectory of sandbox-dir %s (-h)",
          opt.working_dir.c_str(), opt.sandbox_root.c_str());
  }
  if (optind < static_cast<int>(args->size())) {
    if (opt.args.empty()) {
      opt.args.assign(args->begin() + optind, args->end());
    } else {
      Usage(args->front(), "Merging commands not supported.");
    }
  }
}

// Expands a single argument, expanding options @filename to read in the content
// of the file and add it to the list of processed arguments.
static unique_ptr<vector<char *>> ExpandArgument(
    unique_ptr<vector<char *>> expanded, char *arg) {
  if (arg[0] == '@') {
    const char *filename = arg + 1;  // strip off the '@'.
    ifstream f(filename);

    if (!f.is_open()) {
      DIE("opening argument file %s failed", filename);
    }

    for (std::string line; std::getline(f, line);) {
      if (line.length() > 0) {
        expanded = ExpandArgument(std::move(expanded), strdup(line.c_str()));
      }
    }

    if (f.bad()) {
      DIE("error while reading from argument file %s", filename);
    }
  } else {
    expanded->push_back(arg);
  }

  return expanded;
}

// Pre-processes an argument list, expanding options @filename to read in the
// content of the file and add it to the list of arguments. Stops expanding
// arguments once it encounters "--".
static unique_ptr<vector<char *>> ExpandArguments(const vector<char *> &args) {
  unique_ptr<vector<char *>> expanded(new vector<char *>());
  expanded->reserve(args.size());
  for (auto arg = args.begin(); arg != args.end(); ++arg) {
    if (strcmp(*arg, "--") != 0) {
      expanded = ExpandArgument(std::move(expanded), *arg);
    } else {
      expanded->insert(expanded->end(), arg, args.end());
      break;
    }
  }
  return expanded;
}

void ParseOptions(int argc, char *argv[]) {
  vector<char *> args(argv, argv + argc);
  ParseCommandLine(ExpandArguments(args));

  if (opt.args.empty()) {
    Usage(args.front(), "No command specified.");
  }

  if (opt.working_dir.empty()) {
    char *working_dir = getcwd(nullptr, 0);
    if (working_dir == nullptr) {
      DIE("getcwd");
    }
    opt.working_dir = working_dir;
    free(working_dir);
  }
}
