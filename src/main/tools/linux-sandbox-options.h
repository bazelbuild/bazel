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

#ifndef LINUX_SANDBOX_OPTIONS_H__
#define LINUX_SANDBOX_OPTIONS_H__

#include <stdbool.h>
#include <stddef.h>

#include <vector>

// Options parsing result.
struct Options {
  // Temporary root directory (-S)
  const char *sandbox_root_dir;
  // Working directory (-W)
  const char *working_dir;
  // How long to wait before killing the child (-T)
  int timeout_secs;
  // How long to wait before sending SIGKILL in case of timeout (-t)
  int kill_delay_secs;
  // Where to redirect stdout (-l)
  const char *stdout_path;
  // Where to redirect stderr (-L)
  const char *stderr_path;
  // Files or directories to make writable for the sandboxed process (-w)
  std::vector<const char *> writable_files;
  // Files or directories to make inaccessible for the sandboxed process (-i)
  std::vector<const char *> inaccessible_files;
  // Directories where to mount an empty tmpfs (-e)
  std::vector<const char *> tmpfs_dirs;
  // Source of files or directories to explicitly bind mount in the sandbox (-M)
  std::vector<const char *> bind_mount_sources;
  // Target of files or directories to explicitly bind mount in the sandbox (-m)
  std::vector<const char *> bind_mount_targets;
  // Create a new network namespace (-N)
  bool create_netns;
  // Pretend to be root inside the namespace (-R)
  bool fake_root;
  // Print debugging messages (-D)
  bool debug;
  // Command to run (--)
  std::vector<char *> args;
};

extern struct Options opt;

void ParseOptions(int argc, char *argv[]);

#endif
