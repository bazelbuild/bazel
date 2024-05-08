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

#ifndef SRC_MAIN_TOOLS_LINUX_SANDBOX_OPTIONS_H_
#define SRC_MAIN_TOOLS_LINUX_SANDBOX_OPTIONS_H_

#include <stdbool.h>
#include <stddef.h>

#include <string>
#include <vector>

enum NetNamespaceOption { NO_NETNS, NETNS, NETNS_WITH_LOOPBACK };

// Options parsing result.
struct Options {
  // Working directory (-W)
  std::string working_dir;
  // How long to wait before killing the child (-T)
  int timeout_secs;
  // How long to wait before sending SIGKILL in case of timeout (-t)
  int kill_delay_secs;
  // If set, the process persists after the death of its parent thread (-p)
  bool persistent_process;
  // Send a SIGTERM to the child on receipt of a SIGINT (-i)
  bool sigint_sends_sigterm;
  // Where to redirect stdout (-l)
  std::string stdout_path;
  // Where to redirect stderr (-L)
  std::string stderr_path;
  // Files or directories to make writable for the sandboxed process (-w)
  std::vector<std::string> writable_files;
  // Directories where to mount an empty tmpfs (-e)
  std::vector<std::string> tmpfs_dirs;
  // Source of files or directories to explicitly bind mount in the sandbox (-M)
  std::vector<std::string> bind_mount_sources;
  // Target of files or directories to explicitly bind mount in the sandbox (-m)
  std::vector<std::string> bind_mount_targets;
  // Where to write stats, in protobuf format (-S)
  std::string stats_path;
  // Set the hostname inside the sandbox to 'localhost' (-H)
  bool fake_hostname;
  // Create a new network namespace (-n/-N)
  NetNamespaceOption create_netns;
  // Pretend to be root inside the namespace (-R)
  bool fake_root;
  // Set the username inside the sandbox to 'nobody' (-U)
  bool fake_username;
  // Enable writing to /dev/pts and map the user's gid to tty to enable
  // pseudoterminals (-P)
  bool enable_pty;
  // Print debugging messages (-D)
  std::string debug_path;
  // Improved hermetic build using whitelisting strategy (-h)
  bool hermetic;
  // The sandbox root directory (-s)
  std::string sandbox_root;
  // Directory to use for cgroup control
  std::string cgroups_dir;
  // Command to run (--)
  std::vector<char *> args;
};

extern struct Options opt;

// Handles parsing all command line flags and populates the global opt struct.
void ParseOptions(int argc, char *argv[]);

#endif
