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

#ifndef SRC_MAIN_TOOLS_PROCESS_WRAPPER_OPTIONS_H_
#define SRC_MAIN_TOOLS_PROCESS_WRAPPER_OPTIONS_H_

#include <string>
#include <vector>

// Options parsing result.
struct Options {
  // How long to wait before killing the child (-t)
  double timeout_secs;
  // How long to wait before sending SIGKILL in case of timeout (-k)
  double kill_delay_secs;
  // Where to redirect stdout (-o)
  std::string stdout_path;
  // Where to redirect stderr (-e)
  std::string stderr_path;
  // Whether to print debugging messages (-d)
  bool debug;
  // Where to write stats, in protobuf format (-s)
  std::string stats_path;
  // Command to run (--)
  std::vector<char *> args;
};

extern struct Options opt;

// Handles parsing all command line flags and populates the global opt struct.
void ParseOptions(int argc, char *argv[]);

#endif
