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
//
// blaze_globals.h: The global state in the blaze.cc Blaze client.
//

#ifndef BAZEL_SRC_MAIN_CPP_BLAZE_GLOBALS_H_
#define BAZEL_SRC_MAIN_CPP_BLAZE_GLOBALS_H_

#include <signal.h>
#include <sys/types.h>
#include <string>
#include <vector>

#include "src/main/cpp/blaze_startup_options.h"
#include "src/main/cpp/option_processor.h"

using std::vector;

namespace blaze {

// The reason for a blaze server restart.
// Keep in sync with logging.proto
enum RestartReason {
  NO_RESTART = 0,
  NO_DAEMON,
  NEW_VERSION,
  NEW_OPTIONS
};

struct GlobalVariables {
  // Used to make concurrent invocations of this program safe.
  string lockfile;  // = <output_base>/lock
  int lockfd;

  string jvm_log_file;  // = <output_base>/server/jvm.out

  string cwd;

  // The nearest enclosing workspace directory, starting from cwd.
  // If not under a workspace directory, this is equal to cwd.
  string workspace;

  // Option processor responsible for parsing RC files and converting them into
  // the argument list passed on to the server.
  OptionProcessor option_processor;

  pid_t server_pid;

  volatile sig_atomic_t sigint_count;

  // The number of the last received signal that should cause the client
  // to shutdown.  This is saved so that the client's WTERMSIG can be set
  // correctly.  (Currently only SIGPIPE uses this mechanism.)
  volatile sig_atomic_t received_signal;

  // Contains the relative paths of all the files in the attached zip, and is
  // populated during GetInstallDir().
  vector<string> extracted_binaries;

  // Parsed startup options
  BlazeStartupOptions options;

  // The time in ms the launcher spends before sending the request to the Blaze
  uint64_t startup_time;

  // The time spent on extracting the new blaze version
  // This is part of startup_time
  uint64_t extract_data_time;

  // The time in ms if a command had to wait on a busy Blaze server process
  // This is part of startup_time
  uint64_t command_wait_time;

  RestartReason restart_reason;

  // Absolute path of the blaze binary
  string binary_path;

  // MD5 hash of the Blaze binary (includes deploy.jar, extracted binaries, and
  // anything else that ends up under the install_base).
  string install_md5;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_BLAZE_GLOBALS_H_
