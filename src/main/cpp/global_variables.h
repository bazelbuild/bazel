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
// global_variables.h: The global state in the blaze.cc Blaze client.
//

#ifndef BAZEL_SRC_MAIN_CPP_GLOBAL_VARIABLES_H_
#define BAZEL_SRC_MAIN_CPP_GLOBAL_VARIABLES_H_

#include <sys/types.h>
#include <string>
#include <vector>

#include "src/main/cpp/util/port.h"  // pid_t on Windows/MSVC

namespace blaze {

class OptionProcessor;
class StartupOptions;

// The reason for a blaze server restart.
// Keep in sync with logging.proto.
enum RestartReason {
  NO_RESTART = 0,
  NO_DAEMON,
  NEW_VERSION,
  NEW_OPTIONS,
  PID_FILE_BUT_NO_SERVER,
  SERVER_VANISHED,
  SERVER_UNRESPONSIVE
};

struct GlobalVariables {
  GlobalVariables(OptionProcessor *option_processor);

  std::string ServerJarPath() const {
    // The server jar is called "A-server.jar" so it's the first binary we
    // extracted.
    return extracted_binaries.empty() ? "" : extracted_binaries[0];
  }

  // Used to make concurrent invocations of this program safe.
  std::string lockfile;  // = <output_base>/lock

  // Whrere to write the server's JVM's output. Default value is
  // <output_base>/server/jvm.out.
  std::string jvm_log_file;

  // Whether or not the jvm_log_file should be opened with O_APPEND.
  bool jvm_log_file_append;

  std::string cwd;

  // The nearest enclosing workspace directory, starting from cwd.
  // If not under a workspace directory, this is equal to cwd.
  std::string workspace;

  // Option processor responsible for parsing RC files and converting them into
  // the argument list passed on to the server.
  OptionProcessor *option_processor;

  // The path of the JVM executable that should be used to launch Blaze.
  std::string jvm_path;

  // TODO(laszlocsomor) 2016-11-28: move pid_t usage out of here and wherever
  // else it appears. Find some way to not have to declare a pid_t here, either
  // by making PID handling platform-independent or some other idea.
  pid_t server_pid;

  // Contains the relative paths of all the files in the attached zip, and is
  // populated during GetInstallBase().
  std::vector<std::string> extracted_binaries;

  // Parsed startup options.
  StartupOptions *options;  // TODO(jmmv): This should really be const.

  // The time in ms the launcher spends before sending the request to the blaze
  // server.
  uint64_t startup_time;

  // The time in ms spent on extracting the new blaze version.
  // This is part of startup_time.
  uint64_t extract_data_time;

  // The time in ms a command had to wait on a busy Blaze server process.
  // This is part of startup_time.
  uint64_t command_wait_time;

  // The reason for the server restart.
  RestartReason restart_reason;

  // The absolute path of the blaze binary.
  std::string binary_path;

  // MD5 hash of the Blaze binary (includes deploy.jar, extracted binaries, and
  // anything else that ends up under the install_base).
  std::string install_md5;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_GLOBAL_VARIABLES_H_
