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

#ifndef BAZEL_SRC_TOOLS_LAUNCHER_JAVA_LAUNCHER_H_
#define BAZEL_SRC_TOOLS_LAUNCHER_JAVA_LAUNCHER_H_

#include <string>
#include <vector>

#include "src/tools/launcher/launcher.h"

namespace bazel {
namespace launcher {

// Windows per-arg limit is MAX_ARG_STRLEN == 8k,
// here we use a slightly smaller value.
static const int MAX_ARG_STRLEN = 7000;

class JavaBinaryLauncher : public BinaryLauncherBase {
 public:
  JavaBinaryLauncher(const LaunchDataParser::LaunchInfo& launch_info, int argc,
                     wchar_t* argv[])
      : BinaryLauncherBase(launch_info, argc, argv),
        singlejar(false),
        print_javabin(false),
        classpath_limit(MAX_ARG_STRLEN) {}
  ~JavaBinaryLauncher() override = default;
  ExitCode Launch() override;

 private:
  // If present, these flags should either be at the beginning of the command
  // line, or they should be wrapped in a --wrapper_script_flag=FLAG argument.
  //
  // --debug               Launch the JVM in remote debugging mode listening
  // --debug=<port>        to the specified port or the port set in the
  //                       DEFAULT_JVM_DEBUG_PORT environment variable (e.g.
  //                       'export DEFAULT_JVM_DEBUG_PORT=8000') or else the
  //                       default port of 5005.  The JVM starts suspended
  //                       unless the DEFAULT_JVM_DEBUG_SUSPEND environment
  //                       variable is set to 'n'.
  // --main_advice=<class> Run an alternate main class with the usual main
  //                       program and arguments appended as arguments.
  // --main_advice_classpath=<classpath>
  //                       Prepend additional class path entries.
  // --jvm_flag=<flag>     Pass <flag> to the "java" command itself.
  //                       <flag> may contain spaces. Can be used multiple
  //                       times.
  // --jvm_flags=<flags>   Pass space-separated flags to the "java" command
  //                       itself. Can be used multiple times.
  // --singlejar           Start the program from the packed-up deployment
  //                       jar rather than from the classpath.
  // --print_javabin       Print the location of java executable binary and
  // exit.
  // --classpath_limit=<length>
  //                       Specify the maximum classpath length. If the
  //                       classpath is shorter, this script passes it to Java
  //                       as a command line flag, otherwise it creates a
  //                       classpath jar.
  //
  // The remainder of the command line is passed to the program.
  bool ProcessWrapperArgument(const std::wstring& argument);

  // Parse arguments sequentially until the first unrecognized arg is
  // encountered. Scan the remaining args for --wrapper_script_flag=X options
  // and process them.
  //
  // Return the remaining arguments that should be passed to the program.
  std::vector<std::wstring> ProcessesCommandLine();

  std::wstring jvm_debug_port;
  std::wstring main_advice;
  std::wstring main_advice_classpath;
  std::vector<std::wstring> jvm_flags_cmdline;
  bool singlejar;
  bool print_javabin;
  int classpath_limit;

  // Create a classpath jar to pass CLASSPATH value when its length is over
  // limit.
  //
  // Return the path of the classpath jar created.
  std::wstring CreateClasspathJar(const std::wstring& classpath);

  // Creat a directory based on the binary path, all the junctions will be
  // generated under this directory.
  std::wstring GetJunctionBaseDir();

  // Delete all the junction directory and all the junctions under it.
  void DeleteJunctionBaseDir();
};

}  // namespace launcher
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_LAUNCHER_JAVA_LAUNCHER_H_
