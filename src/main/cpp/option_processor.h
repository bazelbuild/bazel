// Copyright 2014 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_
#define BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_

#include <list>
#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/rc_file.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/exit_code.h"

namespace blaze {

class WorkspaceLayout;

// Broken down structure of the command line into logical components. The raw
// arguments should not be referenced after this structure exists. This
// breakdown should suffice to access the parts of the command line that the
// client cares about, notably the binary and startup startup options.
struct CommandLine {
  const std::string path_to_binary;
  const std::vector<std::string> startup_args;
  const std::string command;
  const std::vector<std::string> command_args;

  CommandLine(const std::string& path_to_binary_arg,
              const std::vector<std::string>& startup_args_arg,
              const std::string& command_arg,
              const std::vector<std::string>& command_args_arg)
      : path_to_binary(path_to_binary_arg),
        startup_args(startup_args_arg),
        command(command_arg),
        command_args(command_args_arg) {}
};

// This class is responsible for parsing the command line of the Blaze binary,
// parsing blazerc files, and putting together the command that should be sent
// to the server.
class OptionProcessor {
 public:
  OptionProcessor(const WorkspaceLayout* workspace_layout,
                  std::unique_ptr<StartupOptions> default_startup_options);

  virtual ~OptionProcessor() {}

  // Splits the arguments of a command line invocation.
  //
  // For instance:
  // output/bazel --foo --bar=42 --bar blah build --myflag value :mytarget
  //
  // returns a CommandLine structure with the following values:
  // result.path_to_binary = "output/bazel"
  // result.startup_args = {"--foo", "--bar=42", "--bar=blah"}
  // result.command = "build"
  // result.command_args = {"--myflag", "value", ":mytarget"}
  //
  // Note that result.startup_args is guaranteed to contain only valid
  // startup options (w.r.t. StartupOptions::IsUnary and
  // StartupOptions::IsNullary) and unary startup args of the form '--bar blah'
  // are rewritten as '--bar=blah' for uniformity.
  // In turn, the command and command args are not rewritten nor validated.
  //
  // If the method fails then error will contain the cause, otherwise error
  // remains untouched.
  virtual std::unique_ptr<CommandLine> SplitCommandLine(
      const std::vector<std::string>& args,
      std::string* error) const;

  // Parse a command line and the appropriate blazerc files and stores the
  // results. This should be invoked only once per OptionProcessor object.
  blaze_exit_code::ExitCode ParseOptions(const std::vector<std::string>& args,
                                         const std::string& workspace,
                                         const std::string& cwd,
                                         std::string* error);

  // Get the Blaze command to be executed.
  // Returns an empty string if no command was found on the command line.
  std::string GetCommand() const;

  // Gets the arguments to the command. This is put together from the default
  // options specified in the blazerc file(s), the command line, and various
  // bits and pieces of information about the environment the blaze binary is
  // executed in.
  std::vector<std::string> GetCommandArguments() const;

  // Gets the arguments explicitly provided by the user's command line.
  std::vector<std::string> GetExplicitCommandArguments() const;

  virtual StartupOptions* GetParsedStartupOptions() const;

  // Prints a message about the origin of startup options. This should be called
  // if the server is not started or called, in case the options are related to
  // the failure. Otherwise, the server will handle any required logging.
  void PrintStartupOptionsProvenanceMessage() const;

  // Constructs all synthetic command args that should be passed to the
  // server to configure blazerc options and client environment.
  static std::vector<std::string> GetBlazercAndEnvCommandArgs(
      const std::string& cwd,
      const std::vector<std::unique_ptr<RcFile>>& blazercs,
      const std::vector<std::string>& env);

  // Finds and parses the appropriate RcFiles.
  // TODO(#4502) Change where the bazelrcs are read from.
  virtual blaze_exit_code::ExitCode GetRcFiles(
      const WorkspaceLayout* workspace_layout, const std::string& workspace,
      const std::string& cwd, const CommandLine* cmd_line,
      std::vector<std::unique_ptr<RcFile>>* result_rc_files,
      std::string* error) const;

 protected:
  // Return the path to the user's rc file.  If cmd_line_rc_file != NULL,
  // use it, dying if it is not readable.  Otherwise, return the first
  // readable file called rc_basename from [workspace, $HOME]
  //
  // If no readable .blazerc file is found, return the empty string.
  virtual blaze_exit_code::ExitCode FindUserBlazerc(
      const char* cmd_line_rc_file, const std::string& workspace,
      std::string* user_blazerc_file, std::string* error) const;

 private:
  blaze_exit_code::ExitCode ParseStartupOptions(
      const std::vector<std::unique_ptr<RcFile>>& rc_files,
      std::string* error);

  // An ordered list of command args that contain information about the
  // execution environment and the flags passed via the bazelrc files.
  std::vector<std::string> blazerc_and_env_command_args_;

  // The command line constructed after calling ParseOptions.
  std::unique_ptr<CommandLine> cmd_line_;

  const WorkspaceLayout* workspace_layout_;

  // The startup options parsed from args, this field is initialized by
  // ParseOptions.
  std::unique_ptr<StartupOptions> parsed_startup_options_;
};

// Parses and returns the contents of the rc file.
blaze_exit_code::ExitCode ParseRcFile(const WorkspaceLayout* workspace_layout,
                                      const std::string& workspace,
                                      const std::string& rc_file_path,
                                      std::unique_ptr<RcFile>* result_rc_file,
                                      std::string* error);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_
