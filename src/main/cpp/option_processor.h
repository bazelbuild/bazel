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
#include <map>
#include <memory>
#include <string>
#include <vector>

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

  virtual ~OptionProcessor();

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

  // Parse a command line and the appropriate blazerc files. This should be
  // invoked only once per OptionProcessor object.
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

  virtual blaze_exit_code::ExitCode FindUserBlazerc(
      const char* cmdLineRcFile,
      const std::string& workspace, std::string* user_blazerc_file,
      std::string* error);

  // Prints a message about the origin of startup options. This should be called
  // if the server is not started or called, in case the options are related
  // to the failure. Otherwise, the server will handle any required logging.
  void PrintStartupOptionsProvenanceMessage() const;

 private:
  class RcOption {
   public:
    RcOption(int rcfile_index, const std::string& option);

    const int rcfile_index() const { return rcfile_index_; }
    const std::string& option() const { return option_; }

   private:
    int rcfile_index_;
    std::string option_;
  };

  class RcFile {
   public:
    RcFile(const std::string& filename, int index);
    blaze_exit_code::ExitCode Parse(
        const std::string& workspace, const WorkspaceLayout* workspace_layout,
        std::vector<RcFile*>* rcfiles,
        std::map<std::string, std::vector<RcOption>>* rcoptions,
        std::string* error);
    const std::string& Filename() const { return filename_; }
    const int Index() const { return index_; }

   private:
    static blaze_exit_code::ExitCode Parse(
        const std::string& workspace, const std::string& filename,
        const int index, const WorkspaceLayout* workspace_layout,
        std::vector<RcFile*>* rcfiles,
        std::map<std::string, std::vector<RcOption>>* rcoptions,
        std::list<std::string>* import_stack, std::string* error);

    std::string filename_;
    int index_;
  };

  blaze_exit_code::ExitCode ParseStartupOptions(std::string* error);

  static std::vector<std::string> GetBlazercAndEnvCommandArgs(
      const std::string& cwd,
      const std::vector<RcFile*>& blazercs,
      const std::map<std::string, std::vector<RcOption>>& rcoptions);

  // The list of parsed rc files, this field is initialized by ParseOptions.
  std::vector<RcFile*> blazercs_;

  // A map representing the flags parsed from the bazelrc files.
  // A key is a command (e.g. 'build', 'startup') and its value is an ordered
  // list of RcOptions.
  std::map<std::string, std::vector<RcOption>> rcoptions_;

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

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_
