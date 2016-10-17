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

// This class is responsible for parsing the command line of the Blaze binary,
// parsing blazerc files, and putting together the command that should be sent
// to the server.
class OptionProcessor {
 public:
  OptionProcessor(std::unique_ptr<StartupOptions> default_startup_options);

  virtual ~OptionProcessor();

  // Parse a command line and the appropriate blazerc files. This should be
  // invoked only once per OptionProcessor object.
  blaze_exit_code::ExitCode ParseOptions(const std::vector<std::string>& args,
                                         const std::string& workspace,
                                         const std::string& cwd,
                                         std::string* error);

  blaze_exit_code::ExitCode ParseOptions(int argc, const char* argv[],
                                         const std::string& workspace,
                                         const std::string& cwd,
                                         std::string* error);

  // Get the Blaze command to be executed.
  // Returns an empty string if no command was found on the command line.
  const std::string& GetCommand() const;

  // Gets the arguments to the command. This is put together from the default
  // options specified in the blazerc file(s), the command line, and various
  // bits and pieces of information about the environment the blaze binary is
  // executed in.
  void GetCommandArguments(std::vector<std::string>* result) const;

  StartupOptions* GetParsedStartupOptions() const;

  virtual blaze_exit_code::ExitCode FindUserBlazerc(
      const char* cmdLineRcFile, const std::string& rc_basename,
      const std::string& workspace, std::string* user_blazerc_file,
      std::string* error);

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
        const std::string& workspace, std::vector<RcFile*>* rcfiles,
        std::map<std::string, std::vector<RcOption> >* rcoptions,
        std::string* error);
    const std::string& Filename() const { return filename_; }
    const int Index() const { return index_; }

   private:
    static blaze_exit_code::ExitCode Parse(
        const std::string& workspace, const std::string& filename,
        const int index, std::vector<RcFile*>* rcfiles,
        std::map<std::string, std::vector<RcOption> >* rcoptions,
        std::list<std::string>* import_stack, std::string* error);

    std::string filename_;
    int index_;
  };

  void AddRcfileArgsAndOptions(bool batch, const std::string& cwd);
  blaze_exit_code::ExitCode ParseStartupOptions(std::string* error);

  std::vector<RcFile*> blazercs_;
  std::map<std::string, std::vector<RcOption> > rcoptions_;
  std::vector<std::string> args_;
  unsigned int startup_args_;
  std::string command_;
  std::vector<std::string> command_arguments_;
  bool initialized_;
  std::unique_ptr<StartupOptions> parsed_startup_options_;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_
