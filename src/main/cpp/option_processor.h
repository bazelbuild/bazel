// Copyright 2014 Google Inc. All rights reserved.
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

#include "src/main/cpp/blaze_startup_options.h"
#include "src/main/cpp/util/exit_code.h"

namespace blaze {

using std::string;

// This class is responsible for parsing the command line of the Blaze binary,
// parsing blazerc files, and putting together the command that should be sent
// to the server.
class OptionProcessor {
 public:
  OptionProcessor();

  virtual ~OptionProcessor();

  // Parse a command line and the appropriate blazerc files. This should be
  // invoked only once per OptionProcessor object.
  blaze_exit_code::ExitCode ParseOptions(const std::vector<string>& args,
                                         const string& workspace,
                                         const string& cwd,
                                         string* error);

  blaze_exit_code::ExitCode ParseOptions(int argc, const char* argv[],
                                         const string& workspace,
                                         const string& cwd,
                                         string* error);

  // Get the Blaze command to be executed.
  // Returns an empty string if no command was found on the command line.
  const string& GetCommand() const;

  // Gets the arguments to the command. This is put together from the default
  // options specified in the blazerc file(s), the command line, and various
  // bits and pieces of information about the environment the blaze binary is
  // executed in.
  void GetCommandArguments(std::vector<string>* result) const;

  const BlazeStartupOptions& GetParsedStartupOptions() const;

  virtual string FindDepotBlazerc(const string& workspace);
  virtual string FindAlongsideBinaryBlazerc(const string& cwd,
                                            const string& arg0);
  virtual blaze_exit_code::ExitCode FindUserBlazerc(const char* cmdLineRcFile,
                                                    const string& rc_basename,
                                                    const string& workspace,
                                                    string* user_blazerc_file,
                                                    string* error);

 private:
  class RcOption {
   public:
    RcOption(int rcfile_index, const string& option);

    const int rcfile_index() const { return rcfile_index_; }
    const string& option() const { return option_; }

   private:
    int rcfile_index_;
    string option_;
  };

  class RcFile {
   public:
    RcFile(const string& filename, int index);
    blaze_exit_code::ExitCode Parse(
        std::vector<RcFile*>* rcfiles,
        std::map<string, std::vector<RcOption> >* rcoptions,
        string* error);
    const string& Filename() const { return filename_; }
    const int Index() const { return index_; }

   private:
    static blaze_exit_code::ExitCode Parse(const string& filename,
                                           const int index,
                                           std::vector<RcFile*>* rcfiles,
                                           std::map<string,
                                           std::vector<RcOption> >* rcoptions,
                                           std::list<string>* import_stack,
                                           string* error);

    string filename_;
    int index_;
  };

  void AddRcfileArgsAndOptions(bool batch, const string& cwd);
  blaze_exit_code::ExitCode ParseStartupOptions(string *error);

  std::vector<RcFile*> blazercs_;
  std::map<string, std::vector<RcOption> > rcoptions_;
  std::vector<string> args_;
  unsigned int startup_args_;
  string command_;
  std::vector<string> command_arguments_;
  bool initialized_;
  std::unique_ptr<BlazeStartupOptions> parsed_startup_options_;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_OPTION_PROCESSOR_H_
