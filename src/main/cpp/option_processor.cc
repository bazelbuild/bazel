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

#include "src/main/cpp/option_processor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <utility>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

using std::list;
using std::map;
using std::vector;

// On OSX, there apparently is no header that defines this.
extern char **environ;

namespace blaze {

OptionProcessor::RcOption::RcOption(int rcfile_index, const string& option) {
  rcfile_index_ = rcfile_index;
  option_ = option;
}

OptionProcessor::RcFile::RcFile(const string& filename, int index) {
  filename_ = filename;
  index_ = index;
}

blaze_exit_code::ExitCode OptionProcessor::RcFile::Parse(
    vector<RcFile>* rcfiles,
    map<string, vector<RcOption> >* rcoptions,
    string* error) {
  list<string> initial_import_stack;
  initial_import_stack.push_back(filename_);
  return Parse(
      filename_, index_, rcfiles, rcoptions, &initial_import_stack, error);
}

blaze_exit_code::ExitCode OptionProcessor::RcFile::Parse(
    const string& filename,
    const int index,
    vector<RcFile>* rcfiles,
    map<string, vector<RcOption> >* rcoptions,
    list<string>* import_stack,
    string* error) {
  string contents;
  if (!ReadFile(filename, &contents)) {
    // We checked for file readability before, so this is unexpected.
    blaze_util::StringPrintf(error,
        "Unexpected error reading .blazerc file '%s'", filename.c_str());
    return blaze_exit_code::INTERNAL_ERROR;
  }

  // A '\' at the end of a line continues the line.
  blaze_util::Replace("\\\r\n", "", &contents);
  blaze_util::Replace("\\\n", "", &contents);
  vector<string> startup_options;

  vector<string> lines = blaze_util::Split(contents, '\n');
  for (int line = 0; line < lines.size(); ++line) {
    blaze_util::StripWhitespace(&lines[line]);

    // Check for an empty line.
    if (lines[line].empty()) {
      continue;
    }

    vector<string> words;

    // This will treat "#" as a comment, and properly
    // quote single and double quotes, and treat '\'
    // as an escape character.
    // TODO(bazel-team): This function silently ignores
    // dangling backslash escapes and missing end-quotes.
    blaze_util::Tokenize(lines[line], '#', &words);

    if (words.empty()) {
      // Could happen if line starts with "#"
      continue;
    }

    string command = words[0];

    if (command == "import") {
      if (words.size() != 2) {
        blaze_util::StringPrintf(error,
            "Invalid import declaration in .blazerc file '%s': '%s'",
            filename.c_str(), lines[line].c_str());
        return blaze_exit_code::BAD_ARGV;
      }

      if (std::find(import_stack->begin(), import_stack->end(), words[1]) !=
          import_stack->end()) {
        string loop;
        for (list<string>::const_iterator imported_rc = import_stack->begin();
             imported_rc != import_stack->end(); ++imported_rc) {
          loop += "  " + *imported_rc + "\n";
        }
        blaze_util::StringPrintf(error,
            "Import loop detected:\n%s", loop.c_str());
        return blaze_exit_code::BAD_ARGV;
      }

      rcfiles->push_back(RcFile(words[1], rcfiles->size()));
      import_stack->push_back(words[1]);
      blaze_exit_code::ExitCode parse_exit_code = RcFile::Parse(
          rcfiles->back().Filename(), rcfiles->back().Index(),
          rcfiles, rcoptions, import_stack, error);
      if (parse_exit_code != blaze_exit_code::SUCCESS) {
        return parse_exit_code;
      }
      import_stack->pop_back();
    } else {
      for (int word = 1; word < words.size(); ++word) {
        (*rcoptions)[command].push_back(RcOption(index, words[word]));
        if (command == "startup") {
          startup_options.push_back(words[word]);
        }
      }
    }
  }

  if (!startup_options.empty()) {
    string startup_args;
    blaze_util::JoinStrings(startup_options, ' ', &startup_args);
    fprintf(stderr, "INFO: Reading 'startup' options from %s: %s\n",
            filename.c_str(), startup_args.c_str());
  }
  return blaze_exit_code::SUCCESS;
}

OptionProcessor::OptionProcessor()
    : initialized_(false), parsed_startup_options_(new BlazeStartupOptions()) {
}

// Return the path of the depot .blazerc file.
string OptionProcessor::FindDepotBlazerc(const string& workspace) {
  // Package semantics are ignored here, but that's acceptable because
  // blaze.blazerc is a configuration file.
  vector<string> candidates;
  BlazeStartupOptions::WorkspaceRcFileSearchPath(&candidates);
  for (const auto& candidate : candidates) {
    string blazerc = blaze_util::JoinPath(workspace, candidate);
    if (!access(blazerc.c_str(), R_OK)) {
      return blazerc;
    }
  }

  return "";
}

// Return the path of the .blazerc file that sits alongside the binary.
// This allows for canary or cross-platform Blazes operating on the same depot
// to have customized behavior.
string OptionProcessor::FindAlongsideBinaryBlazerc(const string& cwd,
                                                   const string& arg0) {
  string path = arg0[0] == '/' ? arg0 : blaze_util::JoinPath(cwd, arg0);
  string base = blaze_util::Basename(arg0);
  string binary_blazerc_path = path + "." + base + "rc";
  if (!access(binary_blazerc_path.c_str(), R_OK)) {
    return binary_blazerc_path;
  }
  return "";
}


// Return the path the the user rc file.  If cmdLineRcFile != NULL,
// use it, dying if it is not readable.  Otherwise, return the first
// readable file called rc_basename from [workspace, $HOME]
//
// If no readable .blazerc file is found, return the empty string.
blaze_exit_code::ExitCode OptionProcessor::FindUserBlazerc(
    const char* cmdLineRcFile,
    const string& rc_basename,
    const string& workspace,
    string* blaze_rc_file,
    string* error) {
  if (cmdLineRcFile != NULL) {
    string rcFile = MakeAbsolute(cmdLineRcFile);
    if (access(rcFile.c_str(), R_OK)) {
      blaze_util::StringPrintf(error,
          "Error: Unable to read .blazerc file '%s'.", rcFile.c_str());
      return blaze_exit_code::BAD_ARGV;
    }
    *blaze_rc_file = rcFile;
    return blaze_exit_code::SUCCESS;
  }

  string workspaceRcFile = blaze_util::JoinPath(workspace, rc_basename);
  if (!access(workspaceRcFile.c_str(), R_OK)) {
    *blaze_rc_file = workspaceRcFile;
    return blaze_exit_code::SUCCESS;
  }

  const char* home = getenv("HOME");
  if (home == NULL) {
    *blaze_rc_file = "";
    return blaze_exit_code::SUCCESS;
  }

  string userRcFile = blaze_util::JoinPath(home, rc_basename);
  if (!access(userRcFile.c_str(), R_OK)) {
    *blaze_rc_file = userRcFile;
    return blaze_exit_code::SUCCESS;
  }
  *blaze_rc_file = "";
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    const vector<string>& args,
    const string& workspace,
    const string& cwd,
    string* error) {
  assert(!initialized_);
  initialized_ = true;

  const char* blazerc = NULL;
  bool use_master_blazerc = true;

  // Check if there is a blazerc related option given
  args_ = args;
  for (int i= 1; i < args.size(); i++) {
    const char* arg_chr = args[i].c_str();
    const char* next_arg_chr = (i + 1) < args.size()
        ? args[i + 1].c_str()
        : NULL;
    if (blazerc == NULL) {
      blazerc = GetUnaryOption(arg_chr, next_arg_chr, "--blazerc");
    }
    if (use_master_blazerc &&
        GetNullaryOption(arg_chr, "--nomaster_blazerc")) {
      use_master_blazerc = false;
    }
  }

  // Parse depot and user blazerc files.
  // This is not a little ineffective (copying a multimap around), but it is a
  // small one and this way I don't have to care about memory management.
  if (use_master_blazerc) {
    string depot_blazerc_path = FindDepotBlazerc(workspace);
    if (!depot_blazerc_path.empty()) {
      blazercs_.push_back(RcFile(depot_blazerc_path, blazercs_.size()));
      blaze_exit_code::ExitCode parse_exit_code =
          blazercs_.back().Parse(&blazercs_, &rcoptions_, error);
      if (parse_exit_code != blaze_exit_code::SUCCESS) {
        return parse_exit_code;
      }
    }
    string alongside_binary_blazerc = FindAlongsideBinaryBlazerc(cwd, args[0]);
    if (!alongside_binary_blazerc.empty()) {
      blazercs_.push_back(RcFile(alongside_binary_blazerc, blazercs_.size()));
      blaze_exit_code::ExitCode parse_exit_code =
          blazercs_.back().Parse(&blazercs_, &rcoptions_, error);
      if (parse_exit_code != blaze_exit_code::SUCCESS) {
        return parse_exit_code;
      }
    }
  }

  string user_blazerc_path;
  blaze_exit_code::ExitCode find_blazerc_exit_code = FindUserBlazerc(
      blazerc, BlazeStartupOptions::RcBasename(), workspace, &user_blazerc_path,
      error);
  if (find_blazerc_exit_code != blaze_exit_code::SUCCESS) {
    return find_blazerc_exit_code;
  }
  if (!user_blazerc_path.empty()) {
    blazercs_.push_back(RcFile(user_blazerc_path, blazercs_.size()));
    blaze_exit_code::ExitCode parse_exit_code =
        blazercs_.back().Parse(&blazercs_, &rcoptions_, error);
    if (parse_exit_code != blaze_exit_code::SUCCESS) {
      return parse_exit_code;
    }
  }

  blaze_exit_code::ExitCode parse_startup_options_exit_code =
      ParseStartupOptions(error);
  if (parse_startup_options_exit_code != blaze_exit_code::SUCCESS) {
    return parse_startup_options_exit_code;
  }

  // Determine command
  if (startup_args_ + 1 >= args.size()) {
    command_ = "";
    return blaze_exit_code::SUCCESS;
  }

  command_ = args[startup_args_ + 1];

  AddRcfileArgsAndOptions(parsed_startup_options_->batch, cwd);
  for (unsigned int cmd_arg = startup_args_ + 2;
       cmd_arg < args.size(); cmd_arg++) {
    command_arguments_.push_back(args[cmd_arg]);
  }
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    int argc,
    const char* argv[],
    const string& workspace,
    const string& cwd,
    string* error) {
  vector<string> args(argc);
  for (int arg = 0; arg < argc; arg++) {
    args[arg] = argv[arg];
  }

  return ParseOptions(args, workspace, cwd, error);
}

static bool IsArg(const string& arg) {
  return blaze_util::starts_with(arg, "-") && (arg != "--help")
      && (arg != "-help") && (arg != "-h");
}

blaze_exit_code::ExitCode OptionProcessor::ParseStartupOptions(string *error) {
  // Process rcfile startup options
  map< string, vector<RcOption> >::const_iterator it =
      rcoptions_.find("startup");
  blaze_exit_code::ExitCode process_arg_exit_code;
  bool is_space_separated;
  if (it != rcoptions_.end()) {
    const vector<RcOption>& startup_options = it->second;
    int i = 0;
    // Process all elements except the last one.
    for (; i < startup_options.size() - 1; i++) {
      const RcOption& option = startup_options[i];
      const string& blazerc = blazercs_[option.rcfile_index()].Filename();
      process_arg_exit_code = parsed_startup_options_->ProcessArg(
          option.option(), startup_options[i + 1].option(), blazerc,
          &is_space_separated, error);
      if (process_arg_exit_code != blaze_exit_code::SUCCESS) {
          return process_arg_exit_code;
      }
      if (is_space_separated) {
        i++;
      }
    }
    // Process last element, if any.
    if (i < startup_options.size()) {
      const RcOption& option = startup_options[i];
      if (IsArg(option.option())) {
        const string& blazerc = blazercs_[option.rcfile_index()].Filename();
        process_arg_exit_code = parsed_startup_options_->ProcessArg(
            option.option(), "", blazerc, &is_space_separated, error);
        if (process_arg_exit_code != blaze_exit_code::SUCCESS) {
          return process_arg_exit_code;
        }
      }
    }
  }

  // Process command-line args next, so they override any of the same options
  // from .blazerc. Stop on first non-arg, this includes --help
  unsigned int i = 1;
  if (!args_.empty()) {
    for (;  (i < args_.size() - 1) && IsArg(args_[i]); i++) {
      process_arg_exit_code = parsed_startup_options_->ProcessArg(
          args_[i], args_[i + 1], "", &is_space_separated, error);
      if (process_arg_exit_code != blaze_exit_code::SUCCESS) {
          return process_arg_exit_code;
      }
      if (is_space_separated) {
        i++;
      }
    }
    if (i < args_.size() && IsArg(args_[i])) {
      process_arg_exit_code = parsed_startup_options_->ProcessArg(
          args_[i], "", "", &is_space_separated, error);
      if (process_arg_exit_code != blaze_exit_code::SUCCESS) {
          return process_arg_exit_code;
      }
      i++;
    }
  }
  startup_args_ = i -1;

  return blaze_exit_code::SUCCESS;
}

// Appends the command and arguments from argc/argv to the end of arg_vector,
// and also splices in some additional terminal and environment options between
// the command and the arguments. NB: Keep the options added here in sync with
// BlazeCommandDispatcher.INTERNAL_COMMAND_OPTIONS!
void OptionProcessor::AddRcfileArgsAndOptions(bool batch, const string& cwd) {
  // Push the options mapping .blazerc numbers to filenames.
  for (int i_blazerc = 0; i_blazerc < blazercs_.size(); i_blazerc++) {
    const RcFile& blazerc = blazercs_[i_blazerc];
    command_arguments_.push_back("--rc_source=" + blazerc.Filename());
  }

  // Push the option defaults
  for (map<string, vector<RcOption> >::const_iterator it = rcoptions_.begin();
       it != rcoptions_.end(); ++it) {
    if (it->first == "startup") {
      // Skip startup options, they are parsed in the C++ wrapper
      continue;
    }

    for (int ii = 0; ii < it->second.size(); ii++) {
      const RcOption& rcoption = it->second[ii];
      command_arguments_.push_back(
          "--default_override=" + std::to_string(rcoption.rcfile_index()) + ":"
          + it->first + "=" + rcoption.option());
    }
  }

  // Splice the terminal options.
  command_arguments_.push_back(
      "--isatty=" + std::to_string(IsStandardTerminal()));
  command_arguments_.push_back(
      "--terminal_columns=" + std::to_string(GetTerminalColumns()));

  // Pass the client environment to the server in server mode.
  if (batch) {
    command_arguments_.push_back("--ignore_client_env");
  } else {
    for (char** env = environ; *env != NULL; env++) {
      command_arguments_.push_back("--client_env=" + string(*env));
    }
  }
  command_arguments_.push_back("--client_cwd=" + cwd);

  const char *emacs = getenv("EMACS");
  if (emacs != NULL && strcmp(emacs, "t") == 0) {
    command_arguments_.push_back("--emacs");
  }
}

void OptionProcessor::GetCommandArguments(vector<string>* result) const {
  result->insert(result->end(),
                 command_arguments_.begin(),
                 command_arguments_.end());
}

const string& OptionProcessor::GetCommand() const {
  return command_;
}

const BlazeStartupOptions& OptionProcessor::GetParsedStartupOptions() const {
  return *parsed_startup_options_.get();
}
}  // namespace blaze
