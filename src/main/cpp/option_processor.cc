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

#include "src/main/cpp/option_processor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cassert>
#include <set>
#include <utility>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"

// On OSX, there apparently is no header that defines this.
extern char **environ;

namespace blaze {

using std::list;
using std::map;
using std::set;
using std::string;
using std::vector;

constexpr char WorkspaceLayout::WorkspacePrefix[];

OptionProcessor::RcOption::RcOption(int rcfile_index, const string& option)
    : rcfile_index_(rcfile_index), option_(option) {
}

OptionProcessor::RcFile::RcFile(const string& filename, int index)
    : filename_(filename), index_(index) {
}

blaze_exit_code::ExitCode OptionProcessor::RcFile::Parse(
    const string& workspace,
    const WorkspaceLayout* workspace_layout,
    vector<RcFile*>* rcfiles,
    map<string, vector<RcOption> >* rcoptions,
    string* error) {
  list<string> initial_import_stack;
  initial_import_stack.push_back(filename_);
  return Parse(
      workspace, filename_, index_, workspace_layout,
      rcfiles, rcoptions, &initial_import_stack,
      error);
}

blaze_exit_code::ExitCode OptionProcessor::RcFile::Parse(
    const string& workspace,
    const string& filename_ref,
    const int index,
    const WorkspaceLayout* workspace_layout,
    vector<RcFile*>* rcfiles,
    map<string, vector<RcOption> >* rcoptions,
    list<string>* import_stack,
    string* error) {
  string filename(filename_ref);  // file
  BAZEL_LOG(INFO) << "Parsing the RcFile " << filename;
  string contents;
  if (!blaze_util::ReadFile(filename, &contents)) {
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
      if (words.size() != 2
          || (words[1].compare(0, workspace_layout->WorkspacePrefixLength,
                               workspace_layout->WorkspacePrefix) == 0
              && !workspace_layout->WorkspaceRelativizeRcFilePath(
                  workspace, &words[1]))) {
        blaze_util::StringPrintf(error,
            "Invalid import declaration in .blazerc file '%s': '%s'"
            " (are you in your source checkout/WORKSPACE?)",
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

      rcfiles->push_back(new RcFile(words[1], rcfiles->size()));
      import_stack->push_back(words[1]);
      blaze_exit_code::ExitCode parse_exit_code =
        RcFile::Parse(workspace, rcfiles->back()->Filename(),
                      rcfiles->back()->Index(), workspace_layout,
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

OptionProcessor::OptionProcessor(
    const WorkspaceLayout* workspace_layout,
    std::unique_ptr<StartupOptions> default_startup_options)
    : initialized_(false),
      workspace_layout_(workspace_layout),
      parsed_startup_options_(std::move(default_startup_options)) {
}

std::unique_ptr<CommandLine> OptionProcessor::SplitCommandLine(
    const vector<string>& args,
    string* error) {
  const string lowercase_product_name =
      parsed_startup_options_->GetLowercaseProductName();

  if (args.empty()) {
    blaze_util::StringPrintf(error,
                             "Unable to split command line, args is empty");
    return nullptr;
  }

  const string path_to_binary(args[0]);

  // Process the startup options.
  vector<string> startup_args;
  vector<string>::size_type i = 1;
  while (i < args.size() && IsArg(args[i])) {
    const string current_arg(args[i]);
    // If the current argument is a valid nullary startup option such as
    // --master_bazelrc or --nomaster_bazelrc proceed to examine the next
    // argument.
    if (parsed_startup_options_->IsNullary(current_arg)) {
      startup_args.push_back(current_arg);
      i++;
    } else if (parsed_startup_options_->IsUnary(current_arg)) {
      // If the current argument is a valid unary startup option such as
      // --bazelrc there are two cases to consider.

      // The option is of the form '--bazelrc=value', hence proceed to
      // examine the next argument.
      if (current_arg.find("=") != string::npos) {
        startup_args.push_back(current_arg);
        i++;
      } else {
        // Otherwise, the option is of the form '--bazelrc value', hence
        // skip the next argument and proceed to examine the argument located
        // two places after.
        if (i + 1 >= args.size()) {
          blaze_util::StringPrintf(
              error,
              "Startup option '%s' expects a value.\n"
              "Usage: '%s=somevalue' or '%s somevalue'.\n"
              "  For more info, run '%s help startup_options'.",
              current_arg.c_str(), current_arg.c_str(), current_arg.c_str(),
              lowercase_product_name.c_str());
          return nullptr;
        }
        // In this case we transform it to the form '--bazelrc=value'.
        startup_args.push_back(current_arg + string("=") + args[i + 1]);
        i += 2;
      }
    } else {
      // If the current argument is not a valid unary or nullary startup option
      // then fail.
      blaze_util::StringPrintf(
          error,
          "Unknown startup option: '%s'.\n"
          "  For more info, run '%s help startup_options'.",
          current_arg.c_str(), lowercase_product_name.c_str());
      return nullptr;
    }
  }

  // The command is the arg right after the startup options.
  if (i == args.size()) {
    return std::unique_ptr<CommandLine>(
        new CommandLine(path_to_binary, startup_args, "", {}));
  }
  const string command(args[i]);

  // The rest are the command arguments.
  const vector<string> command_args(args.begin() + i + 1, args.end());

  return std::unique_ptr<CommandLine>(
      new CommandLine(path_to_binary, startup_args, command, command_args));
}

// Return the path to the user's rc file.  If cmdLineRcFile != NULL,
// use it, dying if it is not readable.  Otherwise, return the first
// readable file called rc_basename from [workspace, $HOME]
//
// If no readable .blazerc file is found, return the empty string.
blaze_exit_code::ExitCode OptionProcessor::FindUserBlazerc(
    const char* cmdLineRcFile,
    const string& workspace,
    string* blaze_rc_file,
    string* error) {
  const string rc_basename =
      "." + parsed_startup_options_->GetLowercaseProductName() + "rc";

  if (cmdLineRcFile != NULL) {
    string rcFile = MakeAbsolute(cmdLineRcFile);
    if (!blaze_util::CanReadFile(rcFile)) {
      blaze_util::StringPrintf(error,
          "Error: Unable to read %s file '%s'.", rc_basename.c_str(),
          rcFile.c_str());
      return blaze_exit_code::BAD_ARGV;
    }
    *blaze_rc_file = rcFile;
    return blaze_exit_code::SUCCESS;
  }

  string workspaceRcFile = blaze_util::JoinPath(workspace, rc_basename);
  if (blaze_util::CanReadFile(workspaceRcFile)) {
    *blaze_rc_file = workspaceRcFile;
    return blaze_exit_code::SUCCESS;
  }

  string home = blaze::GetHomeDir();
  if (home.empty()) {
    *blaze_rc_file = "";
    return blaze_exit_code::SUCCESS;
  }

  string userRcFile = blaze_util::JoinPath(home, rc_basename);
  if (blaze_util::CanReadFile(userRcFile)) {
    *blaze_rc_file = userRcFile;
    return blaze_exit_code::SUCCESS;
  }
  *blaze_rc_file = "";
  return blaze_exit_code::SUCCESS;
}

namespace internal {
vector<string> DedupeBlazercPaths(const vector<string>& paths) {
  set<string> canonical_paths;
  vector<string> result;
  for (const string& path : paths) {
    const string canonical_path = blaze_util::MakeCanonical(path.c_str());
    if (canonical_path.empty()) {
      // MakeCanonical returns an empty string when it fails. We ignore this
      // failure since blazerc paths may point to invalid locations.
    } else if (canonical_paths.find(canonical_path) == canonical_paths.end()) {
      result.push_back(path);
      canonical_paths.insert(canonical_path);
    }
  }
  return result;
}
}  // namespace internal

// Parses the arguments provided in args using the workspace path and the
// current working directory (cwd) and stores the results.
blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    const vector<string>& args,
    const string& workspace,
    const string& cwd,
    string* error) {
  assert(!initialized_);
  initialized_ = true;

  args_ = args;
  std::unique_ptr<CommandLine> cmdLine = SplitCommandLine(args, error);
  if (cmdLine == nullptr) {
    return blaze_exit_code::BAD_ARGV;
  }

  const char* blazerc = SearchUnaryOption(cmdLine->startup_args, "--blazerc");
  if (blazerc == NULL) {
    blazerc = SearchUnaryOption(cmdLine->startup_args, "--bazelrc");
  }

  bool use_master_blazerc = true;
  if (SearchNullaryOption(cmdLine->startup_args, "--nomaster_blazerc") ||
      SearchNullaryOption(cmdLine->startup_args, "--nomaster_bazelrc")) {
    use_master_blazerc = false;
  }

  // Use the workspace path, the current working directory, the path to the
  // blaze binary and the startup args to determine the list of possible
  // paths to the rc files. This list may contain duplicates.
  vector<string> candidate_blazerc_paths;
  if (use_master_blazerc) {
    workspace_layout_->FindCandidateBlazercPaths(
        workspace, cwd, cmdLine->path_to_binary, cmdLine->startup_args,
        &candidate_blazerc_paths);
  }

  string user_blazerc_path;
  blaze_exit_code::ExitCode find_blazerc_exit_code = FindUserBlazerc(
      blazerc, workspace, &user_blazerc_path, error);
  if (find_blazerc_exit_code != blaze_exit_code::SUCCESS) {
    return find_blazerc_exit_code;
  }

  vector<string> deduped_blazerc_paths =
      internal::DedupeBlazercPaths(candidate_blazerc_paths);
  // TODO(b/37731193): Decide whether the user blazerc should be included in
  // the deduplication process. If so then we need to handle all cases
  // (e.g. user rc coming from process substitution).
  deduped_blazerc_paths.push_back(user_blazerc_path);

  for (const auto& blazerc_path : deduped_blazerc_paths) {
    if (!blazerc_path.empty()) {
      blazercs_.push_back(new RcFile(blazerc_path, blazercs_.size()));
      blaze_exit_code::ExitCode parse_exit_code =
          blazercs_.back()->Parse(workspace, workspace_layout_, &blazercs_,
                                  &rcoptions_, error);
      if (parse_exit_code != blaze_exit_code::SUCCESS) {
        return parse_exit_code;
      }
    }
  }

  blaze_exit_code::ExitCode parse_startup_options_exit_code =
      ParseStartupOptions(error);
  if (parse_startup_options_exit_code != blaze_exit_code::SUCCESS) {
    return parse_startup_options_exit_code;
  }

  // Once we're done with startup options the next arg is the command.
  if (startup_args_ + 1 >= args.size()) {
    command_ = "";
    return blaze_exit_code::SUCCESS;
  }
  command_ = args[startup_args_ + 1];

  AddRcfileArgsAndOptions(cwd);

  // The rest of the args are the command options.
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
      const string& blazerc = blazercs_[option.rcfile_index()]->Filename();
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
        const string& blazerc = blazercs_[option.rcfile_index()]->Filename();
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

#if defined(COMPILER_MSVC)
static void PreprocessEnvString(string* env_str) {
  static std::set<string> vars_to_uppercase = {"PATH", "TMP", "TEMP", "TEMPDIR",
                                               "SYSTEMROOT"};

  int pos = env_str->find_first_of('=');
  if (pos == string::npos) return;

  string name = env_str->substr(0, pos);
  // We do not care about locale. All variable names are ASCII.
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (vars_to_uppercase.find(name) != vars_to_uppercase.end()) {
    env_str->assign(name + "=" + env_str->substr(pos + 1));
  }
}

#elif defined(__CYGWIN__)  // not defined(COMPILER_MSVC)

static void PreprocessEnvString(string* env_str) {
  int pos = env_str->find_first_of('=');
  if (pos == string::npos) return;
  string name = env_str->substr(0, pos);
  if (name == "PATH") {
    env_str->assign("PATH=" + ConvertPathList(env_str->substr(pos + 1)));
  } else if (name == "TMP") {
    // A valid Windows path "c:/foo" is also a valid Unix path list of
    // ["c", "/foo"] so must use ConvertPath here. See GitHub issue #1684.
    env_str->assign("TMP=" + ConvertPath(env_str->substr(pos + 1)));
  }
}

#else  // Non-Windows platforms.

static void PreprocessEnvString(const string* env_str) {
  // do nothing.
}
#endif  // defined(COMPILER_MSVC)

// Appends the command and arguments from argc/argv to the end of arg_vector,
// and also splices in some additional terminal and environment options between
// the command and the arguments. NB: Keep the options added here in sync with
// BlazeCommandDispatcher.INTERNAL_COMMAND_OPTIONS!
void OptionProcessor::AddRcfileArgsAndOptions(const string& cwd) {
  // Provide terminal options as coming from the least important rc file.
  command_arguments_.push_back("--rc_source=client");
  command_arguments_.push_back("--default_override=0:common=--isatty=" +
                               ToString(IsStandardTerminal()));
  command_arguments_.push_back(
      "--default_override=0:common=--terminal_columns=" +
      ToString(GetTerminalColumns()));

  // Push the options mapping .blazerc numbers to filenames.
  for (int i_blazerc = 0; i_blazerc < blazercs_.size(); i_blazerc++) {
    const RcFile* blazerc = blazercs_[i_blazerc];
    command_arguments_.push_back("--rc_source=" +
                                 blaze::ConvertPath(blazerc->Filename()));
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
          "--default_override=" + ToString(rcoption.rcfile_index() + 1) + ":"
          + it->first + "=" + rcoption.option());
    }
  }

  // Pass the client environment to the server.
  for (char** env = environ; *env != NULL; env++) {
    string env_str(*env);
    PreprocessEnvString(&env_str);
    command_arguments_.push_back("--client_env=" + env_str);
  }
  command_arguments_.push_back("--client_cwd=" + blaze::ConvertPath(cwd));

  if (IsEmacsTerminal()) {
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

StartupOptions* OptionProcessor::GetParsedStartupOptions() const {
  return parsed_startup_options_.get();
}

OptionProcessor::~OptionProcessor() {
  for (auto it : blazercs_) {
    delete it;
  }
}

}  // namespace blaze
