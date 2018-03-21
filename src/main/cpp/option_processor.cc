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
#include <sstream>
#include <utility>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"

// On OSX, there apparently is no header that defines this.
#ifndef environ
extern char **environ;
#endif

namespace blaze {

using std::map;
using std::set;
using std::string;
using std::vector;

constexpr char WorkspaceLayout::WorkspacePrefix[];
static std::vector<std::string> GetProcessedEnv();

OptionProcessor::OptionProcessor(
    const WorkspaceLayout* workspace_layout,
    std::unique_ptr<StartupOptions> default_startup_options)
    : workspace_layout_(workspace_layout),
      parsed_startup_options_(std::move(default_startup_options)) {
}

std::unique_ptr<CommandLine> OptionProcessor::SplitCommandLine(
    const vector<string>& args, string* error) const {
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

blaze_exit_code::ExitCode ParseErrorToExitCode(RcFile::ParseError parse_error) {
  switch (parse_error) {
    case RcFile::ParseError::NONE:
      return blaze_exit_code::SUCCESS;
    case RcFile::ParseError::UNREADABLE_FILE:
      // We check readability before parsing, so this is unexpected.
      return blaze_exit_code::INTERNAL_ERROR;
    case RcFile::ParseError::INVALID_FORMAT:
    case RcFile::ParseError::IMPORT_LOOP:
      return blaze_exit_code::BAD_ARGV;
    default:
      return blaze_exit_code::INTERNAL_ERROR;
  }
}

}  // namespace internal

// Parses the arguments provided in args using the workspace path and the
// current working directory (cwd) and stores the results.
blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    const vector<string>& args,
    const string& workspace,
    const string& cwd,
    string* error) {

  cmd_line_ = SplitCommandLine(args, error);
  if (cmd_line_ == nullptr) {
    return blaze_exit_code::BAD_ARGV;
  }
  const char* blazerc = SearchUnaryOption(cmd_line_->startup_args, "--blazerc");
  if (blazerc == NULL) {
    blazerc = SearchUnaryOption(cmd_line_->startup_args, "--bazelrc");
  }

  bool use_master_blazerc = true;
  if (!SearchNullaryOption(cmd_line_->startup_args, "master_blazerc", true) ||
      !SearchNullaryOption(cmd_line_->startup_args, "master_bazelrc", true)) {
    use_master_blazerc = false;
  }

  // Use the workspace path, the current working directory, the path to the
  // blaze binary and the startup args to determine the list of possible
  // paths to the rc files. This list may contain duplicates.
  vector<string> candidate_blazerc_paths;
  if (use_master_blazerc) {
    candidate_blazerc_paths =
        workspace_layout_->FindCandidateBlazercPaths(
            workspace, cwd, cmd_line_->path_to_binary, cmd_line_->startup_args);
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
      RcFile::ParseError parse_error;
      auto rcfile = RcFile::Parse(blazerc_path, workspace_layout_, workspace,
                                  &parse_error, error);
      if (rcfile == nullptr) {
        return internal::ParseErrorToExitCode(parse_error);
      }
      blazercs_.push_back(std::move(rcfile));
    }
  }

  blaze_exit_code::ExitCode parse_startup_options_exit_code =
      ParseStartupOptions(error);
  if (parse_startup_options_exit_code != blaze_exit_code::SUCCESS) {
    return parse_startup_options_exit_code;
  }

  blazerc_and_env_command_args_ =
      GetBlazercAndEnvCommandArgs(cwd, blazercs_, GetProcessedEnv());
  return blaze_exit_code::SUCCESS;
}

static void PrintStartupOptions(const std::string& source,
                                const std::vector<std::string>& options) {
  if (!source.empty()) {
    std::string startup_args;
    blaze_util::JoinStrings(options, ' ', &startup_args);
    fprintf(stderr, "INFO: Reading 'startup' options from %s: %s\n",
            source.c_str(), startup_args.c_str());
  }
}

void OptionProcessor::PrintStartupOptionsProvenanceMessage() const {
  StartupOptions* parsed_startup_options = GetParsedStartupOptions();

  // Print the startup flags in the order they are parsed, to keep the
  // precendence clear. In order to minimize the number of lines of output in
  // the terminal, group sequential flags by origin. Note that an rc file may
  // turn up multiple times in this list, if, for example, it imports another
  // rc file and contains startup options on either side of the import
  // statement. This is done intentionally to make option priority clear.
  std::string command_line_source;
  std::string& most_recent_blazerc = command_line_source;
  std::vector<std::string> accumulated_options;
  for (auto flag : parsed_startup_options->original_startup_options_) {
    if (flag.source == most_recent_blazerc) {
      accumulated_options.push_back(flag.value);
    } else {
      PrintStartupOptions(most_recent_blazerc, accumulated_options);
      // Start accumulating again.
      accumulated_options.clear();
      accumulated_options.push_back(flag.value);
      most_recent_blazerc = flag.source;
    }
  }
  // Don't forget to print out the last ones.
  PrintStartupOptions(most_recent_blazerc, accumulated_options);
}

blaze_exit_code::ExitCode OptionProcessor::ParseStartupOptions(
    std::string *error) {
  // Rc files can import other files at any point, and these imported rcs are
  // expanded in place. The effective ordering of rc flags is stored in
  // rcoptions_ and should be processed in that order. Here, we isolate just the
  // startup options, but keep the file they came from attached for the
  // option_sources tracking and for sending to the server.
  std::vector<RcStartupFlag> rcstartup_flags;

  for (const auto& blazerc : blazercs_) {
    const auto iter = blazerc->options().find("startup");
    if (iter == blazerc->options().end()) continue;

    for (const RcOption& option : iter->second) {
      rcstartup_flags.push_back({*option.source_path, option.option});
    }
  }

  for (const std::string& arg : cmd_line_->startup_args) {
    if (!IsArg(arg)) {
      break;
    }
    rcstartup_flags.push_back(RcStartupFlag("", arg));
  }

  return parsed_startup_options_->ProcessArgs(rcstartup_flags, error);
}

static bool IsValidEnvName(const char* p) {
#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
  for (; *p && *p != '='; ++p) {
    if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
          (*p >= '0' && *p <= '9') || *p == '_')) {
      return false;
    }
  }
#endif
  return true;
}

#if defined(COMPILER_MSVC)
static void PreprocessEnvString(string* env_str) {
  static constexpr const char* vars_to_uppercase[] = {"PATH", "SYSTEMROOT",
                                                      "TEMP", "TEMPDIR", "TMP"};

  int pos = env_str->find_first_of('=');
  if (pos == string::npos) return;

  string name = env_str->substr(0, pos);
  // We do not care about locale. All variable names are ASCII.
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (std::find(std::begin(vars_to_uppercase), std::end(vars_to_uppercase),
                name) != std::end(vars_to_uppercase)) {
    env_str->assign(name + "=" + env_str->substr(pos + 1));
  }
}

#elif defined(__CYGWIN__)  // not defined(COMPILER_MSVC)

static void PreprocessEnvString(string* env_str) {
  int pos = env_str->find_first_of('=');
  if (pos == string::npos) return;
  string name = env_str->substr(0, pos);
  if (name == "PATH") {
    env_str->assign("PATH=" + env_str->substr(pos + 1));
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

static std::vector<std::string> GetProcessedEnv() {
  std::vector<std::string> processed_env;
  for (char** env = environ; *env != NULL; env++) {
    string env_str(*env);
    if (IsValidEnvName(*env)) {
      PreprocessEnvString(&env_str);
      processed_env.push_back(std::move(env_str));
    }
  }
  return processed_env;
}

// IMPORTANT: The options added here do not come from the user. In order for
// their source to be correctly tracked, the options must either be passed
// as --default_override=0, 0 being "client", or must be listed in
// BlazeOptionHandler.INTERNAL_COMMAND_OPTIONS!
std::vector<std::string> OptionProcessor::GetBlazercAndEnvCommandArgs(
    const std::string& cwd,
    const std::vector<std::unique_ptr<RcFile>>& blazercs,
    const std::vector<std::string>& env) {
  // Provide terminal options as coming from the least important rc file.
  std::vector<std::string> result = {
      "--rc_source=client",
      "--default_override=0:common=--isatty=" + ToString(IsStandardTerminal()),
      "--default_override=0:common=--terminal_columns=" +
          ToString(GetTerminalColumns())};
  if (IsEmacsTerminal()) {
    result.push_back("--default_override=0:common=--emacs");
  }

  EnsurePythonPathOption(&result);

  // Map .blazerc numbers to filenames. The indexes here start at 1 because #0
  // is reserved the "client" options created by this function.
  int cur_index = 1;
  std::map<std::string, int> rcfile_indexes;
  for (const auto& blazerc : blazercs) {
    for (const std::string& source_path : blazerc->sources()) {
      // Deduplicate the rc_source list because the same file might be included
      // from multiple places.
      if (rcfile_indexes.find(source_path) != rcfile_indexes.end()) continue;

      result.push_back("--rc_source=" + blaze::ConvertPath(source_path));
      rcfile_indexes[source_path] = cur_index;
      cur_index++;
    }
  }

  // Add RcOptions as default_overrides.
  for (const auto& blazerc : blazercs) {
    for (const auto& command_options : blazerc->options()) {
      const string& command = command_options.first;
      // Skip startup flags, which are already parsed by the client.
      if (command == "startup") continue;

      for (const RcOption& rcoption : command_options.second) {
        std::ostringstream oss;
        oss << "--default_override=" << rcfile_indexes[*rcoption.source_path]
            << ':' << command << '=' << rcoption.option;
        result.push_back(oss.str());
      }
    }
  }

  // Pass the client environment to the server.
  for (const string& env_var : env) {
    result.push_back("--client_env=" + env_var);
  }
  result.push_back("--client_cwd=" + blaze::ConvertPath(cwd));
  return result;
}

std::vector<std::string> OptionProcessor::GetCommandArguments() const {
  assert(cmd_line_ != nullptr);
  // When the user didn't specify a command, the server expects the command
  // arguments to be empty in order to display the help message.
  if (cmd_line_->command.empty()) {
    return {};
  }

  std::vector<std::string> command_args = blazerc_and_env_command_args_;
  command_args.insert(command_args.end(),
                      cmd_line_->command_args.begin(),
                      cmd_line_->command_args.end());
  return command_args;
}

std::vector<std::string> OptionProcessor::GetExplicitCommandArguments() const {
  assert(cmd_line_ != nullptr);
  return cmd_line_->command_args;
}

std::string OptionProcessor::GetCommand() const {
  assert(cmd_line_ != nullptr);
  return cmd_line_->command;
}

StartupOptions* OptionProcessor::GetParsedStartupOptions() const {
  assert(parsed_startup_options_ != NULL);
  return parsed_startup_options_.get();
}

}  // namespace blaze
