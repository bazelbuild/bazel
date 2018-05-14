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

blaze_exit_code::ExitCode OptionProcessor::FindUserBlazerc(
    const char* cmd_line_rc_file, const string& workspace,
    string* user_blazerc_file, string* error) const {
  const string rc_basename =
      "." + parsed_startup_options_->GetLowercaseProductName() + "rc";

  if (cmd_line_rc_file != nullptr) {
    string rcFile = MakeAbsolute(cmd_line_rc_file);
    if (!blaze_util::CanReadFile(rcFile)) {
      blaze_util::StringPrintf(error,
          "Error: Unable to read %s file '%s'.", rc_basename.c_str(),
          rcFile.c_str());
      return blaze_exit_code::BAD_ARGV;
    }
    *user_blazerc_file = rcFile;
    return blaze_exit_code::SUCCESS;
  }

  string workspaceRcFile = blaze_util::JoinPath(workspace, rc_basename);
  if (blaze_util::CanReadFile(workspaceRcFile)) {
    *user_blazerc_file = workspaceRcFile;
    return blaze_exit_code::SUCCESS;
  }

  string home = blaze::GetHomeDir();
  if (!home.empty()) {
    string userRcFile = blaze_util::JoinPath(home, rc_basename);
    if (blaze_util::CanReadFile(userRcFile)) {
      *user_blazerc_file = userRcFile;
      return blaze_exit_code::SUCCESS;
    }
  }

  BAZEL_LOG(INFO) << "User provided no rc file.";
  *user_blazerc_file = "";
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

string FindRcAlongsideBinary(const string& cwd, const string& path_to_binary) {
  const string path = blaze_util::IsAbsolute(path_to_binary)
                          ? path_to_binary
                          : blaze_util::JoinPath(cwd, path_to_binary);
  const string base = blaze_util::Basename(path_to_binary);
  const string binary_blazerc_path = path + "." + base + "rc";
  if (blaze_util::CanReadFile(binary_blazerc_path)) {
    return binary_blazerc_path;
  }
  return "";
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

// TODO(#4502) Consider simplifying result_rc_files to a vector of RcFiles, no
// unique_ptrs.
blaze_exit_code::ExitCode OptionProcessor::GetRcFiles(
    const WorkspaceLayout* workspace_layout, const std::string& workspace,
    const std::string& cwd, const CommandLine* cmd_line,
    std::vector<std::unique_ptr<RcFile>>* result_rc_files,
    std::string* error) const {
  assert(cmd_line != nullptr);
  assert(result_rc_files != nullptr);

  // Find the master bazelrcs if requested. This list may contain duplicates.
  vector<string> candidate_bazelrc_paths;
  if (SearchNullaryOption(cmd_line->startup_args, "master_bazelrc", true)) {
    const string workspace_rc =
        workspace_layout->GetWorkspaceRcPath(workspace, cmd_line->startup_args);
    const string binary_rc =
        internal::FindRcAlongsideBinary(cwd, cmd_line->path_to_binary);
    const string system_rc = FindSystemWideBlazerc();
    BAZEL_LOG(INFO)
        << "Looking for master bazelrcs in the following three paths: "
        << workspace_rc << ", " << binary_rc << ", " << system_rc;
    candidate_bazelrc_paths = {workspace_rc, binary_rc, system_rc};
  }

  string user_bazelrc_path;
  blaze_exit_code::ExitCode find_bazelrc_exit_code =
      FindUserBlazerc(SearchUnaryOption(cmd_line->startup_args, "--bazelrc"),
                      workspace, &user_bazelrc_path, error);
  if (find_bazelrc_exit_code != blaze_exit_code::SUCCESS) {
    return find_bazelrc_exit_code;
  }

  vector<string> deduped_blazerc_paths =
      internal::DedupeBlazercPaths(candidate_bazelrc_paths);
  deduped_blazerc_paths.push_back(user_bazelrc_path);

  for (const auto& bazelrc_path : deduped_blazerc_paths) {
    if (bazelrc_path.empty()) {
      continue;
    }
    std::unique_ptr<RcFile> parsed_rc;
    blaze_exit_code::ExitCode parse_rcfile_exit_code = ParseRcFile(
        workspace_layout, workspace, bazelrc_path, &parsed_rc, error);
    if (parse_rcfile_exit_code != blaze_exit_code::SUCCESS) {
      return parse_rcfile_exit_code;
    }
    result_rc_files->push_back(std::move(parsed_rc));
  }

  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode ParseRcFile(const WorkspaceLayout* workspace_layout,
                                      const std::string& workspace,
                                      const std::string& rc_file_path,
                                      std::unique_ptr<RcFile>* result_rc_file,
                                      std::string* error) {
  assert(!rc_file_path.empty());
  assert(result_rc_file != nullptr);

  RcFile::ParseError parse_error;
  std::unique_ptr<RcFile> parsed_file = RcFile::Parse(
      rc_file_path, workspace_layout, workspace, &parse_error, error);
  if (parsed_file == nullptr) {
    return internal::ParseErrorToExitCode(parse_error);
  }
  *result_rc_file = std::move(parsed_file);
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    const vector<string>& args, const string& workspace, const string& cwd,
    string* error) {
  cmd_line_ = SplitCommandLine(args, error);
  if (cmd_line_ == nullptr) {
    return blaze_exit_code::BAD_ARGV;
  }

  // Read the rc files. This depends on the startup options in argv since these
  // may contain rc-modifying options. For all other options, the precedence of
  // options will be rc first, then command line options, though, despite this
  // exception.
  std::vector<std::unique_ptr<RcFile>> rc_files;
  const blaze_exit_code::ExitCode rc_parsing_exit_code = GetRcFiles(
      workspace_layout_, workspace, cwd, cmd_line_.get(), &rc_files, error);
  if (rc_parsing_exit_code != blaze_exit_code::SUCCESS) {
    return rc_parsing_exit_code;
  }

  // Parse the startup options in the correct priority order.
  const blaze_exit_code::ExitCode parse_startup_options_exit_code =
      ParseStartupOptions(rc_files, error);
  if (parse_startup_options_exit_code != blaze_exit_code::SUCCESS) {
    return parse_startup_options_exit_code;
  }

  blazerc_and_env_command_args_ =
      GetBlazercAndEnvCommandArgs(cwd, rc_files, GetProcessedEnv());
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
    const std::vector<std::unique_ptr<RcFile>> &rc_files,
    std::string *error) {
  // Rc files can import other files at any point, and these imported rcs are
  // expanded in place. Here, we isolate just the startup options but keep the
  // file they came from attached for the option_sources tracking and for
  // sending to the server.
  std::vector<RcStartupFlag> rcstartup_flags;

  for (const auto& blazerc : rc_files) {
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
