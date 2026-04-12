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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor-internal.h"
#include "src/main/cpp/sem_ver.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "src/main/cpp/rc_file.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/exit_code.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace blaze {

using std::set;
using std::string;
using std::vector;

static constexpr const char* kRcBasename = ".bazelrc";

OptionProcessor::OptionProcessor(
    const WorkspaceLayout* workspace_layout,
    std::unique_ptr<StartupOptions> default_startup_options)
    : workspace_layout_(workspace_layout),
      startup_options_(std::move(default_startup_options)),
      parse_options_called_(false),
      system_bazelrc_path_(BAZEL_SYSTEM_BAZELRC_PATH) {}

OptionProcessor::OptionProcessor(
    const WorkspaceLayout* workspace_layout,
    std::unique_ptr<StartupOptions> default_startup_options,
    const std::string& system_bazelrc_path)
    : workspace_layout_(workspace_layout),
      startup_options_(std::move(default_startup_options)),
      parse_options_called_(false),
      system_bazelrc_path_(system_bazelrc_path) {}

std::string OptionProcessor::GetLowercaseProductName() const {
  return startup_options_->GetLowercaseProductName();
}

std::unique_ptr<CommandLine> OptionProcessor::SplitCommandLine(
    vector<string> args, string* error) const {
  const string lowercase_product_name = GetLowercaseProductName();

  if (args.empty()) {
    blaze_util::StringPrintf(error,
                             "Unable to split command line, args is empty");
    return nullptr;
  }

  string& path_to_binary = args[0];

  // Process the startup options.
  vector<string> startup_args;
  vector<string>::size_type i = 1;
  while (i < args.size() && IsArg(args[i])) {
    string& current_arg = args[i];

    bool is_nullary;
    if (!startup_options_->MaybeCheckValidNullary(current_arg, &is_nullary,
                                                  error)) {
      return nullptr;
    }

    // If the current argument is a valid nullary startup option such as
    // --master_bazelrc or --nomaster_bazelrc proceed to examine the next
    // argument.
    if (is_nullary) {
      startup_args.push_back(current_arg);
      i++;
    } else if (startup_options_->IsUnary(current_arg)) {
      // If the current argument is a valid unary startup option such as
      // --bazelrc there are two cases to consider.

      // The option is of the form '--bazelrc=value', hence proceed to
      // examine the next argument.
      if (current_arg.find('=') != string::npos) {
        startup_args.push_back(std::move(current_arg));
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
        startup_args.push_back(std::move(current_arg) + "=" +
                               std::move(args[i + 1]));
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
    return std::unique_ptr<CommandLine>(new CommandLine(
        std::move(path_to_binary), std::move(startup_args), "", {}));
  }
  string& command = args[i];
  // Distinguish an empty command from the case of no command above.
  if (command.empty()) {
    blaze_util::StringPrintf(error,
                             "Command cannot be the empty string.\n"
                             "  For more info, run '%s help'.",
                             lowercase_product_name.c_str());
    return nullptr;
  }

  // The rest are the command arguments.
  vector<string> command_args(std::make_move_iterator(args.begin() + i + 1),
                              std::make_move_iterator(args.end()));

  return std::unique_ptr<CommandLine>(
      new CommandLine(std::move(path_to_binary), std::move(startup_args),
                      std::move(command), std::move(command_args)));
}

namespace internal {

// Deduplicates the given paths based on their canonical form.
// Computes the canonical form using blaze_util::MakeCanonical.
// Returns the unique paths in their original form (not the canonical one).
std::vector<std::string> DedupeBlazercPaths(
    const std::vector<std::string>& paths) {
  std::set<std::string> canonical_paths;
  std::vector<std::string> result;
  for (const std::string& path : paths) {
    const std::string canonical_path = blaze_util::MakeCanonical(path.c_str());
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
      // We check readability before parsing, so this is unexpected for
      // top-level rc files, so is an INTERNAL_ERROR. It can happen for imported
      // files, however, which should be BAD_ARGV, but we don't currently
      // differentiate.
      // TODO(bazel-team): fix RcFile to reclassify unreadable files that were
      // read from a recursive call due to a malformed import.
      return blaze_exit_code::INTERNAL_ERROR;
    case RcFile::ParseError::INVALID_FORMAT:
    case RcFile::ParseError::IMPORT_LOOP:
    case RcFile::ParseError::IMPORT_DEPTH_EXCEEDED:
      return blaze_exit_code::BAD_ARGV;
    default:
      return blaze_exit_code::INTERNAL_ERROR;
  }
}

void WarnAboutDuplicateRcFiles(const std::set<std::string>& read_files,
                               const std::vector<std::string>& loaded_rcs) {
  // The first rc file in the queue is the top-level one, the one that would
  // have imported all the others in the queue. The top-level rc is one of the
  // default locations (system, workspace, home) or the explicit path passed by
  // --bazelrc.
  const std::string& top_level_rc = loaded_rcs.front();

  const std::set<std::string> unique_loaded_rcs(loaded_rcs.begin(),
                                                loaded_rcs.end());
  // First check if each of the newly loaded rc files was already read.
  for (const std::string& loaded_rc : unique_loaded_rcs) {
    if (read_files.count(loaded_rc) > 0) {
      if (loaded_rc == top_level_rc) {
        BAZEL_LOG(WARNING)
            << "Duplicate rc file: " << loaded_rc
            << " is read multiple times, it is a standard rc file location "
               "but must have been unnecessarily imported earlier.";
      } else {
        BAZEL_LOG(WARNING)
            << "Duplicate rc file: " << loaded_rc
            << " is read multiple times, most recently imported from "
            << top_level_rc;
      }
    }
    // Now check if the top-level rc file loads up its own duplicates (it can't
    // be a cycle, since that would be an error and we would have already
    // exited, but it could have a diamond dependency of some sort.)
    if (std::count(loaded_rcs.begin(), loaded_rcs.end(), loaded_rc) > 1) {
      BAZEL_LOG(WARNING) << "Duplicate rc file: " << loaded_rc
                         << " is imported multiple times from " << top_level_rc;
    }
  }
}

blaze_exit_code::ExitCode GetAndValidateExistingRcFiles(
    const std::vector<std::string>& input_rc_files,
    std::vector<std::string>* result_rc_files) {
  for (const auto& rc_file : input_rc_files) {
    string absolute_rc = blaze::AbsolutePathFromFlag(rc_file);
    if (!blaze_util::CanReadFile(absolute_rc)) {
      BAZEL_LOG(ERROR) << "Error: Unable to read .bazelrc file '" << absolute_rc
                       << "'.";
      return blaze_exit_code::BAD_ARGV;
    }
    result_rc_files->push_back(absolute_rc);
  }
  return blaze_exit_code::SUCCESS;
}
}  // namespace internal

// TODO(#4502) Consider simplifying result_rc_files to a vector of RcFiles, no
// unique_ptrs.
blaze_exit_code::ExitCode OptionProcessor::GetRcFiles(
    const WorkspaceLayout* workspace_layout, const std::string& workspace,
    const std::string& cwd, const CommandLine* cmd_line,
    std::vector<std::unique_ptr<RcFile>>* result_rc_files, std::string* error,
    int max_import_depth) const {
  assert(cmd_line != nullptr);
  assert(result_rc_files != nullptr);

  std::vector<std::string> rc_files;

  // Get the system rc (unless --nosystem_rc).
  if (SearchNullaryOption(cmd_line->startup_args, "system_rc", true)) {
    // MakeAbsoluteAndResolveEnvvars will standardize the form of the
    // provided path. This also means we accept relative paths, which is
    // is convenient for testing.
    const std::string system_rc =
        blaze_util::MakeAbsoluteAndResolveEnvvars(system_bazelrc_path_);
    rc_files.push_back(system_rc);
  }

  // Get the workspace rc: %workspace%/.bazelrc (unless --noworkspace_rc), but
  // only if we are in a workspace: invoking commands like "help" from outside
  // a workspace should work.
  if (!workspace.empty() &&
      SearchNullaryOption(cmd_line->startup_args, "workspace_rc", true)) {
    const std::string workspaceRcFile =
        blaze_util::JoinPath(workspace, kRcBasename);
    rc_files.push_back(workspaceRcFile);
  }

  // Get the user rc: $HOME/.bazelrc (unless --nohome_rc)
  if (SearchNullaryOption(cmd_line->startup_args, "home_rc", true)) {
    const std::string home = blaze::GetHomeDir();
    if (!home.empty()) {
      rc_files.push_back(blaze_util::JoinPath(home, kRcBasename));
    }
  }

  // Get the env var provided rc.
  const std::vector<std::string> env_var_rc_files =
      blaze_util::Split(GetEnv("BAZELRC"), ',');
  blaze_exit_code::ExitCode env_exit_code =
      internal::GetAndValidateExistingRcFiles(env_var_rc_files, &rc_files);
  if (env_exit_code != blaze_exit_code::SUCCESS) {
    return env_exit_code;
  }

  // Get the command-line provided rc, passed as --bazelrc or nothing if the
  // flag is absent.
  vector<std::string> cmd_line_rc_files =
      GetAllUnaryOptionValues(cmd_line->startup_args, "--bazelrc", "/dev/null");
  blaze_exit_code::ExitCode cmd_exit_code =
      internal::GetAndValidateExistingRcFiles(cmd_line_rc_files, &rc_files);
  if (cmd_exit_code != blaze_exit_code::SUCCESS) {
    return cmd_exit_code;
  }

  // Log which files we're looking for before removing duplicates and
  // non-existent files, so that this can serve to debug why a certain file is
  // not being read. The final files which are read will be logged as they are
  // parsed, and can be found using --announce_rc.
  std::string joined_rcs;
  blaze_util::JoinStrings(rc_files, ',', &joined_rcs);
  BAZEL_LOG(INFO) << "Looking for the following rc files: " << joined_rcs;

  // It's possible that workspace == home, that files are symlinks for each
  // other, or that the --bazelrc flag is a duplicate. Dedupe them to minimize
  // the likelihood of repeated options. Since bazelrcs can include one another,
  // this isn't sufficient to prevent duplicate options, so we also warn if we
  // discover duplicate loads later. This also has the effect of removing paths
  // that don't point to real files.
  rc_files = internal::DedupeBlazercPaths(rc_files);

  const std::optional<SemVer>& sem_ver = SemVer::Parse(build_label_);
  std::set<std::string> read_files_canonical_paths;
  // Parse these potential files, in priority order;
  for (const std::string& top_level_bazelrc_path : rc_files) {
    std::unique_ptr<RcFile> parsed_rc;
    blaze_exit_code::ExitCode parse_rcfile_exit_code =
        ParseRcFile(workspace_layout, workspace, top_level_bazelrc_path,
                    build_label_, sem_ver, &parsed_rc, error, max_import_depth);
    if (parse_rcfile_exit_code != blaze_exit_code::SUCCESS) {
      return parse_rcfile_exit_code;
    }

    // Check that none of the rc files loaded this time are duplicate.
    const auto& sources = parsed_rc->canonical_source_paths();
    internal::WarnAboutDuplicateRcFiles(read_files_canonical_paths, sources);
    read_files_canonical_paths.insert(sources.begin(), sources.end());

    result_rc_files->push_back(std::move(parsed_rc));
  }
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode ParseRcFile(const WorkspaceLayout* workspace_layout,
                                      const std::string& workspace,
                                      const std::string& rc_file_path,
                                      const std::string& build_label,
                                      const std::optional<SemVer>& sem_ver,
                                      std::unique_ptr<RcFile>* result_rc_file,
                                      std::string* error,
                                      int max_import_depth) {
  assert(!rc_file_path.empty());
  assert(result_rc_file != nullptr);

  RcFile::ParseError parse_error;
  std::unique_ptr<RcFile> parsed_file =
      RcFile::Parse(rc_file_path, workspace_layout, workspace, build_label,
                    sem_ver, &parse_error, error, max_import_depth);
  if (parsed_file == nullptr) {
    return internal::ParseErrorToExitCode(parse_error);
  }
  *result_rc_file = std::move(parsed_file);
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode ParseRcFile(const WorkspaceLayout* workspace_layout,
                                      const std::string& workspace,
                                      const std::string& rc_file_path,
                                      std::unique_ptr<RcFile>* result_rc_file,
                                      std::string* error,
                                      int max_import_depth) {
  return ParseRcFile(workspace_layout, workspace, rc_file_path,
                     /*build_label=*/"", /*sem_ver=*/std::nullopt,
                     result_rc_file, error, max_import_depth);
}

blaze_exit_code::ExitCode OptionProcessor::ParseOptions(
    const vector<string>& args, const string& workspace, const string& cwd,
    string* error, int max_import_depth) {
  assert(!parse_options_called_);
  parse_options_called_ = true;
  cmd_line_ = SplitCommandLine(args, error);
  if (cmd_line_ == nullptr) {
    return blaze_exit_code::BAD_ARGV;
  }

  // Read the rc files, unless --ignore_all_rc_files was provided on the command
  // line. This depends on the startup options in argv since these may contain
  // other rc-modifying options. For all other options, the precedence of
  // options will be rc first, then command line options, though, despite this
  // exception.
  std::vector<std::unique_ptr<RcFile>> rc_files;
  if (!SearchNullaryOption(cmd_line_->startup_args, "ignore_all_rc_files",
                           false)) {
    const blaze_exit_code::ExitCode rc_parsing_exit_code =
        GetRcFiles(workspace_layout_, workspace, cwd, cmd_line_.get(),
                   &rc_files, error, max_import_depth);
    if (rc_parsing_exit_code != blaze_exit_code::SUCCESS) {
      return rc_parsing_exit_code;
    }
  }

  // The helpers expect regular pointers, not unique_ptrs.
  std::vector<RcFile*> rc_file_ptrs;
  rc_file_ptrs.reserve(rc_files.size());
  for (auto& rc_file : rc_files) { rc_file_ptrs.push_back(rc_file.get()); }

  // Parse the startup options in the correct priority order.
  const blaze_exit_code::ExitCode parse_startup_options_exit_code =
      ParseStartupOptions(rc_file_ptrs, error);
  if (parse_startup_options_exit_code != blaze_exit_code::SUCCESS) {
    return parse_startup_options_exit_code;
  }

  parsed_blazercs_ = GetBlazercOptions(cwd, rc_file_ptrs);
  blazerc_and_env_command_args_ = GetBlazercAndEnvCommandArgs(
      cwd, parsed_blazercs_, blaze::GetProcessedEnv());
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
  // precedence clear. In order to minimize the number of lines of output in
  // the terminal, group sequential flags by origin. Note that an rc file may
  // turn up multiple times in this list, if, for example, it imports another
  // rc file and contains startup options on either side of the import
  // statement. This is done intentionally to make option priority clear.
  std::string command_line_source;
  std::string& most_recent_blazerc = command_line_source;
  std::vector<std::string> accumulated_options;
  for (const auto& flag : parsed_startup_options->original_startup_options_) {
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
    const std::vector<RcFile*> &rc_files, std::string *error) {
  // Rc files can import other files at any point, and these imported rcs are
  // expanded in place. Here, we isolate just the startup options but keep the
  // file they came from attached for the option_sources tracking and for
  // sending to the server.
  std::vector<RcStartupFlag> rcstartup_flags;

  for (const auto* blazerc : rc_files) {
    const auto iter = blazerc->options().find("startup");
    if (iter == blazerc->options().end()) continue;

    for (const RcOption& option : iter->second) {
      const std::string& source_path =
          blazerc->canonical_source_paths()[option.source_index];
      rcstartup_flags.push_back({source_path, option.option});
    }
  }

  std::string platform_config;
#if defined(__linux__)
  platform_config = "linux";
#elif defined(__APPLE__)
  platform_config = "macos";
#elif defined(_WIN32)
  platform_config = "windows";
#elif defined(__FreeBSD__)
  platform_config = "freebsd";
#elif defined(__OpenBSD__)
  platform_config = "openbsd";
#endif

  if (!platform_config.empty()) {
    for (const auto* blazerc : rc_files) {
      const auto iter = blazerc->options().find("startup:" + platform_config);
      if (iter == blazerc->options().end()) continue;

      for (const RcOption& option : iter->second) {
        const std::string& source_path =
            blazerc->canonical_source_paths()[option.source_index];
        rcstartup_flags.push_back({source_path, option.option});
      }
    }
  }

  for (const std::string& arg : cmd_line_->startup_args) {
    if (!IsArg(arg)) {
      break;
    }
    rcstartup_flags.push_back(RcStartupFlag("", arg));
  }

  return startup_options_->ProcessArgs(rcstartup_flags, error);
}

std::vector<BlazercOption> OptionProcessor::GetBlazercOptions(
    const std::string& cwd, const std::vector<RcFile*>& blazercs) {
  std::vector<BlazercOption> result;

  // Provide terminal options as least important options.
  // LINT.IfChange
  std::string terminal_rc_source = "client";
  // LINT.ThenChange(//src/main/java/com/google/devtools/common/options/GlobalRcUtils.java)
  result.push_back({terminal_rc_source, "common",
                    "--isatty=" + blaze_util::ToString(IsStandardTerminal())});
  result.push_back(
      {terminal_rc_source, "common",
       "--terminal_columns=" + blaze_util::ToString(GetTerminalColumns())});
  if (IsEmacsTerminal()) {
    result.push_back({terminal_rc_source, "common", "--emacs"});
  }

  for (const auto* blazerc : blazercs) {
    for (const auto& command_options : blazerc->options()) {
      const string& command = command_options.first;
      // Skip startup flags, which are already parsed by the client.
      if (command == "startup") continue;

      for (const RcOption& rcoption : command_options.second) {
        const std::string& source_path =
            blazerc->canonical_source_paths()[rcoption.source_index];
        result.push_back({source_path, command, rcoption.option});
      }
    }
  }

  return result;
}

// IMPORTANT: The options added here do not come from the user. In order for
// their source to be correctly tracked, the options must either be passed
// as --default_override=0, 0 being "client", or must be listed in
// BlazeOptionHandler.INTERNAL_COMMAND_OPTIONS!
std::vector<std::string> OptionProcessor::GetBlazercAndEnvCommandArgs(
    const std::string& cwd, const std::vector<BlazercOption>& blazerc_options,
    const std::vector<std::string>& env) {
  std::vector<std::string> result;
  absl::flat_hash_map<std::string, int> rcfile_index;

  // Fix first value of `--rc_source` to allow for user-defined
  // `--default_override=0` values.
  // LINT.IfChange
  result.push_back("--rc_source=client");
  rcfile_index["client"] = 0;
  // LINT.ThenChange(//src/main/java/com/google/devtools/common/options/GlobalRcUtils.java)
  int cur_index = 1;
  for (const auto& option : blazerc_options) {
    if (rcfile_index.contains(option.source_path)) continue;
    rcfile_index[option.source_path] = cur_index;
    result.push_back(absl::StrCat("--rc_source=",
                                  blaze_util::ConvertPath(option.source_path)));
    ++cur_index;
  }

  for (const auto& option : blazerc_options) {
    result.push_back(
        absl::StrCat("--default_override=", rcfile_index[option.source_path],
                     ":", option.command, "=", option.option));
  }

  // Pass the client environment to the server.
  for (const string& env_var : env) {
    result.push_back("--client_env=" + env_var);
  }
  result.push_back("--client_cwd=" + blaze_util::ConvertPath(cwd));
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
  assert(parse_options_called_);
  return startup_options_.get();
}

std::vector<BlazercOption> OptionProcessor::GetParsedBlazercOptions() const {
  assert(parse_options_called_);
  return parsed_blazercs_;
}

}  // namespace blaze
