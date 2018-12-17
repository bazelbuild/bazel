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
#include "src/main/cpp/startup_options.h"

#include <assert.h>

#include <cstdio>
#include <cstdlib>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"

namespace blaze {

using std::string;
using std::vector;

StartupFlag::~StartupFlag() {}

bool UnaryStartupFlag::NeedsParameter() const {
  return true;
}

bool UnaryStartupFlag::IsValid(const std::string &arg) const {
  // The second argument of GetUnaryOption is not relevant to determine
  // whether the option is unary or not, hence we set it to the empty string
  // by default.
  //
  // TODO(lpino): Improve GetUnaryOption to only require the arg and the
  // option we are looking for.
  return GetUnaryOption(arg.c_str(), "", ("--" + name_).c_str()) != NULL;
}

bool NullaryStartupFlag::NeedsParameter() const {
  return false;
}

bool NullaryStartupFlag::IsValid(const std::string &arg) const {
  return GetNullaryOption(arg.c_str(), ("--" + name_).c_str()) ||
      GetNullaryOption(arg.c_str(), ("--no" + name_).c_str());
}

void StartupOptions::RegisterNullaryStartupFlag(const std::string &flag_name) {
  valid_startup_flags.insert(std::unique_ptr<NullaryStartupFlag>(
      new NullaryStartupFlag(flag_name)));
}

void StartupOptions::RegisterUnaryStartupFlag(const std::string &flag_name) {
  valid_startup_flags.insert(std::unique_ptr<UnaryStartupFlag>(
      new UnaryStartupFlag(flag_name)));
}

StartupOptions::StartupOptions(const string &product_name,
                               const WorkspaceLayout *workspace_layout)
    : product_name(product_name),
      ignore_all_rc_files(false),
      deep_execroot(true),
      block_for_lock(true),
      host_jvm_debug(false),
      batch(false),
      batch_cpu_scheduling(false),
      io_nice_level(-1),
      shutdown_on_low_sys_mem(false),
      oom_more_eagerly(false),
      oom_more_eagerly_threshold(100),
      write_command_log(true),
      rotating_server_log(true),
      watchfs(false),
      fatal_event_bus_exceptions(false),
      command_port(0),
      connect_timeout_secs(30),
      invocation_policy(NULL),
      client_debug(false),
      java_logging_formatter(
          "com.google.devtools.build.lib.util.SingleLineFormatter"),
      expand_configs_in_place(true),
      digest_function(),
      idle_server_tasks(true),
      original_startup_options_(std::vector<RcStartupFlag>()),
      unlimit_coredumps(false) {
  if (blaze::IsRunningWithinTest()) {
    output_root = blaze_util::MakeAbsolute(blaze::GetEnv("TEST_TMPDIR"));
    max_idle_secs = 15;
    BAZEL_LOG(USER) << "$TEST_TMPDIR defined: output root default is '"
                    << output_root << "' and max_idle_secs default is '"
                    << max_idle_secs << "'.";
  } else {
    output_root = workspace_layout->GetOutputRoot();
    max_idle_secs = 3 * 3600;
    BAZEL_LOG(INFO) << "output root is '" << output_root
                    << "' and max_idle_secs default is '" << max_idle_secs
                    << "'.";
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  string windows_unix_root = WindowsUnixRoot(blaze::GetEnv("BAZEL_SH"));
  if (!windows_unix_root.empty()) {
    host_jvm_args.push_back(string("-Dbazel.windows_unix_root=") +
                            windows_unix_root);
  }
#endif  // defined(_WIN32) || defined(__CYGWIN__)

  const string product_name_lower = GetLowercaseProductName();
  output_user_root = blaze_util::JoinPath(
      output_root, "_" + product_name_lower + "_" + GetUserName());

  // IMPORTANT: Before modifying the statements below please contact a Bazel
  // core team member that knows the internal procedure for adding/deprecating
  // startup flags.
  RegisterNullaryStartupFlag("batch");
  RegisterNullaryStartupFlag("batch_cpu_scheduling");
  RegisterNullaryStartupFlag("block_for_lock");
  RegisterNullaryStartupFlag("client_debug");
  RegisterNullaryStartupFlag("deep_execroot");
  RegisterNullaryStartupFlag("expand_configs_in_place");
  RegisterNullaryStartupFlag("experimental_oom_more_eagerly");
  RegisterNullaryStartupFlag("fatal_event_bus_exceptions");
  RegisterNullaryStartupFlag("host_jvm_debug");
  RegisterNullaryStartupFlag("idle_server_tasks");
  RegisterNullaryStartupFlag("shutdown_on_low_sys_mem");
  RegisterNullaryStartupFlag("ignore_all_rc_files");
  RegisterNullaryStartupFlag("unlimit_coredumps");
  RegisterNullaryStartupFlag("watchfs");
  RegisterNullaryStartupFlag("write_command_log");
  RegisterNullaryStartupFlag("rotating_server_log");
  RegisterUnaryStartupFlag("command_port");
  RegisterUnaryStartupFlag("connect_timeout_secs");
  RegisterUnaryStartupFlag("digest_function");
  RegisterUnaryStartupFlag("experimental_oom_more_eagerly_threshold");
  RegisterUnaryStartupFlag("server_javabase");
  RegisterUnaryStartupFlag("host_jvm_args");
  RegisterUnaryStartupFlag("host_jvm_profile");
  RegisterUnaryStartupFlag("invocation_policy");
  RegisterUnaryStartupFlag("io_nice_level");
  RegisterUnaryStartupFlag("install_base");
  RegisterUnaryStartupFlag("max_idle_secs");
  RegisterUnaryStartupFlag("output_base");
  RegisterUnaryStartupFlag("output_user_root");
  RegisterUnaryStartupFlag("server_jvm_out");
}

StartupOptions::~StartupOptions() {}

string StartupOptions::GetLowercaseProductName() const {
  string lowercase_product_name = product_name;
  blaze_util::ToLower(&lowercase_product_name);
  return lowercase_product_name;
}

bool StartupOptions::IsNullary(const string& arg) const {
  for (const auto& flag : valid_startup_flags) {
    if (!flag->NeedsParameter() && flag->IsValid(arg)) {
      return true;
    }
  }
  return false;
}

bool StartupOptions::IsUnary(const string& arg) const {
  for (const auto& flag : valid_startup_flags) {
    if (flag->NeedsParameter() && flag->IsValid(arg)) {
      return true;
    }
  }
  return false;
}

void StartupOptions::AddExtraOptions(vector<string> *result) const {}

blaze_exit_code::ExitCode StartupOptions::ProcessArg(
      const string &argstr, const string &next_argstr, const string &rcfile,
      bool *is_space_separated, string *error) {
  // We have to parse a specific option syntax, so GNU getopts won't do.  All
  // options begin with "--" or "-". Values are given together with the option
  // delimited by '=' or in the next option.
  const char* arg = argstr.c_str();
  const char* next_arg = next_argstr.empty() ? NULL : next_argstr.c_str();
  const char* value = NULL;

  if ((value = GetUnaryOption(arg, next_arg, "--output_base")) != NULL) {
    output_base = blaze::AbsolutePathFromFlag(value);
    option_sources["output_base"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--install_base")) != NULL) {
    install_base = blaze::AbsolutePathFromFlag(value);
    option_sources["install_base"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--output_user_root")) != NULL) {
    output_user_root = blaze::AbsolutePathFromFlag(value);
    option_sources["output_user_root"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--server_jvm_out")) != NULL) {
    server_jvm_out = blaze::AbsolutePathFromFlag(value);
    option_sources["server_jvm_out"] = rcfile;
  } else if (GetNullaryOption(arg, "--deep_execroot")) {
    deep_execroot = true;
    option_sources["deep_execroot"] = rcfile;
  } else if (GetNullaryOption(arg, "--nodeep_execroot")) {
    deep_execroot = false;
    option_sources["deep_execroot"] = rcfile;
  } else if (GetNullaryOption(arg, "--block_for_lock")) {
    block_for_lock = true;
    option_sources["block_for_lock"] = rcfile;
  } else if (GetNullaryOption(arg, "--noblock_for_lock")) {
    block_for_lock = false;
    option_sources["block_for_lock"] = rcfile;
  } else if (GetNullaryOption(arg, "--host_jvm_debug")) {
    host_jvm_debug = true;
    option_sources["host_jvm_debug"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--host_jvm_profile")) !=
             NULL) {
    host_jvm_profile = value;
    option_sources["host_jvm_profile"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--server_javabase")) !=
             NULL) {
    // TODO(bazel-team): Consider examining the javabase and re-execing in case
    // of architecture mismatch.
    server_javabase_ = blaze::AbsolutePathFromFlag(value);
    option_sources["server_javabase"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--host_jvm_args")) !=
             NULL) {
    host_jvm_args.push_back(value);
    option_sources["host_jvm_args"] = rcfile;  // NB: This is incorrect
  } else if (GetNullaryOption(arg, "--ignore_all_rc_files")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --ignore_all_rc_files in an rc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    ignore_all_rc_files = true;
    option_sources["ignore_all_rc_files"] = rcfile;
  } else if (GetNullaryOption(arg, "--noignore_all_rc_files")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --noignore_all_rc_files in an rc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    ignore_all_rc_files = false;
    option_sources["ignore_all_rc_files"] = rcfile;
  } else if (GetNullaryOption(arg, "--batch")) {
    batch = true;
    option_sources["batch"] = rcfile;
  } else if (GetNullaryOption(arg, "--nobatch")) {
    batch = false;
    option_sources["batch"] = rcfile;
  } else if (GetNullaryOption(arg, "--batch_cpu_scheduling")) {
    batch_cpu_scheduling = true;
    option_sources["batch_cpu_scheduling"] = rcfile;
  } else if (GetNullaryOption(arg, "--nobatch_cpu_scheduling")) {
    batch_cpu_scheduling = false;
    option_sources["batch_cpu_scheduling"] = rcfile;
  } else if (GetNullaryOption(arg, "--fatal_event_bus_exceptions")) {
    fatal_event_bus_exceptions = true;
    option_sources["fatal_event_bus_exceptions"] = rcfile;
  } else if (GetNullaryOption(arg, "--nofatal_event_bus_exceptions")) {
    fatal_event_bus_exceptions = false;
    option_sources["fatal_event_bus_exceptions"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--io_nice_level")) !=
             NULL) {
    if (!blaze_util::safe_strto32(value, &io_nice_level) ||
        io_nice_level > 7) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --io_nice_level: '%s'. Must not exceed 7.",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["io_nice_level"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--max_idle_secs")) !=
             NULL) {
    if (!blaze_util::safe_strto32(value, &max_idle_secs) ||
        max_idle_secs < 0) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --max_idle_secs: '%s'.", value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["max_idle_secs"] = rcfile;
  } else if (GetNullaryOption(arg, "--shutdown_on_low_sys_mem")) {
    shutdown_on_low_sys_mem = true;
    option_sources["shutdown_on_low_sys_mem"] = rcfile;
  } else if (GetNullaryOption(arg, "--noshutdown_on_low_sys_mem")) {
    shutdown_on_low_sys_mem = false;
    option_sources["shutdown_on_low_sys_mem"] = rcfile;
  } else if (GetNullaryOption(arg, "--experimental_oom_more_eagerly")) {
    oom_more_eagerly = true;
    option_sources["experimental_oom_more_eagerly"] = rcfile;
  } else if (GetNullaryOption(arg, "--noexperimental_oom_more_eagerly")) {
    oom_more_eagerly = false;
    option_sources["experimental_oom_more_eagerly"] = rcfile;
  } else if ((value = GetUnaryOption(
                  arg, next_arg,
                  "--experimental_oom_more_eagerly_threshold")) != NULL) {
    if (!blaze_util::safe_strto32(value, &oom_more_eagerly_threshold) ||
        oom_more_eagerly_threshold < 0) {
      blaze_util::StringPrintf(error,
                               "Invalid argument to "
                               "--experimental_oom_more_eagerly_threshold: "
                               "'%s'.",
                               value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["experimental_oom_more_eagerly_threshold"] = rcfile;
  } else if (GetNullaryOption(arg, "--write_command_log")) {
    write_command_log = true;
    option_sources["write_command_log"] = rcfile;
  } else if (GetNullaryOption(arg, "--nowrite_command_log")) {
    write_command_log = false;
    option_sources["write_command_log"] = rcfile;
  } else if (GetNullaryOption(arg, "--rotating_server_log")) {
    rotating_server_log = true;
    option_sources["rotating_server_log"] = rcfile;
  } else if (GetNullaryOption(arg, "--norotating_server_log")) {
    rotating_server_log = false;
    option_sources["rotating_server_log"] = rcfile;
  } else if (GetNullaryOption(arg, "--watchfs")) {
    watchfs = true;
    option_sources["watchfs"] = rcfile;
  } else if (GetNullaryOption(arg, "--nowatchfs")) {
    watchfs = false;
    option_sources["watchfs"] = rcfile;
  } else if (GetNullaryOption(arg, "--client_debug")) {
    client_debug = true;
    option_sources["client_debug"] = rcfile;
  } else if (GetNullaryOption(arg, "--noclient_debug")) {
    client_debug = false;
    option_sources["client_debug"] = rcfile;
  } else if (GetNullaryOption(arg, "--expand_configs_in_place")) {
    expand_configs_in_place = true;
    option_sources["expand_configs_in_place"] = rcfile;
  } else if (GetNullaryOption(arg, "--noexpand_configs_in_place")) {
    expand_configs_in_place = false;
    option_sources["expand_configs_in_place"] = rcfile;
  } else if (GetNullaryOption(arg, "--idle_server_tasks")) {
    idle_server_tasks = true;
    option_sources["idle_server_tasks"] = rcfile;
  } else if (GetNullaryOption(arg, "--noidle_server_tasks")) {
    idle_server_tasks = false;
    option_sources["idle_server_tasks"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--connect_timeout_secs")) != NULL) {
    if (!blaze_util::safe_strto32(value, &connect_timeout_secs) ||
        connect_timeout_secs < 1 || connect_timeout_secs > 120) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --connect_timeout_secs: '%s'.\n"
          "Must be an integer between 1 and 120.\n",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["connect_timeout_secs"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--digest_function")) !=
             NULL) {
    digest_function = value;
    option_sources["digest_function"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--command_port")) !=
             NULL) {
    if (!blaze_util::safe_strto32(value, &command_port) ||
        command_port < 0 || command_port > 65535) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --command_port: '%s'.\n"
          "Must be a valid port number or 0.\n",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["command_port"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--invocation_policy")) !=
             NULL) {
    if (invocation_policy == NULL) {
      invocation_policy = value;
      option_sources["invocation_policy"] = rcfile;
    } else {
      *error = "The startup flag --invocation_policy cannot be specified "
          "multiple times.";
      return blaze_exit_code::BAD_ARGV;
    }
  } else if (GetNullaryOption(arg, "--unlimit_coredumps")) {
    unlimit_coredumps = true;
    option_sources["unlimit_coredumps"] = rcfile;
  } else if (GetNullaryOption(arg, "--nounlimit_coredumps")) {
    unlimit_coredumps = false;
    option_sources["unlimit_coredumps"] = rcfile;
  } else {
    bool extra_argument_processed;
    blaze_exit_code::ExitCode process_extra_arg_exit_code = ProcessArgExtra(
        arg, next_arg, rcfile, &value, &extra_argument_processed, error);
    if (process_extra_arg_exit_code != blaze_exit_code::SUCCESS) {
      return process_extra_arg_exit_code;
    }
    if (!extra_argument_processed) {
      blaze_util::StringPrintf(
          error,
          "Unknown startup option: '%s'.\n"
          "  For more info, run '%s help startup_options'.",
          arg, GetLowercaseProductName().c_str());
      return blaze_exit_code::BAD_ARGV;
    }
  }

  *is_space_separated = ((value == next_arg) && (value != NULL));
  return blaze_exit_code::SUCCESS;
}

blaze_exit_code::ExitCode StartupOptions::ProcessArgs(
    const std::vector<RcStartupFlag>& rcstartup_flags,
    std::string *error) {
  std::vector<RcStartupFlag>::size_type i = 0;
  while (i < rcstartup_flags.size()) {
    bool is_space_separated = false;
    const std::string next_value =
        (i == rcstartup_flags.size() - 1) ? "" : rcstartup_flags[i + 1].value;
    const blaze_exit_code::ExitCode process_arg_exit_code =
        ProcessArg(rcstartup_flags[i].value, next_value,
                   rcstartup_flags[i].source, &is_space_separated, error);
    // Store the provided option in --flag(=value)? form. Store these before
    // propagating any error code, since we want to have the correct
    // information for the output. The fact that the options aren't parseable
    // doesn't matter for this step.
    if (is_space_separated) {
      const std::string combined_value =
          rcstartup_flags[i].value + "=" + next_value;
      original_startup_options_.push_back(
          RcStartupFlag(rcstartup_flags[i].source, combined_value));
      i += 2;
    } else {
      original_startup_options_.push_back(
          RcStartupFlag(rcstartup_flags[i].source, rcstartup_flags[i].value));
      i++;
    }

    if (process_arg_exit_code != blaze_exit_code::SUCCESS) {
      return process_arg_exit_code;
    }
  }
  return blaze_exit_code::SUCCESS;
}

string StartupOptions::GetSystemJavabase() const {
  return blaze::GetSystemJavabase();
}

string StartupOptions::GetEmbeddedJavabase() {
  string bundled_jre_path = blaze_util::JoinPath(
      install_base, "_embedded_binaries/embedded_tools/jdk");
  if (blaze_util::CanExecuteFile(blaze_util::JoinPath(
          bundled_jre_path, GetJavaBinaryUnderJavabase()))) {
    return bundled_jre_path;
  }
  return "";
}

string StartupOptions::GetServerJavabase() {
  // 1) Allow overriding the server_javabase via --server_javabase.
  if (!server_javabase_.empty()) {
    return server_javabase_;
  }
  if (default_server_javabase_.empty()) {
    string bundled_jre_path = GetEmbeddedJavabase();
    if (!bundled_jre_path.empty()) {
      // 2) Use a bundled JVM if we have one.
      default_server_javabase_ = bundled_jre_path;
    } else {
      // 3) Otherwise fall back to using the default system JVM.
      string system_javabase = GetSystemJavabase();
      if (system_javabase.empty()) {
        BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
            << "Could not find system javabase. Ensure JAVA_HOME is set, or "
               "javac is on your PATH.";
      }
      default_server_javabase_ = system_javabase;
    }
  }
  return default_server_javabase_;
}

string StartupOptions::GetExplicitServerJavabase() const {
  return server_javabase_;
}

string StartupOptions::GetJvm() {
  string java_program =
      blaze_util::JoinPath(GetServerJavabase(), GetJavaBinaryUnderJavabase());
  if (!blaze_util::CanExecuteFile(java_program)) {
    if (!blaze_util::PathExists(java_program)) {
      BAZEL_LOG(ERROR) << "Couldn't find java at '" << java_program << "'.";
    } else {
      BAZEL_LOG(ERROR) << "Java at '" << java_program
                       << "' exists but is not executable: "
                       << blaze_util::GetLastErrorString();
    }
    exit(1);
  }
  // If the full JDK is installed
  string jdk_rt_jar =
      blaze_util::JoinPath(GetServerJavabase(), "jre/lib/rt.jar");
  // If just the JRE is installed
  string jre_rt_jar = blaze_util::JoinPath(GetServerJavabase(), "lib/rt.jar");
  // rt.jar does not exist in java 9+ so check for java instead
  string jre_java = blaze_util::JoinPath(GetServerJavabase(), "bin/java");
  string jre_java_exe =
      blaze_util::JoinPath(GetServerJavabase(), "bin/java.exe");
  if (blaze_util::CanReadFile(jdk_rt_jar) ||
      blaze_util::CanReadFile(jre_rt_jar) ||
      blaze_util::CanReadFile(jre_java) ||
      blaze_util::CanReadFile(jre_java_exe)) {
    return java_program;
  }
  BAZEL_LOG(ERROR) << "Problem with java installation: couldn't find/access "
                      "rt.jar or java in "
                   << GetServerJavabase();
  exit(1);
}

string StartupOptions::GetExe(const string &jvm, const string &jar_path) {
  return jvm;
}

void StartupOptions::AddJVMArgumentPrefix(const string &javabase,
    std::vector<string> *result) const {
}

void StartupOptions::AddJVMArgumentSuffix(const string &real_install_dir,
                                          const string &jar_path,
    std::vector<string> *result) const {
  result->push_back("-jar");
  result->push_back(blaze_util::PathAsJvmFlag(
      blaze_util::JoinPath(real_install_dir, jar_path)));
}

blaze_exit_code::ExitCode StartupOptions::AddJVMArguments(
    const string &server_javabase, std::vector<string> *result,
    const vector<string> &user_options, string *error) const {
  AddJVMLoggingArguments(result);
  return AddJVMMemoryArguments(server_javabase, result, user_options, error);
}

static std::string GetJavaUtilLoggingFileHandlerProps(
    const std::string &java_log, const std::string &java_logging_formatter) {
  return "handlers=java.util.logging.FileHandler\n"
         ".level=INFO\n"
         "java.util.logging.FileHandler.level=INFO\n"
         "java.util.logging.FileHandler.pattern=" +
         java_log +
         "\n"
         "java.util.logging.FileHandler.limit=1024000\n"
         "java.util.logging.FileHandler.count=1\n"
         "java.util.logging.FileHandler.formatter=" +
         java_logging_formatter + "\n";
}

static std::string GetSimpleLogHandlerProps(
    const std::string &java_log, const std::string &java_logging_formatter) {
  return "handlers=com.google.devtools.build.lib.util.SimpleLogHandler\n"
         ".level=INFO\n"
         "com.google.devtools.build.lib.util.SimpleLogHandler.level=INFO\n"
         "com.google.devtools.build.lib.util.SimpleLogHandler.prefix=" +
         java_log +
         "\n"
         "com.google.devtools.build.lib.util.SimpleLogHandler.limit=1024000\n"
         "com.google.devtools.build.lib.util.SimpleLogHandler.total_limit="
         "20971520\n"  // 20 MB.
         "com.google.devtools.build.lib.util.SimpleLogHandler.formatter=" +
         java_logging_formatter + "\n";
}

void StartupOptions::AddJVMLoggingArguments(std::vector<string> *result) const {
  // Configure logging
  const string propFile = blaze_util::PathAsJvmFlag(
      blaze_util::JoinPath(output_base, "javalog.properties"));
  const string java_log(
      blaze_util::PathAsJvmFlag(blaze_util::JoinPath(output_base, "java.log")));
  const std::string loggingProps =
      rotating_server_log
          ? GetSimpleLogHandlerProps(java_log, java_logging_formatter)
          : GetJavaUtilLoggingFileHandlerProps(java_log,
                                               java_logging_formatter);
  const std::string loggingQuerierClass =
      rotating_server_log
          ? "com.google.devtools.build.lib.util.SimpleLogHandler$HandlerQuerier"
          : "com.google.devtools.build.lib.util.FileHandlerQuerier";

  if (!blaze_util::WriteFile(loggingProps, propFile)) {
    perror(("Couldn't write logging file " + propFile).c_str());
  } else {
    result->push_back("-Djava.util.logging.config.file=" + propFile);
    result->push_back(
        "-Dcom.google.devtools.build.lib.util.LogHandlerQuerier.class=" +
        loggingQuerierClass);
  }
}

blaze_exit_code::ExitCode StartupOptions::AddJVMMemoryArguments(
    const string &, std::vector<string> *, const vector<string> &,
    string *) const {
  return blaze_exit_code::SUCCESS;
}

#if defined(_WIN32) || defined(__CYGWIN__)
// Extract the Windows path of "/" from $BAZEL_SH.
// $BAZEL_SH usually has the form `<prefix>/usr/bin/bash.exe` or
// `<prefix>/bin/bash.exe`, and this method returns that `<prefix>` part.
// If $BAZEL_SH doesn't end with "usr/bin/bash.exe" or "bin/bash.exe" then this
// method returns an empty string.
string StartupOptions::WindowsUnixRoot(const string &bazel_sh) {
  if (bazel_sh.empty()) {
    return string();
  }
  std::pair<string, string> split = blaze_util::SplitPath(bazel_sh);
  if (blaze_util::AsLower(split.second) != "bash.exe") {
    return string();
  }
  split = blaze_util::SplitPath(split.first);
  if (blaze_util::AsLower(split.second) != "bin") {
    return string();
  }

  std::pair<string, string> split2 = blaze_util::SplitPath(split.first);
  if (blaze_util::AsLower(split2.second) == "usr") {
    return split2.first;
  } else {
    return split.first;
  }
}
#endif  // defined(_WIN32) || defined(__CYGWIN__)

}  // namespace blaze
