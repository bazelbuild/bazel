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
#include "src/main/cpp/util/numbers.h"
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

StartupOptions::StartupOptions(const WorkspaceLayout* workspace_layout)
    : StartupOptions("Bazel", workspace_layout) {}

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
      deep_execroot(true),
      block_for_lock(true),
      host_jvm_debug(false),
      batch(false),
      batch_cpu_scheduling(false),
      io_nice_level(-1),
      oom_more_eagerly(false),
      oom_more_eagerly_threshold(100),
      write_command_log(true),
      watchfs(false),
      allow_configurable_attributes(false),
      fatal_event_bus_exceptions(false),
      command_port(0),
      connect_timeout_secs(10),
      invocation_policy(NULL),
      client_debug(false),
      java_logging_formatter(
          "com.google.devtools.build.lib.util.SingleLineFormatter") {
  bool testing = !blaze::GetEnv("TEST_TMPDIR").empty();
  if (testing) {
    output_root = MakeAbsolute(blaze::GetEnv("TEST_TMPDIR"));
  } else {
    output_root = workspace_layout->GetOutputRoot();
  }

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
  string windows_unix_root = WindowsUnixRoot(blaze::GetEnv("BAZEL_SH"));
  if (!windows_unix_root.empty()) {
    host_jvm_args.push_back(string("-Dbazel.windows_unix_root=") +
                            windows_unix_root);
  }
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

  const string product_name_lower = GetLowercaseProductName();
  output_user_root = blaze_util::JoinPath(
      output_root, "_" + product_name_lower + "_" + GetUserName());
  // 3 hours (but only 15 seconds if used within a test)
  max_idle_secs = testing ? 15 : (3 * 3600);

  // IMPORTANT: Before modifying the statements below please contact a Bazel
  // core team member that knows the internal procedure for adding/deprecating
  // startup flags.
  RegisterNullaryStartupFlag("allow_configurable_attributes");
  RegisterNullaryStartupFlag("batch");
  RegisterNullaryStartupFlag("batch_cpu_scheduling");
  RegisterNullaryStartupFlag("block_for_lock");
  RegisterNullaryStartupFlag("client_debug");
  RegisterNullaryStartupFlag("deep_execroot");
  RegisterNullaryStartupFlag("experimental_oom_more_eagerly");
  RegisterNullaryStartupFlag("fatal_event_bus_exceptions");
  RegisterNullaryStartupFlag("host_jvm_debug");
  RegisterNullaryStartupFlag("master_bazelrc");
  RegisterNullaryStartupFlag("master_blazerc");
  RegisterNullaryStartupFlag("watchfs");
  RegisterNullaryStartupFlag("write_command_log");
  RegisterUnaryStartupFlag("bazelrc");
  RegisterUnaryStartupFlag("blazerc");
  RegisterUnaryStartupFlag("command_port");
  RegisterUnaryStartupFlag("connect_timeout_secs");
  RegisterUnaryStartupFlag("experimental_oom_more_eagerly_threshold");
  RegisterUnaryStartupFlag("host_javabase");
  RegisterUnaryStartupFlag("host_jvm_args");
  RegisterUnaryStartupFlag("host_jvm_profile");
  RegisterUnaryStartupFlag("invocation_policy");
  RegisterUnaryStartupFlag("io_nice_level");
  RegisterUnaryStartupFlag("install_base");
  RegisterUnaryStartupFlag("max_idle_secs");
  RegisterUnaryStartupFlag("output_base");
  RegisterUnaryStartupFlag("output_user_root");
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
    output_base = MakeAbsolute(value);
    option_sources["output_base"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--install_base")) != NULL) {
    install_base = MakeAbsolute(value);
    option_sources["install_base"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--output_user_root")) != NULL) {
    output_user_root = MakeAbsolute(value);
    option_sources["output_user_root"] = rcfile;
  } else if (GetNullaryOption(arg, "--deep_execroot")) {
    deep_execroot = true;
    option_sources["deep_execroot"] = rcfile;
  } else if (GetNullaryOption(arg, "--nodeep_execroot")) {
    deep_execroot = false;
    option_sources["deep_execroot"] = rcfile;
  } else if (GetNullaryOption(arg, "--noblock_for_lock")) {
    block_for_lock = false;
    option_sources["block_for_lock"] = rcfile;
  } else if (GetNullaryOption(arg, "--host_jvm_debug")) {
    host_jvm_debug = true;
    option_sources["host_jvm_debug"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--host_jvm_profile")) != NULL) {
    host_jvm_profile = value;
    option_sources["host_jvm_profile"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--host_javabase")) != NULL) {
    // TODO(bazel-team): Consider examining the javabase and re-execing in case
    // of architecture mismatch.
    host_javabase = MakeAbsolute(value);
    option_sources["host_javabase"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--host_jvm_args")) !=
             NULL) {
    host_jvm_args.push_back(value);
    option_sources["host_jvm_args"] = rcfile;  // NB: This is incorrect
  } else if ((value = GetUnaryOption(arg, next_arg, "--bazelrc")) != NULL) {
    if (!rcfile.empty()) {
      *error = "Can't specify --bazelrc in the .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
  } else if ((value = GetUnaryOption(arg, next_arg, "--blazerc")) != NULL) {
    if (!rcfile.empty()) {
      *error = "Can't specify --blazerc in the .blazerc file.";
      return blaze_exit_code::BAD_ARGV;
    }
  } else if (GetNullaryOption(arg, "--nomaster_blazerc") ||
             GetNullaryOption(arg, "--master_blazerc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --[no]master_blazerc in .blazerc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["blazerc"] = rcfile;
  } else if (GetNullaryOption(arg, "--nomaster_bazelrc") ||
             GetNullaryOption(arg, "--master_bazelrc")) {
    if (!rcfile.empty()) {
      *error = "Can't specify --[no]master_bazelrc in .bazelrc file.";
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["blazerc"] = rcfile;
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
  } else if (GetNullaryOption(arg, "--allow_configurable_attributes")) {
    allow_configurable_attributes = true;
    option_sources["allow_configurable_attributes"] = rcfile;
  } else if (GetNullaryOption(arg, "--noallow_configurable_attributes")) {
    allow_configurable_attributes = false;
    option_sources["allow_configurable_attributes"] = rcfile;
  } else if (GetNullaryOption(arg, "--fatal_event_bus_exceptions")) {
    fatal_event_bus_exceptions = true;
    option_sources["fatal_event_bus_exceptions"] = rcfile;
  } else if (GetNullaryOption(arg, "--nofatal_event_bus_exceptions")) {
    fatal_event_bus_exceptions = false;
    option_sources["fatal_event_bus_exceptions"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--io_nice_level")) != NULL) {
    if (!blaze_util::safe_strto32(value, &io_nice_level) ||
        io_nice_level > 7) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --io_nice_level: '%s'. Must not exceed 7.",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["io_nice_level"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg,
                                     "--max_idle_secs")) != NULL) {
    if (!blaze_util::safe_strto32(value, &max_idle_secs) ||
        max_idle_secs < 0) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --max_idle_secs: '%s'.", value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["max_idle_secs"] = rcfile;
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
  } else if ((value = GetUnaryOption(
      arg, next_arg, "--connect_timeout_secs")) != NULL) {
    if (!blaze_util::safe_strto32(value, &connect_timeout_secs) ||
        connect_timeout_secs < 1 || connect_timeout_secs > 120) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --connect_timeout_secs: '%s'.\n"
          "Must be an integer between 1 and 120.\n",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["connect_timeout_secs"] = rcfile;
  } else if ((value = GetUnaryOption(
      arg, next_arg, "--command_port")) != NULL) {
    if (!blaze_util::safe_strto32(value, &command_port) ||
        command_port < 0 || command_port > 65535) {
      blaze_util::StringPrintf(error,
          "Invalid argument to --command_port: '%s'.\n"
          "Must be a valid port number or 0.\n",
          value);
      return blaze_exit_code::BAD_ARGV;
    }
    option_sources["command_port"] = rcfile;
  } else if ((value = GetUnaryOption(arg, next_arg, "--invocation_policy"))
              != NULL) {
    if (invocation_policy == NULL) {
      invocation_policy = value;
      option_sources["invocation_policy"] = rcfile;
    } else {
      *error = "The startup flag --invocation_policy cannot be specified "
          "multiple times.";
      return blaze_exit_code::BAD_ARGV;
    }
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

blaze_exit_code::ExitCode StartupOptions::ProcessArgExtra(
    const char *arg, const char *next_arg, const string &rcfile,
    const char **value, bool *is_processed, string *error) {
  *is_processed = false;
  return blaze_exit_code::SUCCESS;
}

string StartupOptions::GetDefaultHostJavabase() const {
  return blaze::GetDefaultHostJavabase();
}

string StartupOptions::GetHostJavabase() {
  // 1) Allow overriding the host_javabase via --host_javabase.
  if (host_javabase.empty()) {
    if (default_host_javabase.empty()) {
      string bundled_jre_path = blaze_util::JoinPath(
          install_base, "_embedded_binaries/embedded_tools/jdk");
      if (blaze_util::CanExecuteFile(blaze_util::JoinPath(
              bundled_jre_path, GetJavaBinaryUnderJavabase()))) {
        // 2) Use a bundled JVM if we have one.
        default_host_javabase = bundled_jre_path;
      } else {
        // 3) Otherwise fall back to using the default system JVM.
        default_host_javabase = GetDefaultHostJavabase();
      }
    }

    return default_host_javabase;
  } else {
    return host_javabase;
  }
}

string StartupOptions::GetExplicitHostJavabase() const {
  return host_javabase;
}

string StartupOptions::GetJvm() {
  string java_program =
      blaze_util::JoinPath(GetHostJavabase(), GetJavaBinaryUnderJavabase());
  if (!blaze_util::CanExecuteFile(java_program)) {
    if (!blaze_util::PathExists(java_program)) {
      fprintf(stderr, "Couldn't find java at '%s'.\n", java_program.c_str());
    } else {
      fprintf(stderr, "Java at '%s' exists but is not executable: %s\n",
              java_program.c_str(), blaze_util::GetLastErrorString().c_str());
    }
    exit(1);
  }
  // If the full JDK is installed
  string jdk_rt_jar = blaze_util::JoinPath(GetHostJavabase(), "jre/lib/rt.jar");
  // If just the JRE is installed
  string jre_rt_jar = blaze_util::JoinPath(GetHostJavabase(), "lib/rt.jar");
  if (blaze_util::CanReadFile(jdk_rt_jar) ||
      blaze_util::CanReadFile(jre_rt_jar)) {
    return java_program;
  }
  fprintf(stderr, "Problem with java installation: "
      "couldn't find/access rt.jar in %s\n", GetHostJavabase().c_str());
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
  result->push_back(
      blaze::PathAsJvmFlag(blaze_util::JoinPath(real_install_dir, jar_path)));
}

blaze_exit_code::ExitCode StartupOptions::AddJVMArguments(
    const string &host_javabase, vector<string> *result,
    const vector<string> &user_options, string *error) const {
  // Configure logging
  const string propFile =
      blaze_util::JoinPath(output_base, "javalog.properties");
  string java_log(
      blaze::PathAsJvmFlag(blaze_util::JoinPath(output_base, "java.log")));
  if (!blaze_util::WriteFile("handlers=java.util.logging.FileHandler\n"
                             ".level=INFO\n"
                             "java.util.logging.FileHandler.level=INFO\n"
                             "java.util.logging.FileHandler.pattern=" +
                                 java_log +
                                 "\n"
                                 "java.util.logging.FileHandler.limit=1024000\n"
                                 "java.util.logging.FileHandler.count=1\n"
                                 "java.util.logging.FileHandler.formatter=" +
                                 java_logging_formatter + "\n",
                             propFile)) {
    perror(("Couldn't write logging file " + propFile).c_str());
  } else {
    result->push_back("-Djava.util.logging.config.file=" + propFile);
  }
  return blaze_exit_code::SUCCESS;
}

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
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
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

}  // namespace blaze
