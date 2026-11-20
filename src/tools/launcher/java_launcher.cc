// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include "src/tools/launcher/java_launcher.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/launcher/util/launcher_util.h"

namespace bazel {
namespace launcher {

using std::getline;
using std::ofstream;
using std::string;
using std::vector;
using std::wostringstream;
using std::wstring;
using std::wstringstream;

// The runfile path of java binary, eg. local_jdk/bin/java.exe
static constexpr const char* JAVA_BIN_PATH = "java_bin_path";
static constexpr const char* CLASSPATH = "classpath";
static constexpr const char* JAVA_START_CLASS = "java_start_class";
static constexpr const char* JVM_FLAGS = "jvm_flags";
static constexpr const char* JACOCO_MAIN_CLASS = "jacoco_main_class";

// Check if a string start with a certain prefix.
// If it's true, store the substring without the prefix in value.
// If value is quoted, then remove the quotes.
static bool GetFlagValue(const wstring& str, const wstring& prefix,
                         wstring* value_ptr) {
  if (str.compare(0, prefix.length(), prefix)) {
    return false;
  }
  wstring& value = *value_ptr;
  value = str.substr(prefix.length());
  int len = value.length();
  if (len >= 2 && value[0] == L'"' && value[len - 1] == L'"') {
    value = value.substr(1, len - 2);
  }
  return true;
}

// Parses one launcher flag and updates this object's state accordingly.
//
// Returns true if the flag is a valid launcher flag; false otherwise.
bool JavaBinaryLauncher::ProcessWrapperArgument(const wstring& argument) {
  wstring flag_value;
  if (argument.compare(L"--debug") == 0) {
    wstring default_jvm_debug_port;
    if (GetEnv(L"DEFAULT_JVM_DEBUG_PORT", &default_jvm_debug_port)) {
      this->jvm_debug_port = default_jvm_debug_port;
    } else {
      this->jvm_debug_port = L"5005";
    }
  } else if (GetFlagValue(argument, L"--debug=", &flag_value)) {
    this->jvm_debug_port = flag_value;
  } else if (GetFlagValue(argument, L"--main_advice=", &flag_value)) {
    this->main_advice = flag_value;
  } else if (GetFlagValue(argument, L"--main_advice_classpath=", &flag_value)) {
    this->main_advice_classpath = flag_value;
  } else if (GetFlagValue(argument, L"--jvm_flag=", &flag_value)) {
    this->jvm_flags_cmdline.push_back(flag_value);
  } else if (GetFlagValue(argument, L"--jvm_flags=", &flag_value)) {
    wstringstream flag_value_ss(flag_value);
    wstring item;
    while (getline(flag_value_ss, item, L' ')) {
      this->jvm_flags_cmdline.push_back(item);
    }
  } else if (argument.compare(L"--singlejar") == 0) {
    this->singlejar = true;
  } else if (argument.compare(L"--print_javabin") == 0) {
    this->print_javabin = true;
  } else if (GetFlagValue(argument, L"--classpath_limit=", &flag_value)) {
    this->classpath_limit = std::stoi(flag_value);
  } else {
    return false;
  }
  return true;
}

vector<wstring> JavaBinaryLauncher::ProcessesCommandLine() {
  vector<wstring> args;
  bool first = 1;
  for (const auto& arg : this->GetCommandlineArguments()) {
    // Skip the first argument.
    if (first) {
      first = 0;
      continue;
    }
    wstring flag_value;
    // TODO(pcloudy): Should rename this flag to --native_launcher_flag.
    // But keep it as it is for now to be consistent with the shell script
    // launcher.
    if (GetFlagValue(arg, L"--wrapper_script_flag=", &flag_value)) {
      if (!ProcessWrapperArgument(flag_value)) {
        die(L"invalid wrapper argument '%ls'", arg.c_str());
      }
    } else if (!args.empty() || !ProcessWrapperArgument(arg)) {
      args.push_back(arg);
    }
  }
  return args;
}

wstring JavaBinaryLauncher::CreateClasspathFlagfile(const wstring& classpath) {
  wstring binary_base_path = GetBinaryPathWithoutExtension(GetLauncherPath());

  // Wrap in quotes so that space isn't interpreted as an argument delimiter.
  // Replace backslashes with forward slashes to avoid accidental interpretation
  // as escape sequences.
  wstring contents;
  contents.reserve(classpath.length() + 3);
  contents += L"\"";
  for (const wchar_t& c : classpath) {
    contents += (c == L'\\' ? L'/' : c);
  }
  contents += L"\"\n";

  // Flagfiles must be UTF-8.
  std::string contents_utf8;
  if (!blaze_util::WcsToUtf8(contents, &contents_utf8)) {
    die(L"Couldn't convert classpath to UTF-8: %ls", contents.c_str());
  }

  // Enable long path support for flagfile_path.
  wstring flagfile_path =
      binary_base_path + L"-" + rand_id_ + L"-classpath.txt";
  blaze_util::AddUncPrefixMaybe(&flagfile_path);
  ofstream flagfile{flagfile_path.c_str()};
  flagfile << contents_utf8;
  flagfile.close();
  if (flagfile.fail()) {
    die(L"Couldn't write classpath flagfile: %ls", flagfile_path.c_str());
  }

  return flagfile_path;
}

ExitCode JavaBinaryLauncher::Launch() {
  // Parse the original command line.
  vector<wstring> remaining_args = this->ProcessesCommandLine();

  // Set JAVA_RUNFILES
  wstring java_runfiles;
  if (!GetEnv(L"JAVA_RUNFILES", &java_runfiles)) {
    java_runfiles = this->GetRunfilesPath();
  }
  SetEnv(L"JAVA_RUNFILES", java_runfiles);

  // Set JACOCO_MAIN_CLASS for JaCoCo coverage support.
  // rules_java writes jacoco_main_class into the launch info when coverage is
  // enabled; the Linux shell stub exports it as JACOCO_MAIN_CLASS but the
  // Windows launcher was missing this step.
  wstring jacoco_main_class =
      this->GetLaunchInfoByKeyOrEmpty(JACOCO_MAIN_CLASS);
  if (!jacoco_main_class.empty()) {
    SetEnv(L"JACOCO_MAIN_CLASS", jacoco_main_class);
  }

  // Print Java binary path if needed
  wstring java_bin = this->Rlocation(this->GetLaunchInfoByKey(JAVA_BIN_PATH),
                                     /*has_workspace_name =*/true);
  if (this->print_javabin ||
      this->GetLaunchInfoByKey(JAVA_START_CLASS) == L"--print_javabin") {
    wprintf(L"%ls\n", java_bin.c_str());
    return 0;
  }

  wostringstream classpath;

  // Run deploy jar if needed, otherwise generate the CLASSPATH by rlocation.
  if (this->singlejar) {
    wstring deploy_jar =
        GetBinaryPathWithoutExtension(GetLauncherPath()) + L"_deploy.jar";
    if (!DoesFilePathExist(deploy_jar.c_str())) {
      die(L"Option --singlejar was passed, but %ls does not exist.\n  (You may "
          "need to build it explicitly.)",
          deploy_jar.c_str());
    }
    classpath << deploy_jar << L';';
  } else {
    wstring path;
    // Add main advice classpath if exists
    if (!this->main_advice_classpath.empty()) {
      wstringstream main_advice_classpath_ss(this->main_advice_classpath);
      while (getline(main_advice_classpath_ss, path, L';')) {
        classpath << this->Rlocation(path) << L';';
      }
    }
    wstringstream classpath_ss(this->GetLaunchInfoByKey(CLASSPATH));
    while (getline(classpath_ss, path, L';')) {
      classpath << this->Rlocation(path) << L';';
    }
  }

  // Set jvm debug options
  wostringstream jvm_debug_flags;
  if (!this->jvm_debug_port.empty()) {
    wstring jvm_debug_suspend;
    if (!GetEnv(L"DEFAULT_JVM_DEBUG_SUSPEND", &jvm_debug_suspend)) {
      jvm_debug_suspend = L"y";
    }
    jvm_debug_flags << L"-agentlib:jdwp=transport=dt_socket,server=y";
    jvm_debug_flags << L",suspend=" << jvm_debug_suspend;
    jvm_debug_flags << L",address=" << jvm_debug_port;

    wstring value;
    if (GetEnv(L"PERSISTENT_TEST_RUNNER", &value) && value == L"true") {
      jvm_debug_flags << L",quiet=y";
    }
  }

  // Get jvm flags from JVM_FLAGS environment variable and JVM_FLAGS launch info
  vector<wstring> jvm_flags;
  wstring jvm_flags_env;
  GetEnv(L"JVM_FLAGS", &jvm_flags_env);
  wstring flag;
  wstringstream jvm_flags_env_ss(jvm_flags_env);
  while (getline(jvm_flags_env_ss, flag, L' ')) {
    jvm_flags.push_back(flag);
  }
  wstringstream jvm_flags_launch_info_ss(this->GetLaunchInfoByKey(JVM_FLAGS));
  while (getline(jvm_flags_launch_info_ss, flag, L'\t')) {
    jvm_flags.push_back(flag);
  }

  // Check if TEST_TMPDIR is available to use for scratch.
  wstring test_tmpdir;
  if (GetEnv(L"TEST_TMPDIR", &test_tmpdir) &&
      DoesDirectoryPathExist(test_tmpdir.c_str())) {
    jvm_flags.push_back(L"-Djava.io.tmpdir=" + test_tmpdir);
  }

  // Construct the final command line arguments
  vector<wstring> arguments;
  // Add classpath flags
  arguments.push_back(L"-classpath");
  // Check whether the classpath exceeds the length limit.
  // If it does, spill it to a flagfile.
  wstring classpath_str = classpath.str();
  wstring classpath_file;
  if (classpath_str.length() > classpath_limit) {
    classpath_file = CreateClasspathFlagfile(classpath_str);
    arguments.push_back(L"@" + classpath_file);
  } else {
    arguments.push_back(classpath_str);
  }
  // Add JVM debug flags
  wstring jvm_debug_flags_str = jvm_debug_flags.str();
  if (!jvm_debug_flags_str.empty()) {
    arguments.push_back(jvm_debug_flags_str);
  }
  // Add JVM flags parsed from env and launch info.
  for (const auto& arg : jvm_flags) {
    arguments.push_back(arg);
  }
  // Add JVM flags parsed from command line.
  for (const auto& arg : this->jvm_flags_cmdline) {
    arguments.push_back(arg);
  }
  // Add main advice class
  if (!this->main_advice.empty()) {
    arguments.push_back(this->main_advice);
  }
  // Add java start class
  arguments.push_back(this->GetLaunchInfoByKey(JAVA_START_CLASS));
  // Add the remaininng arguments, they will be passed to the program.
  for (const auto& arg : remaining_args) {
    arguments.push_back(arg);
  }

  ExitCode exit_code = this->LaunchProcess(java_bin, arguments);

  // Delete classpath file after execution.
  if (!classpath_file.empty()) {
    DeleteFileByPath(classpath_file.c_str());
  }

  return exit_code;
}

}  // namespace launcher
}  // namespace bazel
