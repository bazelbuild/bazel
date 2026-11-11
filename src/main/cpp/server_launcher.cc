// Copyright 2026 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/server_launcher.h"

#include <climits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/option_processor.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/bazel_log_handler.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"
#include "absl/time/time.h"

using blaze_util::GetLastErrorString;

namespace blaze {

using std::map;
using std::set;
using std::string;
using std::vector;

// Determines the maximum import depth for rc files based on the environment.
static int SelectMaxImportDepth() {
  return ExistsEnv("BAZEL_UNLIMITED_IMPORT_DEPTH")
             ? INT_MAX
             : OptionProcessor::MaxImportDepth;
}

std::pair<ClientLocks, DurationMillis> AcquireClientLocks(
    const StartupOptions& startup_options) {
  ClientLocks locks;
  DurationMillis wait_time;

  if (startup_options.lock_install_base) {
    // Take a shared lock on the install base, because two simultaneous commands
    // may run against the same install base. The lock file is a sibling of the
    // install base directory to make it possible to synchronize concurrent
    // creation through an atomic rename. An exclusive lock is only required to
    // run automatic garbage collection of stale install bases.
    // Always block waiting for the lock: in the event that the install base is
    // currently being garbage collected, we want to recreate it.
    blaze_util::Path install_base_parent =
        startup_options.install_base.GetParent();
    blaze_util::MakeDirectories(install_base_parent, 0777);
    auto install_base_result = blaze::AcquireLock(
        "install base",
        install_base_parent.GetRelative(
            startup_options.install_base.GetBaseName() + ".lock"),
        LockMode::kShared, startup_options.batch,
        /* timeout= */ absl::InfiniteDuration());
    locks.install_base_lock = install_base_result.first;
    wait_time += install_base_result.second;
  }

  // Take an exclusive lock on the output base, because two simultaneous
  // commands may not run against the same output base.
  auto output_base_result = blaze::AcquireLock(
      "output base", startup_options.output_base.GetRelative("lock"),
      LockMode::kExclusive, startup_options.batch,
      startup_options.block_for_lock_timeout);
  locks.output_base_lock = output_base_result.first;
  wait_time += output_base_result.second;

  return std::make_pair(std::move(locks), wait_time);
}

void ReleaseClientLocks(bool lock_install_base, ClientLocks* locks) {
  if (!locks->output_base_lock.has_value()) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "ReleaseLocks() called but the output base lock has already been "
           "released.";
  }
  blaze::ReleaseLock(*locks->output_base_lock);
  locks->output_base_lock = std::nullopt;

  if (lock_install_base) {
    if (!locks->install_base_lock.has_value()) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "ReleaseLocks() called but the install base lock has already been "
             "released.";
    }
    blaze::ReleaseLock(*locks->install_base_lock);
    locks->install_base_lock = std::nullopt;
  }
}

// Escapes colons by replacing them with '_C' and underscores by replacing
// them with '_U'. E.g. "name:foo_bar" becomes "name_Cfoo_Ubar"
static string EscapeForOptionSource(const string& input) {
  string result = input;
  blaze_util::Replace("_", "_U", &result);
  blaze_util::Replace(":", "_C", &result);
  return result;
}

// Returns the JVM command argument array.
vector<string> GetServerExeArgs(const blaze_util::Path& jvm_path,
                                const string& server_jar_path,
                                const string& install_md5,
                                const WorkspaceLayout& workspace_layout,
                                const string& workspace,
                                const StartupOptions& startup_options) {
  vector<string> result;

  // e.g. A Blaze server process running in ~/src/build_root (where there's a
  // ~/src/build_root/WORKSPACE file) will appear in ps(1) as "blaze(src)".
  result.push_back(startup_options.GetLowercaseProductName() + "(" +
                   workspace_layout.GetPrettyWorkspaceName(workspace) + ")");
  startup_options.AddJVMArgumentPrefix(jvm_path.GetParent().GetParent(),
                                       &result);

  // com.google.devtools.build.lib.unsafe.StringUnsafe uses reflection to
  // access private fields in java.lang.String. The Bazel server requires Java
  // 11, so this option is known to be supported.
  result.push_back("--add-opens=java.base/java.lang=ALL-UNNAMED");

  result.push_back("-XX:+UnlockDiagnosticVMOptions");
  result.push_back("-XX:-BytecodeVerificationLocal");
  result.push_back("-XX:-BytecodeVerificationRemote");

  vector<string> user_options = startup_options.host_jvm_args;

  // Add JVM arguments particular to building blaze64 and particular JVM
  // versions.
  string error;
  blaze_exit_code::ExitCode jvm_args_exit_code =
      startup_options.AddJVMArguments(startup_options.GetServerJavabase(),
                                      &result, user_options, &error);
  if (jvm_args_exit_code != blaze_exit_code::SUCCESS) {
    BAZEL_DIE(jvm_args_exit_code) << error;
  }

  // We put all directories on java.library.path that contain .so/.dll files.
  set<string> java_library_paths;
  std::stringstream java_library_path;
  java_library_path << "-Djava.library.path=";
  blaze_util::Path real_install_dir =
      blaze_util::Path(startup_options.install_base);

  vector<blaze_util::Path> extracted_files;
  blaze_util::GetAllFilesUnder(real_install_dir, &extracted_files);
  for (const auto& it : extracted_files) {
    if (IsSharedLibrary(it.GetBaseName())) {
      java_library_paths.insert(it.GetParent().AsJvmArgument());
    }
  }
  bool first_libpath = true;
  for (const auto& libpath : java_library_paths) {
    if (!first_libpath) {
      java_library_path << kListSeparator;
    }
    first_libpath = false;
    java_library_path << libpath;
  }
  result.push_back(java_library_path.str());

  // TODO: Investigate whether this still has any effect. File name encoding
  // is governed by sun.jnu.encoding in JDKs with JEP 400, which can't be
  // influenced by setting a property.
  result.push_back("-Dfile.encoding=ISO-8859-1");
  // Force into the root locale to ensure consistent behavior of string
  // operations across machines (e.g. in the tr_TR locale, capital ASCII 'I'
  // turns into a special Unicode 'i' when converted to lower case).
  // https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Locale.html#ROOT
  result.push_back("-Duser.country=");
  result.push_back("-Duser.language=");
  result.push_back("-Duser.variant=");

  // Allow more files to be watched per directory than the default limit of 500.
  // The limit of 10,000 is arbitrary, but should be sufficient for most cases
  // and can always be increased by the user if necessary.
  // https://github.com/openjdk/jdk/blob/2faf8b8d582183275b1fdc92313a1c63c1753e80/src/java.base/share/classes/sun/nio/fs/AbstractWatchKey.java#L40
  result.push_back("-Djdk.nio.file.WatchService.maxEventsPerPoll=10000");

  // Disable warnings about unsafe memory access, which still occurs in
  // protobuf.
  // TODO: Drop this when protobuf uses VarHandle.
  result.push_back("-Dsun.misc.unsafe.memory.access=allow");

  // Let the system decide whether to prefer IPv6
  result.push_back("-Djava.net.preferIPv6Addresses=system");

#if defined(_WIN32)
  // See and use more than 64 CPUs on Windows.
  // https://bugs.openjdk.org/browse/JDK-6942632
  result.push_back("-XX:+IgnoreUnrecognizedVMOptions");
  result.push_back("-XX:+UseAllWindowsProcessorGroups");
#endif

  if (startup_options.host_jvm_debug) {
    BAZEL_LOG(USER)
        << "Running host JVM under debugger (listening on TCP port 5005).";
    // Start JVM so that it listens for a connection from a
    // JDWP-compliant debugger:
    result.push_back(
        "-agentlib:jdwp=transport=dt_socket,server=y,address=5005");
  }
  result.insert(result.end(), user_options.begin(), user_options.end());

  startup_options.AddJVMArgumentSuffix(real_install_dir, server_jar_path,
                                       &result);

  // JVM arguments are complete. Now pass in Blaze startup options.
  // Note that we always use the --flag=ARG form (instead of the --flag ARG
  // one) so that BlazeRuntime#splitStartupOptions has an easy job.

  // TODO(b/152047869): Test that whatever the list constructed after this
  // line is actually a list of parseable startup options.
  if (!startup_options.batch) {
    result.push_back("--max_idle_secs=" +
                     blaze_util::ToString(startup_options.max_idle_secs));
    if (startup_options.shutdown_on_low_sys_mem) {
      result.push_back("--shutdown_on_low_sys_mem");
    } else {
      result.push_back("--noshutdown_on_low_sys_mem");
    }
  } else {
    // --batch must come first in the arguments to Java main() because
    // the code expects it to be at args[0] if it's been set.
    result.push_back("--batch");
  }

  if (startup_options.command_port != 0) {
    result.push_back("--command_port=" +
                     blaze_util::ToString(startup_options.command_port));
  }

  result.push_back("--connect_timeout_secs=" +
                   blaze_util::ToString(startup_options.connect_timeout_secs));

  result.push_back("--output_user_root=" +
                   startup_options.output_user_root.AsCommandLineArgument());
  result.push_back("--install_base=" +
                   startup_options.install_base.AsCommandLineArgument());
  result.push_back("--install_md5=" + install_md5);
  if (startup_options.lock_install_base) {
    // This flag is not user-settable. Its sole purpose is to alter the behavior
    // for Blaze and Bazel. Do *not* explicitly set it to disabled, because at
    // Google we rely on the ability to run the client code against a server
    // built before the flag was added.
    result.push_back("--lock_install_base");
  }
  result.push_back("--output_base=" +
                   startup_options.output_base.AsCommandLineArgument());
  result.push_back("--workspace_directory=" +
                   blaze_util::ConvertPath(workspace));
  if (startup_options.autodetect_server_javabase) {
    result.push_back("--default_system_javabase=" + GetSystemJavabase());
  }

  if (!startup_options.server_jvm_out.IsEmpty()) {
    result.push_back("--server_jvm_out=" +
                     startup_options.server_jvm_out.AsCommandLineArgument());
  }

  if (!startup_options.failure_detail_out.IsEmpty()) {
    result.push_back(
        "--failure_detail_out=" +
        startup_options.failure_detail_out.AsCommandLineArgument());
  }

  if (!startup_options.digest_function.empty()) {
    // Only include this if a value is requested - we rely on the empty case
    // being "null" to set the programmatic default in the server.
    result.push_back("--digest_function=" + startup_options.digest_function);
  }
  if (!startup_options.unix_digest_hash_attribute_name.empty()) {
    result.push_back("--unix_digest_hash_attribute_name=" +
                     startup_options.unix_digest_hash_attribute_name);
  }
  if (startup_options.idle_server_tasks) {
    result.push_back("--idle_server_tasks");
  } else {
    result.push_back("--noidle_server_tasks");
  }

  if (startup_options.write_command_log) {
    result.push_back("--write_command_log");
  } else {
    result.push_back("--nowrite_command_log");
  }

  if (startup_options.fatal_event_bus_exceptions) {
    result.push_back("--fatal_event_bus_exceptions");
  } else {
    result.push_back("--nofatal_event_bus_exceptions");
  }
  if (startup_options.windows_enable_symlinks) {
    result.push_back("--windows_enable_symlinks");
  } else {
    result.push_back("--nowindows_enable_symlinks");
  }
  if (startup_options.delete_output_base_on_shutdown) {
    result.push_back("--delete_output_base_on_shutdown");
  } else {
    result.push_back("--nodelete_output_base_on_shutdown");
  }
  if (startup_options.remote_repo_contents_cache) {
    result.push_back("--experimental_remote_repo_contents_cache");
    // Don't set the flag to false if it's not set - non-OSS Blaze does not know
    // about this flag.
  }
  // We use this syntax so that the logic in AreStartupOptionsDifferent() that
  // decides whether the server needs killing is simpler. This is parsed by
  // the Java code where --noclient_debug and --client_debug=false are
  // equivalent. Note that --client_debug false (separated by space) won't
  // work either, because the logic in AreStartupOptionsDifferent() assumes
  // that every argument is in the --arg=value form.
  if (startup_options.client_debug) {
    result.push_back("--client_debug=true");
  } else {
    result.push_back("--client_debug=false");
  }

  // These flags are passed to the java process only for Blaze reporting
  // purposes; the real interpretation of the jvm flags occurs when we set up
  // the java command line.
  if (!startup_options.GetExplicitServerJavabase().IsEmpty()) {
    result.push_back(
        "--server_javabase=" +
        startup_options.GetExplicitServerJavabase().AsCommandLineArgument());
  }
  if (startup_options.host_jvm_debug) {
    result.push_back("--host_jvm_debug");
  }
  if (!startup_options.host_jvm_args.empty()) {
    for (const auto& arg : startup_options.host_jvm_args) {
      result.push_back("--host_jvm_args=" + arg);
    }
  }
  if (!startup_options.extra_classpath.empty()) {
    result.push_back("--extra_classpath=" + startup_options.extra_classpath);
  }

  // Pass in invocation policy as a startup argument for batch mode only.
  if (startup_options.batch && !startup_options.invocation_policy.empty()) {
    result.push_back("--invocation_policy=" +
                     startup_options.invocation_policy);
  }

  result.push_back("--product_name=" + startup_options.product_name);

#ifdef __linux__
  if (!startup_options.cgroup_parent.empty()) {
    result.push_back("--experimental_cgroup_parent=" +
                     startup_options.cgroup_parent);
  }

  if (startup_options.run_in_user_cgroup) {
    result.push_back("--experimental_run_in_user_cgroup");
  }
#endif

  startup_options.AddExtraOptions(&result);

  // The option sources are transmitted in the following format:
  // --option_sources=option1:source1:option2:source2:...
  string option_sources = "--option_sources=";
  bool first = true;
  for (const auto& it : startup_options.option_sources) {
    if (!first) {
      option_sources += ':';
    }

    first = false;
    option_sources += EscapeForOptionSource(it.first) + ":" +
                      EscapeForOptionSource(it.second);
  }

  result.push_back(option_sources);

#if defined(__OpenBSD__)
  // When spawning the server's JVM process, we normally set argv[0] to
  // "bazel(workspace)". On OpenBSD, doing so causes the JVM process to fail
  // during startup; ld.so fails to find a shared library that exists in
  // /usr/local/jdk-1.8.0/jre/lib/amd64. Setting LD_LIBRARY_PATH does not help,
  // but setting argv[0] to the JVM binary's path
  // (/usr/local/jdk-1.8.0/bin/java) allows the JVM process to run. The JVM
  // process apparently tries to compute a path to where the shared libraries
  // should be, via a relative path from the JVM executable's path -- but
  // OpenBSD does not provide a way for a process to determine a path to its
  // own executable, and so the JVM falls back to searching the PATH for
  // argv[0], which of course fails when argv[0] looks like "bazel(workspace)".
  //
  // TODO(aldersondrive): This hack is unnecessary on FreeBSD, but the relevant
  // OpenJDK code doesn't seem to include anything FreeBSD-specific.
  // Investigate why and possibly remove this.
  result[0] = startup_options.GetExe(jvm_path, server_jar_path).AsNativePath();
#endif

  return result;
}

// Join the elements of the specified array with NUL's (\0's), akin to the
// format of /proc/$PID/cmdline.
string GetArgumentString(const vector<string>& argument_array) {
  string result;
  blaze_util::JoinStrings(argument_array, '\0', &result);
  return result;
}

void EnsureServerDir(const blaze_util::Path& server_dir) {
  // The server dir has the connection info - don't allow access by other users.
  if (!blaze_util::MakeDirectories(server_dir, 0700)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "server directory '" << server_dir.AsPrintablePath()
        << "' could not be created: " << GetLastErrorString();
  }
}

// Do a chdir into the workspace, and die if it fails.
void GoToWorkspace(const WorkspaceLayout& workspace_layout,
                   const string& workspace) {
  if (workspace_layout.InWorkspace(workspace) &&
      !blaze_util::ChangeDirectory(workspace)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "changing directory into " << workspace
        << " failed: " << GetLastErrorString();
  }
}

bool IsServerMode(const string& command) { return "exec-server" == command; }

// Keep this in sync with built-in commands annotated with
// @Command(mustRunInWorkspace = false).
static bool CanRunOutsideWorkspace(const string& command) {
  return command == "canonicalize-flags" || command == "dump" ||
         command == "help" || command == "info" || command == "license" ||
         command == "shutdown" || command == "version";
}

// Returns the path of the file that the server JVM reads its arguments from on
// Windows.
blaze_util::Path GetServerJvmArgFilePath(const blaze_util::Path& output_base) {
  return output_base.GetRelative("server_jvm.args");
}

// Returns the server's PID from the pid file, or -1 if there was an error.
int GetServerPid(const blaze_util::Path& server_dir) {
  // Note: there is no race here on startup since the server creates
  // the pid file strictly before it binds the socket.
  blaze_util::Path pid_file = server_dir.GetRelative(kServerPidFile);
  string bufstr;
  int result;
  if (!blaze_util::ReadFile(pid_file, &bufstr, 32) ||
      !blaze_util::safe_strto32(bufstr, &result)) {
    return -1;
  }

  return result;
}

bool NeedsNewInstallationSymlink(const StartupOptions& startup_options) {
  const blaze_util::Path install_base_symlink =
      startup_options.output_base.GetRelative("install");
  blaze_util::Path prev_install_base;
  return !blaze_util::ReadDirectorySymlink(install_base_symlink,
                                           &prev_install_base) ||
         !blaze_util::ArePathsEquivalent(prev_install_base,
                                         startup_options.install_base);
}

void UpdateInstallationSymlink(const StartupOptions& startup_options) {
  const blaze_util::Path install_base_symlink =
      startup_options.output_base.GetRelative("install");
  blaze_util::UnlinkPath(install_base_symlink);
  if (!SymlinkDirectories(startup_options.install_base, install_base_symlink)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "failed to create installation symlink '"
        << install_base_symlink.AsPrintablePath()
        << "': " << GetLastErrorString();
  }

  // Update the mtime of the install base so that cleanup tools can
  // find install bases that haven't been used for a long time.
  // Ignore permissions errors (i.e. if the install base is not writable).
  if (!SetMtimeToNowIfPossible(
          blaze_util::Path(startup_options.install_base))) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "failed to set timestamp on '"
        << startup_options.install_base.AsPrintablePath()
        << "': " << GetLastErrorString();
  }
}

// Parse the options.
static void ParseOptionsOrDie(const string& cwd, const string& workspace,
                              OptionProcessor& option_processor, int argc,
                              const char* const* argv, int max_import_depth) {
  std::string error;
  std::vector<std::string> args(argv, argv + argc);
  const blaze_exit_code::ExitCode parse_exit_code =
      option_processor.ParseOptions(args, workspace, cwd, &error,
                                    max_import_depth);

  if (parse_exit_code != blaze_exit_code::SUCCESS) {
    option_processor.PrintStartupOptionsProvenanceMessage();
    BAZEL_DIE(parse_exit_code) << error;
  }
}

string GetCanonicalCwd() {
  string result = blaze_util::MakeCanonical(blaze_util::GetCwd().c_str());
  if (result.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "blaze_util::MakeCanonical('" << blaze_util::GetCwd()
        << "') failed: " << GetLastErrorString();
  }
  return result;
}

void PrepareDirectories(StartupOptions* startup_options) {
  blaze::CreateSecureDirectory(
      blaze_util::Path(startup_options->output_user_root));

  if (!blaze_util::PathExists(startup_options->output_base)) {
    if (!blaze_util::MakeDirectories(startup_options->output_base, 0777)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "Output base directory '"
          << startup_options->output_base.AsPrintablePath()
          << "' could not be created: " << GetLastErrorString();
    }
  } else {
    if (!blaze_util::IsDirectory(startup_options->output_base)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "Output base directory '"
          << startup_options->output_base.AsPrintablePath()
          << "' could not be created. It exists but is not a directory.";
    }
  }
  if (!blaze_util::CanAccessDirectory(startup_options->output_base)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Output base directory '"
        << startup_options->output_base.AsPrintablePath()
        << "' must be readable and writable.";
  }

  ExcludePathFromBackup(startup_options->output_base);

  startup_options->output_base = startup_options->output_base.Canonicalize();
  if (startup_options->output_base.IsEmpty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Output base directory could not be canonicalized: "
        << GetLastErrorString();
  }
}

// Prepares the environment to be suitable to start a JVM.
// Changes made to the environment in this function *will not* be part
// of '--client_env'.
map<string, EnvVarValue> PrepareEnvironmentForJvm() {
  map<string, EnvVarValue> result;

  // Make sure all existing environment variables appear as part of the
  // resulting map unless they are overridden below by UNSET values.
  //
  // Even though the map we return is intended to represent a "delta" of
  // environment variables to modify the current process, we may actually use
  // such map to configure a process from scratch (via interfaces like execvpe
  // or posix_spawn), so we need to inherit any untouched variables.
  for (const auto& var_value : blaze::GetProcessedEnv()) {
    std::string::size_type equals = var_value.find('=');
    if (equals == std::string::npos) {
      // Ignore possibly-bad environment. We don't control what we see in this
      // global variable, so it could be invalid.
      continue;
    }
    const std::string var = var_value.substr(0, equals);
    const std::string value = var_value.substr(equals + 1);
    result[var] = EnvVarValue(EnvVarAction::SET, value);
  }

  if (blaze::ExistsEnv("LD_ASSUME_KERNEL")) {
    // Fix for bug: if ulimit -s and LD_ASSUME_KERNEL are both
    // specified, the JVM fails to create threads.  See thread_stack_regtest.
    // This is also provoked by LD_LIBRARY_PATH=/usr/lib/debug,
    // or anything else that causes the JVM to use LinuxThreads.
    BAZEL_LOG(WARNING) << "ignoring LD_ASSUME_KERNEL in environment.";
    result["LD_ASSUME_KERNEL"] = EnvVarValue(EnvVarAction::UNSET, "");
  }

  if (blaze::ExistsEnv("LD_PRELOAD")) {
    BAZEL_LOG(WARNING) << "ignoring LD_PRELOAD in environment.";
    result["LD_PRELOAD"] = EnvVarValue(EnvVarAction::UNSET, "");
  }

  // These would override --host_jvm_args
  for (const auto& var :
       {"_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "JAVA_TOOL_OPTIONS"}) {
    if (blaze::ExistsEnv(var)) {
      BAZEL_LOG(WARNING) << "ignoring " << var << " in environment.";
      result[var] = EnvVarValue(EnvVarAction::UNSET, "");
    }
  }

  // TODO(bazel-team):  We've also seen a failure during loading (creating
  // threads?) when ulimit -Hs 8192.  Characterize that and check for it here.

  // Ensure that the JVM runs with a locale that supports Unicode characters in
  // filenames, environment variables, etc. The approach differs by OS:
  // - On macOS, the JVM always uses UTF-8, so we don't need to do anything.
  // - On Windows, the JVM uses the system code page to determine the encoding.
  //   For the embedded JDK, we force this to UTF-8 in minimize_jdk.sh.
  // - On Linux, the JVM goes through the regular locale mechanism. In
  //   particular, we can only force UTF-8 if we can find a locale that supports
  //   it. Furthermore, for backwards compatibility with setups using other
  //   encodings, we pick a Latin-1 locale if possible to support arbitrary byte
  //   sequences, not just valid UTF-8.
#ifdef __linux__
  const char* want_locale = nullptr;
  for (auto candidate_locale : {"en_US.ISO-8859-1", "C.UTF-8", "en_US.UTF-8"}) {
    locale_t locale = newlocale(LC_CTYPE_MASK, candidate_locale, nullptr);
    if (locale != nullptr) {
      freelocale(locale);
      want_locale = candidate_locale;
      break;
    }
  }

  if (want_locale != nullptr) {
    result["LC_ALL"] = EnvVarValue(EnvVarAction::SET, want_locale);
  }
#endif

  return result;
}

// Replace this process with the blaze server. Does not exit.
void RunServerMode(const string& install_md5,
                   const WorkspaceLayout& workspace_layout,
                   const string& workspace,
                   const StartupOptions& startup_options) {
  if (startup_options.batch) {
    BAZEL_DIE(blaze_exit_code::BAD_ARGV)
        << "exec-server command is not compatible with --batch";
  }

  const blaze_util::Path jvm_path = startup_options.GetJvm();
  const string server_jar_path = startup_options.GetServerJarFilename();
  const blaze_util::Path server_exe =
      startup_options.GetExe(jvm_path, server_jar_path);
  vector<string> server_exe_args =
      GetServerExeArgs(jvm_path, server_jar_path, install_md5, workspace_layout,
                       workspace, startup_options);
  const blaze_util::Path server_dir =
      blaze_util::Path(startup_options.output_base).GetRelative("server");

  BAZEL_LOG(INFO) << "Running in server mode.";

  int server_pid = GetServerPid(server_dir);
  if (server_pid > 0 &&
      VerifyServerProcess(server_pid, startup_options.output_base)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "exec-server failed, please shut down existing server pid="
        << server_pid << " and retry.";
  }

  if (NeedsNewInstallationSymlink(startup_options)) {
    UpdateInstallationSymlink(startup_options);
  }

  EnsureServerDir(server_dir);

  // Remove any stale server.starttime file from a previous daemon run so
  // VerifyServerProcess checks /proc/<pid>/stat without comparing against an
  // outdated start time.
  (void)blaze_util::UnlinkPath(server_dir.GetRelative("server.starttime"));
  blaze_util::WriteFile(blaze::GetProcessIdAsString(),
                        server_dir.GetRelative("server.pid.txt"));
  blaze_util::WriteFile(GetArgumentString(server_exe_args),
                        server_dir.GetRelative("cmdline"));

  GoToWorkspace(workspace_layout, workspace);

  SetScheduling(startup_options.batch_cpu_scheduling,
                startup_options.io_nice_level);

  {
    WithEnvVars env_obj(PrepareEnvironmentForJvm());
#ifdef __linux__
    bool run_in_user_cgroup = startup_options.run_in_user_cgroup;
#else
    bool run_in_user_cgroup = false;
#endif
    ExecuteServerJvm(server_exe, server_exe_args,
                     GetServerJvmArgFilePath(startup_options.output_base),
                     run_in_user_cgroup);
  }
}

StartupOptions* SetupAndParseOptions(int argc, const char* const* argv,
                                     const string& cwd, const string& workspace,
                                     WorkspaceLayout* workspace_layout,
                                     OptionProcessor* option_processor) {
  // Best-effort operation to raise the resource limits from soft to hard.  We
  // do this early during the main program instead of just before execing the
  // Blaze server binary, because it's easier (for testing purposes) and because
  // the Blaze client also benefits from this (e.g. during installation).
  UnlimitResources();

#if defined(_WIN32) || defined(__CYGWIN__)
  // Must be done before command line parsing.
  // ParseOptionsOrDie already populate --client_env, so detect bash before it
  // happens.
  (void)DetectBashAndExportBazelSh();
#endif  // if defined(_WIN32) || defined(__CYGWIN__)

  if (blaze::IsRunningWithinTest()) {
    BAZEL_LOG(USER) << "$TEST_TMPDIR defined, some defaults will be overridden";
  }
  int max_import_depth = blaze::SelectMaxImportDepth();
  ParseOptionsOrDie(cwd, workspace, *option_processor, argc, argv,
                    max_import_depth);
  StartupOptions* startup_options = option_processor->GetParsedStartupOptions();
  startup_options->MaybeLogStartupOptionWarnings();

  if (startup_options->client_debug) {
    SetDebugLog(blaze_util::LOGGINGDETAIL_DEBUG);
  } else if (startup_options->quiet) {
    SetDebugLog(blaze_util::LOGGINGDETAIL_QUIET);
  } else {
    SetDebugLog(blaze_util::LOGGINGDETAIL_USER);
  }

  // If client_debug was false, this is ignored, so it's accurate.
  BAZEL_LOG(INFO) << "Debug logging requested, sending all client log "
                     "statements to stderr";

  BAZEL_LOG(INFO) << "Running (pid=" << GetProcessIdAsString() << ")";

  if (startup_options->unlimit_coredumps) {
    UnlimitCoredumps();
  }

  if (!workspace_layout->InWorkspace(workspace)) {
    const string command = option_processor->GetCommand();
    if (!CanRunOutsideWorkspace(command)) {
      BAZEL_DIE(blaze_exit_code::BAD_ARGV)
          << startup_options->product_name
          << (command.empty() ? "" : " '" + command + "'")
          << " must be invoked from within a workspace (below a"
          << " directory having a MODULE.bazel file).\n"
          << "See documentation at"
          << " https://bazel.build/concepts/build-ref#workspace";
    }
    startup_options->batch = true;
  }

  return startup_options;
}

int ExecServer(int argc, const char* const* argv,
               WorkspaceLayout* workspace_layout,
               OptionProcessor* option_processor) {
  blaze_util::InitializeStdOutErrForUtf8();

  // Logging must be set first to assure no log statements are missed.
  std::unique_ptr<blaze_util::BazelLogHandler> default_handler(
      new blaze_util::BazelLogHandler());
  blaze_util::SetLogHandler(std::move(default_handler));

  string cwd = GetCanonicalCwd();
  blaze::SetupStdStreams();

  const string workspace = workspace_layout->GetWorkspace(cwd);
  StartupOptions* startup_options = SetupAndParseOptions(
      argc, argv, cwd, workspace, workspace_layout, option_processor);

  if (!IsServerMode(option_processor->GetCommand())) {
    BAZEL_DIE(blaze_exit_code::BAD_ARGV)
        << "Expected 'exec-server' command, got '"
        << option_processor->GetCommand() << "'";
  }

  startup_options->UpdateConfiguration("", workspace, /*server_mode=*/true);
  PrepareDirectories(startup_options);

  auto [client_locks, command_wait_duration] =
      AcquireClientLocks(*startup_options);
  BAZEL_LOG(INFO) << "Acquired the client lock, waited "
                  << command_wait_duration.millis << " milliseconds";

  WarnFilesystemType(startup_options->output_base);

  if (!blaze_util::IsDirectory(startup_options->install_base)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Install base directory '"
        << startup_options->install_base.AsPrintablePath()
        << "' does not exist or is not a directory.";
  }

  string install_md5;
  blaze_util::Path install_base_key_path =
      startup_options->install_base.GetRelative("install_base_key");
  if (!blaze_util::ReadFile(install_base_key_path, &install_md5)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to read '" << install_base_key_path.AsPrintablePath()
        << "': " << GetLastErrorString();
  }

  RunServerMode(install_md5, *workspace_layout, workspace, *startup_options);
  return 0;
}

}  // namespace blaze
