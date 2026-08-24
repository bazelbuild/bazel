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

#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "tools/test/posix/tw_outputs.h"

namespace bazel {
namespace tools {
namespace test_wrapper {
namespace {

constexpr char kPagerHeader[] =
    "exec ${PAGER:-/usr/bin/less} \"$0\" || exit 1\n";
constexpr char kTestLogDelimiter[] =
    "--------------------------------------------------------------------------"
    "---\n";
constexpr char kMissingRunfilesDirectory[] =
    "ERROR: RUNFILES_DIR does not exist. This can happen when using "
    "--nobuild_runfile_manifests with local execution. Use a different "
    "execution strategy, or build with runfile manifests.";
constexpr char kRlocationFunctionName[] = "BASH_FUNC_rlocation%%";
constexpr char kRlocationFunction[] =
    "() {  caller 0 | { read LINE SUB FILE; "
    "echo >&2 \"ERROR: rlocation is no longer implicitly provided by Bazel's "
    "test setup, but called from $SUB in line $LINE of $FILE. Please use "
    "https://github.com/bazelbuild/rules_shell/blob/main/shell/runfiles/"
    "runfiles.bash instead.\"; exit 1; }; }";

constexpr std::array<int, 17> kForwardedSignals = {
    SIGHUP,   SIGINT,  SIGQUIT, SIGTERM,   SIGALRM, SIGUSR1,
    SIGUSR2,  SIGPIPE, SIGPROF, SIGVTALRM, SIGXCPU, SIGXFSZ,
    SIGWINCH, SIGTSTP, SIGTTIN, SIGTTOU,   SIGCONT,
};

volatile sig_atomic_t g_child_process_group = 0;
volatile sig_atomic_t g_received_sigterm = 0;

std::string GetEnv(const char* name) {
  const char* value = getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

bool SetEnv(const char* name, const std::string& value) {
  if (setenv(name, value.c_str(), 1) == 0) {
    return true;
  }
  fprintf(stderr, "ERROR: could not set %s: %s\n", name, strerror(errno));
  return false;
}

bool UnsetEnv(const char* name) {
  if (unsetenv(name) == 0) {
    return true;
  }
  fprintf(stderr, "ERROR: could not unset %s: %s\n", name, strerror(errno));
  return false;
}

bool IsAbsolute(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  if (path[0] == '/') {
    return true;
  }
  return path.size() >= 3 &&
         ((path[0] >= 'a' && path[0] <= 'z') ||
          (path[0] >= 'A' && path[0] <= 'Z')) &&
         path[1] == ':' && (path[2] == '/' || path[2] == '\\');
}

std::string JoinPath(const std::string& parent, const std::string& child) {
  if (parent.empty() || IsAbsolute(child)) {
    return child;
  }
  if (parent.back() == '/') {
    return parent + child;
  }
  return parent + "/" + child;
}

std::string Dirname(std::string path) {
  while (path.size() > 1 && path.back() == '/') {
    path.pop_back();
  }
  const std::string::size_type separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return ".";
  }
  return separator == 0 ? "/" : path.substr(0, separator);
}

bool GetCurrentDirectory(std::string* result) {
  const std::string logical_directory = GetEnv("PWD");
  struct stat logical_info;
  struct stat current_info;
  if (IsAbsolute(logical_directory) &&
      stat(logical_directory.c_str(), &logical_info) == 0 &&
      stat(".", &current_info) == 0 &&
      logical_info.st_dev == current_info.st_dev &&
      logical_info.st_ino == current_info.st_ino) {
    *result = logical_directory;
    return true;
  }

  size_t buffer_size = 256;
  while (buffer_size <= 1024 * 1024) {
    std::vector<char> buffer(buffer_size);
    if (getcwd(buffer.data(), buffer.size()) != nullptr) {
      *result = buffer.data();
      return true;
    }
    if (errno != ERANGE) {
      break;
    }
    buffer_size *= 2;
  }
  fprintf(stderr, "ERROR: could not determine current directory: %s\n",
          strerror(errno));
  return false;
}

bool IsDirectory(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool PathExists(const std::string& path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

bool PathOrSymlinkExists(const std::string& path) {
  struct stat info;
  return lstat(path.c_str(), &info) == 0;
}

bool MakeDirectories(const std::string& path) {
  if (path.empty()) {
    return true;
  }
  std::string current;
  if (path[0] == '/') {
    current = "/";
  }
  size_t position = current.size();
  while (position < path.size()) {
    const size_t separator = path.find('/', position);
    const size_t length = separator == std::string::npos
                              ? path.size() - position
                              : separator - position;
    if (length != 0) {
      current = JoinPath(current, path.substr(position, length));
      if (mkdir(current.c_str(), 0777) != 0 &&
          (errno != EEXIST || !IsDirectory(current))) {
        fprintf(stderr, "ERROR: could not create directory %s: %s\n",
                current.c_str(), strerror(errno));
        return false;
      }
    }
    if (separator == std::string::npos) {
      break;
    }
    position = separator + 1;
  }
  return true;
}

bool AbsolutizeEnv(const char* name, const std::string& exec_root,
                   bool preserve_empty, std::string* result = nullptr) {
  std::string value = GetEnv(name);
  if (!(preserve_empty && value.empty()) && !IsAbsolute(value)) {
    value = JoinPath(exec_root, value);
  }
  if (result != nullptr) {
    *result = value;
  }
  return SetEnv(name, value);
}

bool ExportUserName() {
  if (!GetEnv("USER").empty()) {
    return true;
  }
  long suggested_size = sysconf(_SC_GETPW_R_SIZE_MAX);
  size_t buffer_size =
      suggested_size <= 0 ? 16384 : static_cast<size_t>(suggested_size);
  buffer_size = std::min<size_t>(buffer_size, 1024 * 1024);

  while (buffer_size <= 1024 * 1024) {
    std::vector<char> buffer(buffer_size);
    struct passwd account;
    struct passwd* result = nullptr;
    const int status =
        getpwuid_r(getuid(), &account, buffer.data(), buffer.size(), &result);
    if (status == 0 && result != nullptr) {
      return SetEnv("USER", account.pw_name);
    }
    if (status != ERANGE) {
      break;
    }
    buffer_size *= 2;
  }

  const std::string login = GetEnv("LOGNAME");
  if (!login.empty()) {
    return SetEnv("USER", login);
  }
  return SetEnv("USER", std::to_string(static_cast<unsigned long>(getuid())));
}

bool ExportGtestVariables(const std::string& test_tmpdir) {
  const char* total_shards = getenv("TEST_TOTAL_SHARDS");
  if (total_shards != nullptr && total_shards[0] != '\0' &&
      strcmp(total_shards, "0") != 0) {
    if (!SetEnv("GTEST_SHARD_INDEX", GetEnv("TEST_SHARD_INDEX")) ||
        !SetEnv("GTEST_TOTAL_SHARDS", total_shards) ||
        !SetEnv("GTEST_SHARD_STATUS_FILE", GetEnv("TEST_SHARD_STATUS_FILE"))) {
      return false;
    }
  }
  return SetEnv("GTEST_TMP_DIR", test_tmpdir);
}

bool PrepareEnvironment(const std::string& exec_root,
                        UndeclaredOutputs* undeclared_outputs,
                        std::string* test_srcdir) {
  if (!SetEnv("BAZEL_TEST", "1")) {
    return false;
  }

  for (const char* name :
       {"TEST_PREMATURE_EXIT_FILE", "TEST_WARNINGS_OUTPUT_FILE",
        "TEST_LOGSPLITTER_OUTPUT_FILE", "TEST_INFRASTRUCTURE_FAILURE_FILE",
        "TEST_UNUSED_RUNFILES_LOG_FILE"}) {
    if (!AbsolutizeEnv(name, exec_root, false)) {
      return false;
    }
  }

  if (!AbsolutizeEnv("TEST_UNDECLARED_OUTPUTS_DIR", exec_root, false,
                     &undeclared_outputs->root) ||
      !AbsolutizeEnv("TEST_UNDECLARED_OUTPUTS_MANIFEST", exec_root, false,
                     &undeclared_outputs->manifest) ||
      !AbsolutizeEnv("TEST_UNDECLARED_OUTPUTS_ZIP", exec_root, true,
                     &undeclared_outputs->zip) ||
      !AbsolutizeEnv("TEST_UNDECLARED_OUTPUTS_ANNOTATIONS", exec_root, false,
                     &undeclared_outputs->annotations) ||
      !AbsolutizeEnv("TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR", exec_root,
                     false, &undeclared_outputs->annotations_dir) ||
      !AbsolutizeEnv("TEST_SRCDIR", exec_root, false, test_srcdir)) {
    return false;
  }

  std::string test_tmpdir;
  if (!AbsolutizeEnv("TEST_TMPDIR", exec_root, false, &test_tmpdir)) {
    return false;
  }
  if (!IsAbsolute(GetEnv("HOME")) && !SetEnv("HOME", test_tmpdir)) {
    return false;
  }

  std::string xml_output;
  if (!AbsolutizeEnv("XML_OUTPUT_FILE", exec_root, false, &xml_output) ||
      !SetEnv("GUNIT_OUTPUT", "xml:" + xml_output) || !ExportUserName()) {
    return false;
  }

  const std::string shard_status = GetEnv("TEST_SHARD_STATUS_FILE");
  if (!shard_status.empty()) {
    std::string absolute_shard_status;
    if (!AbsolutizeEnv("TEST_SHARD_STATUS_FILE", exec_root, false,
                       &absolute_shard_status) ||
        !MakeDirectories(Dirname(absolute_shard_status))) {
      return false;
    }
  }

  std::string runfiles_dir;
  if (!AbsolutizeEnv("RUNFILES_DIR", exec_root, false, &runfiles_dir)) {
    return false;
  }
  if (!IsDirectory(runfiles_dir)) {
    fprintf(stderr, "%s\n", kMissingRunfilesDirectory);
    return false;
  }
  if (!AbsolutizeEnv("JAVA_RUNFILES", exec_root, false) ||
      !AbsolutizeEnv("PYTHON_RUNFILES", exec_root, false) ||
      !MakeDirectories(Dirname(xml_output)) ||
      !MakeDirectories(undeclared_outputs->root) ||
      !MakeDirectories(undeclared_outputs->annotations_dir) ||
      !MakeDirectories(test_tmpdir) ||
      !UnsetEnv("TEST_UNDECLARED_OUTPUTS_MANIFEST") ||
      !UnsetEnv("TEST_UNDECLARED_OUTPUTS_ZIP") ||
      !UnsetEnv("TEST_UNDECLARED_OUTPUTS_ANNOTATIONS") ||
      !ExportGtestVariables(test_tmpdir)) {
    return false;
  }

  const std::string runfiles_manifest = JoinPath(*test_srcdir, "MANIFEST");
  const bool manifest_was_exported =
      getenv("RUNFILES_MANIFEST_FILE") != nullptr;
  if ((manifest_was_exported || (GetEnv("RUNFILES_MANIFEST_ONLY") == "1" &&
                                 PathExists(runfiles_manifest))) &&
      !SetEnv("RUNFILES_MANIFEST_FILE", runfiles_manifest)) {
    return false;
  }
  return true;
}

bool ChangeToRunfiles(const std::string& exec_root,
                      const std::string& test_srcdir, bool coverage_mode) {
  if (coverage_mode) {
    return true;
  }

  std::string directory = test_srcdir;
  const std::string test_workspace = GetEnv("TEST_WORKSPACE");
  if (!test_workspace.empty()) {
    directory = JoinPath(directory, test_workspace);
  }
  if (!GetEnv("RUNTEST_PRESERVE_CWD").empty()) {
    directory = exec_root;
  }
  if (chdir(directory.c_str()) != 0) {
    fprintf(stderr, "Could not chdir %s: %s\n", directory.c_str(),
            strerror(errno));
    return false;
  }
  return SetEnv("PWD", directory);
}

bool ResolveRunfile(const std::string& test_srcdir, const std::string& runfile,
                    std::string* result) {
  if (IsAbsolute(runfile)) {
    *result = runfile;
    return true;
  }

  const std::string direct_path = JoinPath(test_srcdir, runfile);
  if (PathExists(direct_path)) {
    *result = direct_path;
    return true;
  }

  std::ifstream manifest(JoinPath(test_srcdir, "MANIFEST"));
  std::string line;
  const std::string prefix = runfile + " ";
  while (std::getline(manifest, line)) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      *result = line.substr(prefix.size());
      return true;
    }
  }

  fprintf(stderr, "ERROR: could not resolve test executable %s\n",
          runfile.c_str());
  return false;
}

bool ResolveTestPath(const std::string& test_srcdir, std::string executable,
                     std::string* result) {
  if (executable.compare(0, 2, "./") == 0) {
    executable.erase(0, 2);
  }
  if (IsAbsolute(executable)) {
    *result = std::move(executable);
    return true;
  }
  if (executable.compare(0, 3, "../") == 0) {
    executable.erase(0, 3);
  } else {
    executable = JoinPath(GetEnv("TEST_WORKSPACE"), executable);
  }
  return ResolveRunfile(test_srcdir, executable, result);
}

bool ShortenTestPath(const std::string& exec_root, std::string* test_path) {
  if (GetEnv("TEST_SHORT_EXEC_PATH").empty()) {
    return true;
  }

  unsigned int qualifier = 0;
  std::string base;
  do {
    base = JoinPath(exec_root, "t" + std::to_string(qualifier++));
  } while (PathOrSymlinkExists(base) || PathOrSymlinkExists(base + ".exe") ||
           PathOrSymlinkExists(base + ".zip"));

  std::string extensionless = *test_path;
  const size_t dot = extensionless.find_last_of('.');
  if (dot != std::string::npos) {
    extensionless.erase(dot);
  }
  (void)symlink(extensionless.c_str(), base.c_str());
  (void)symlink((extensionless + ".zip").c_str(), (base + ".zip").c_str());
  if (symlink(test_path->c_str(), (base + ".exe").c_str()) != 0) {
    fprintf(stderr, "ERROR: could not shorten test executable %s: %s\n",
            test_path->c_str(), strerror(errno));
    return false;
  }
  *test_path = base + ".exe";
  return true;
}

bool PrepareCommand(int argc, char** argv, const std::string& exec_root,
                    const std::string& test_srcdir, bool coverage_mode,
                    std::vector<std::string>* command) {
  const int executable_index = coverage_mode ? 2 : 1;
  if (argc <= executable_index) {
    fprintf(stderr,
            coverage_mode
                ? "Usage: %s <coverage_wrapper> <test_path> [test_args...]\n"
                : "Usage: %s <test_path> [test_args...]\n",
            argv[0]);
    return false;
  }

  std::string test_path;
  if (!ResolveTestPath(test_srcdir, argv[executable_index], &test_path) ||
      !ShortenTestPath(exec_root, &test_path)) {
    return false;
  }

  if (coverage_mode) {
    command->emplace_back(argv[1]);
  }
  command->push_back(std::move(test_path));
  for (int i = executable_index + 1; i < argc; ++i) {
    command->emplace_back(argv[i]);
  }
  return true;
}

std::vector<char*> CommandPointers(std::vector<std::string>* command) {
  std::vector<char*> arguments;
  arguments.reserve(command->size() + 1);
  for (std::string& argument : *command) {
    arguments.push_back(&argument[0]);
  }
  arguments.push_back(nullptr);
  return arguments;
}

void ForwardSignal(int signal_number) {
  const int saved_errno = errno;
  if (signal_number == SIGTERM) {
    g_received_sigterm = 1;
  }
  const sig_atomic_t process_group = g_child_process_group;
  if (process_group > 0) {
    (void)kill(-static_cast<pid_t>(process_group), signal_number);
  }
  errno = saved_errno;
}

class SignalHandlers {
 public:
  bool Install() {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = ForwardSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    for (size_t i = 0; i < kForwardedSignals.size(); ++i) {
      if (sigaction(kForwardedSignals[i], &action, &previous_[i]) != 0) {
        fprintf(stderr, "ERROR: could not handle signal %d: %s\n",
                kForwardedSignals[i], strerror(errno));
        Restore();
        return false;
      }
      ++installed_;
    }
    return true;
  }

  void Restore() {
    while (installed_ != 0) {
      --installed_;
      (void)sigaction(kForwardedSignals[installed_], &previous_[installed_],
                      nullptr);
    }
  }

  ~SignalHandlers() { Restore(); }

 private:
  std::array<struct sigaction, kForwardedSignals.size()> previous_;
  size_t installed_ = 0;
};

class ForwardedSignalMask {
 public:
  bool Block() {
    sigset_t blocked_signals;
    if (sigemptyset(&blocked_signals) != 0) {
      fprintf(stderr, "ERROR: could not initialize signal mask: %s\n",
              strerror(errno));
      return false;
    }
    for (int signal_number : kForwardedSignals) {
      if (sigaddset(&blocked_signals, signal_number) != 0) {
        fprintf(stderr, "ERROR: could not block signal %d: %s\n", signal_number,
                strerror(errno));
        return false;
      }
    }
    if (sigprocmask(SIG_BLOCK, &blocked_signals, &previous_) != 0) {
      fprintf(stderr, "ERROR: could not block forwarded signals: %s\n",
              strerror(errno));
      return false;
    }
    blocked_ = true;
    return true;
  }

  bool Restore() {
    if (!blocked_) {
      return true;
    }
    if (sigprocmask(SIG_SETMASK, &previous_, nullptr) != 0) {
      fprintf(stderr, "ERROR: could not restore signal mask: %s\n",
              strerror(errno));
      return false;
    }
    blocked_ = false;
    return true;
  }

  ~ForwardedSignalMask() { (void)Restore(); }

 private:
  sigset_t previous_;
  bool blocked_ = false;
};

bool CloseOnExec(int descriptor) {
  const int flags = fcntl(descriptor, F_GETFD);
  return flags != -1 && fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != -1;
}

void PrintTimeoutMessage() {
  time_t now = time(nullptr);
  struct tm local_time;
  char formatted[128];
  if (localtime_r(&now, &local_time) != nullptr &&
      strftime(formatted, sizeof(formatted), "%Y-%m-%d %H:%M:%S %Z",
               &local_time) != 0) {
    fprintf(stdout, "-- Test timed out at %s --\n", formatted);
  } else {
    fprintf(stdout, "-- Test timed out --\n");
  }
  fflush(stdout);
}

void ResetChildSignals() {
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = SIG_DFL;
  sigemptyset(&action.sa_mask);
  for (int signal_number : kForwardedSignals) {
    (void)sigaction(signal_number, &action, nullptr);
  }
  sigset_t unblocked;
  sigemptyset(&unblocked);
  (void)sigprocmask(SIG_SETMASK, &unblocked, nullptr);
}

void ExecuteTest(std::vector<std::string>* command, bool null_stdin) {
  if (null_stdin) {
    const int null_descriptor = open("/dev/null", O_RDONLY);
    if (null_descriptor == -1 || dup2(null_descriptor, STDIN_FILENO) == -1) {
      fprintf(stderr, "ERROR: could not configure test stdin: %s\n",
              strerror(errno));
      _exit(1);
    }
    if (null_descriptor != STDIN_FILENO) {
      close(null_descriptor);
    }
  }

  std::vector<char*> arguments = CommandPointers(command);
  execv(arguments[0], arguments.data());
  if (errno == ENOEXEC) {
    command->insert(command->begin(), "bash");
    arguments = CommandPointers(command);
    execvp(arguments[0], arguments.data());
  }
  const int execution_error = errno;
  fprintf(stderr, "ERROR: could not execute %s: %s\n", arguments[0],
          strerror(execution_error));
  _exit(execution_error == ENOENT ? 127 : 126);
}

void WatchParent(int descriptor, pid_t child_process_group) {
  (void)setpgid(0, 0);
  char byte = 0;
  ssize_t result;
  do {
    result = read(descriptor, &byte, 1);
  } while (result == -1 && errno == EINTR);
  if (result <= 0) {
    (void)kill(-child_process_group, SIGKILL);
  }
  close(descriptor);
  _exit(0);
}

bool WaitForProcess(pid_t process, int* status) {
  while (waitpid(process, status, 0) == -1) {
    if (errno != EINTR) {
      fprintf(stderr, "ERROR: could not wait for test process: %s\n",
              strerror(errno));
      return false;
    }
    if (g_received_sigterm != 0) {
      g_received_sigterm = 0;
      PrintTimeoutMessage();
    }
  }
  if (g_received_sigterm != 0) {
    g_received_sigterm = 0;
    PrintTimeoutMessage();
  }
  return true;
}

bool RunTest(std::vector<std::string>* command, int* exit_code) {
  int watchdog_pipe[2] = {-1, -1};
  if (pipe(watchdog_pipe) != 0 || !CloseOnExec(watchdog_pipe[0]) ||
      !CloseOnExec(watchdog_pipe[1])) {
    fprintf(stderr, "ERROR: could not create process cleanup pipe: %s\n",
            strerror(errno));
    if (watchdog_pipe[0] != -1) {
      close(watchdog_pipe[0]);
    }
    if (watchdog_pipe[1] != -1) {
      close(watchdog_pipe[1]);
    }
    return false;
  }

  ForwardedSignalMask signal_mask;
  if (!signal_mask.Block()) {
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    return false;
  }

  SignalHandlers handlers;
  if (!handlers.Install()) {
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    return false;
  }

  const pid_t child = fork();
  if (child < 0) {
    fprintf(stderr, "ERROR: could not start test process: %s\n",
            strerror(errno));
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    return false;
  }
  if (child == 0) {
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    if (setpgid(0, 0) != 0) {
      fprintf(stderr, "ERROR: could not create test process group: %s\n",
              strerror(errno));
      _exit(1);
    }
    ResetChildSignals();
    ExecuteTest(command, true);
  }

  g_child_process_group = static_cast<sig_atomic_t>(child);
  if (setpgid(child, child) != 0 && errno != EACCES && errno != ESRCH) {
    fprintf(stderr, "ERROR: could not configure test process group: %s\n",
            strerror(errno));
    (void)kill(child, SIGKILL);
    int ignored;
    (void)waitpid(child, &ignored, 0);
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    g_child_process_group = 0;
    return false;
  }
  if (!signal_mask.Restore()) {
    (void)kill(-child, SIGKILL);
    int ignored;
    (void)waitpid(child, &ignored, 0);
    close(watchdog_pipe[0]);
    close(watchdog_pipe[1]);
    g_child_process_group = 0;
    return false;
  }

  const pid_t watchdog = fork();
  if (watchdog == 0) {
    close(watchdog_pipe[1]);
    ResetChildSignals();
    WatchParent(watchdog_pipe[0], child);
  }
  close(watchdog_pipe[0]);
  if (watchdog < 0) {
    fprintf(stderr, "ERROR: could not start process cleanup: %s\n",
            strerror(errno));
    (void)kill(-child, SIGKILL);
    int ignored;
    (void)waitpid(child, &ignored, 0);
    close(watchdog_pipe[1]);
    g_child_process_group = 0;
    return false;
  }

  int status = 0;
  const bool waited = WaitForProcess(child, &status);
  (void)kill(-child, SIGKILL);
  g_child_process_group = 0;

  const char normal_exit = 1;
  ssize_t written;
  do {
    written = write(watchdog_pipe[1], &normal_exit, sizeof(normal_exit));
  } while (written == -1 && errno == EINTR);
  close(watchdog_pipe[1]);

  int watchdog_status;
  while (waitpid(watchdog, &watchdog_status, 0) == -1 && errno == EINTR) {
  }
  handlers.Restore();
  if (!waited) {
    return false;
  }

  if (WIFEXITED(status)) {
    *exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    *exit_code = 128 + WTERMSIG(status);
  } else {
    fprintf(stderr, "ERROR: unexpected test process status %d\n", status);
    return false;
  }
  return true;
}

}  // namespace

int TestWrapperMain(int argc, char** argv) {
  if (dup2(STDOUT_FILENO, STDERR_FILENO) == -1) {
    return 1;
  }
  setvbuf(stdout, nullptr, _IONBF, 0);

  fputs(kPagerHeader, stdout);
  fprintf(stdout, "Executing tests from %s\n", GetEnv("TEST_TARGET").c_str());

  std::string exec_root;
  std::string test_srcdir;
  UndeclaredOutputs undeclared_outputs;
  if (!GetCurrentDirectory(&exec_root) ||
      !PrepareEnvironment(exec_root, &undeclared_outputs, &test_srcdir)) {
    return 1;
  }

  const bool coverage_mode = !GetEnv("COVERAGE_DIR").empty();
  if (!ChangeToRunfiles(exec_root, test_srcdir, coverage_mode)) {
    return 1;
  }

  fputs(kTestLogDelimiter, stdout);
  if (!SetEnv("PATH", ".:" + GetEnv("PATH")) ||
      !SetEnv(kRlocationFunctionName, kRlocationFunction)) {
    return 1;
  }

  std::vector<std::string> command;
  if (!PrepareCommand(argc, argv, exec_root, test_srcdir, coverage_mode,
                      &command)) {
    return 1;
  }

  if (!GetEnv("BUILD_EXECROOT").empty()) {
    ExecuteTest(&command, false);
  }

  int exit_code = 1;
  if (!RunTest(&command, &exit_code)) {
    return 1;
  }

  std::string error;
  if (!ProcessUndeclaredOutputs(undeclared_outputs, &error)) {
    fprintf(stderr, "ERROR: could not process undeclared test outputs: %s\n",
            error.c_str());
    return 1;
  }

  if (exit_code > 128 && exit_code - 128 < NSIG) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    const int signal_number = exit_code - 128;
    if (signal_number == SIGKILL || signal_number == SIGSTOP ||
        sigaction(signal_number, &action, nullptr) == 0) {
      raise(signal_number);
    }
  }
  return exit_code;
}

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel
