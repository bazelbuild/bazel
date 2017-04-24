// Copyright 2015 The Bazel Authors. All rights reserved.
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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>  // PATH_MAX
#include <pwd.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/global_variables.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/md5.h"
#include "src/main/cpp/util/numbers.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;
using blaze_exit_code::INTERNAL_ERROR;

using std::string;
using std::vector;

SignalHandler SignalHandler::INSTANCE;

// The number of the last received signal that should cause the client
// to shutdown.  This is saved so that the client's WTERMSIG can be set
// correctly.  (Currently only SIGPIPE uses this mechanism.)
static volatile sig_atomic_t signal_handler_received_signal = 0;

// Signal handler.
static void handler(int signum) {
  int saved_errno = errno;

  static volatile sig_atomic_t sigint_count = 0;

  switch (signum) {
    case SIGINT:
      if (++sigint_count >= 3) {
        SigPrintf(
            "\n%s caught third interrupt signal; killed.\n\n",
            SignalHandler::Get().GetGlobals()->options->product_name.c_str());
        if (SignalHandler::Get().GetGlobals()->server_pid != -1) {
          KillServerProcess(SignalHandler::Get().GetGlobals()->server_pid);
        }
        _exit(1);
      }
      SigPrintf(
          "\n%s caught interrupt signal; shutting down.\n\n",
          SignalHandler::Get().GetGlobals()->options->product_name.c_str());
      SignalHandler::Get().CancelServer();
      break;
    case SIGTERM:
      SigPrintf(
          "\n%s caught terminate signal; shutting down.\n\n",
          SignalHandler::Get().GetGlobals()->options->product_name.c_str());
      SignalHandler::Get().CancelServer();
      break;
    case SIGPIPE:
      signal_handler_received_signal = SIGPIPE;
      break;
    case SIGQUIT:
      SigPrintf("\nSending SIGQUIT to JVM process %d (see %s).\n\n",
                SignalHandler::Get().GetGlobals()->server_pid,
                SignalHandler::Get().GetGlobals()->jvm_log_file.c_str());
      kill(SignalHandler::Get().GetGlobals()->server_pid, SIGQUIT);
      break;
  }

  errno = saved_errno;
}

void SignalHandler::Install(GlobalVariables* globals,
                            SignalHandler::Callback cancel_server) {
  _globals = globals;
  _cancel_server = cancel_server;

  // Unblock all signals.
  sigset_t sigset;
  sigemptyset(&sigset);
  sigprocmask(SIG_SETMASK, &sigset, NULL);

  signal(SIGINT, handler);
  signal(SIGTERM, handler);
  signal(SIGPIPE, handler);
  signal(SIGQUIT, handler);
}

ATTRIBUTE_NORETURN void SignalHandler::PropagateSignalOrExit(int exit_code) {
  if (signal_handler_received_signal) {
    // Kill ourselves with the same signal, so that callers see the
    // right WTERMSIG value.
    signal(signal_handler_received_signal, SIG_DFL);
    raise(signal_handler_received_signal);
    exit(1);  // (in case raise didn't kill us for some reason)
  } else {
    exit(exit_code);
  }
}

string GetProcessIdAsString() {
  return ToString(getpid());
}

string GetHomeDir() { return GetEnv("HOME"); }

string FindSystemWideBlazerc() {
  string path = "/etc/bazel.bazelrc";
  if (blaze_util::CanReadFile(path)) {
    return path;
  }
  return "";
}

string GetJavaBinaryUnderJavabase() { return "bin/java"; }

void ExecuteProgram(const string &exe, const vector<string> &args_vector) {
  if (VerboseLogging()) {
    string dbg;
    for (const auto &s : args_vector) {
      dbg.append(s);
      dbg.append(" ");
    }

    string cwd = blaze_util::GetCwd();
    fprintf(stderr, "Invoking binary %s in %s:\n  %s\n", exe.c_str(),
            cwd.c_str(), dbg.c_str());
  }

  // Copy to a char* array for execv:
  int n = args_vector.size();
  const char **argv = new const char *[n + 1];
  for (int i = 0; i < n; ++i) {
    argv[i] = args_vector[i].c_str();
  }
  argv[n] = NULL;

  execv(exe.c_str(), const_cast<char **>(argv));
}

std::string ConvertPath(const std::string &path) { return path; }

std::string ConvertPathList(const std::string& path_list) { return path_list; }

std::string PathAsJvmFlag(const std::string& path) { return path; }

const char kListSeparator = ':';

bool SymlinkDirectories(const string &target, const string &link) {
  return symlink(target.c_str(), link.c_str()) == 0;
}

static void CheckSingleThreaded() {
#ifdef __linux__
  DIR *dir = opendir("/proc/self/task");
  if (!dir) pdie(INTERNAL_ERROR, "can't list /proc/self/task");
  vector<string> tids;
  while (dirent *dent = readdir(dir)) {
    if (dent->d_name[0] != '.') tids.push_back(dent->d_name);
  }
  closedir(dir);
  if (tids.size() == 1) return;

  // If there are multiple threads, show their names as a debugging aid.
  fprintf(stderr, "Trying to fork, but found %zu threads:\n", tids.size());
  for (const string &t : tids) {
    string path = string("/proc/self/task/") + t + "/comm";
    if (FILE *f = fopen(path.c_str(), "r")) {
      char comm[4096];
      int len = fread(comm, 1, sizeof comm, f);
      fprintf(stderr, "  Thread %s: %.*s", t.c_str(), len, comm);
      fclose(f);
    } else {
      fprintf(stderr, "can't open %s", path.c_str());
    }
  }
  die(INTERNAL_ERROR, "can't fork() after creating threads");
#endif
  // This can probably be checked on darwin via <sys/proc_info.h>.
}

// Causes the current process to become a daemon (i.e. a child of
// init, detached from the terminal, in its own session.)  We don't
// change cwd, though.
static void Daemonize(const string& daemon_output) {
  // Don't call die() or exit() in this function; we're already in a
  // child process so it won't work as expected.  Just don't do
  // anything that can possibly fail. :)

  signal(SIGHUP, SIG_IGN);
  CheckSingleThreaded();
  if (fork() > 0) {
    // This second fork is required iff there's any chance cmd will
    // open an specific tty explicitly, e.g., open("/dev/tty23"). If
    // not, this fork can be removed.
    _exit(blaze_exit_code::SUCCESS);
  }

  setsid();

  close(0);
  close(1);
  close(2);

  open("/dev/null", O_RDONLY);  // stdin
  // stdout:
  if (open(daemon_output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666) == -1) {
    // In a daemon, no-one can hear you scream.
    open("/dev/null", O_WRONLY);
  }
  (void) dup(STDOUT_FILENO);  // stderr (2>&1)
}

class PipeBlazeServerStartup : public BlazeServerStartup {
 public:
  PipeBlazeServerStartup(int pipe_fd);
  virtual ~PipeBlazeServerStartup();
  virtual bool IsStillAlive();

 private:
  int pipe_fd;
};

PipeBlazeServerStartup::PipeBlazeServerStartup(int pipe_fd) {
  this->pipe_fd = pipe_fd;
  if (fcntl(pipe_fd, F_SETFL, O_NONBLOCK | fcntl(pipe_fd, F_GETFL))) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "Failed: fcntl to enable O_NONBLOCK on pipe");
  }
}

PipeBlazeServerStartup::~PipeBlazeServerStartup() {
  close(pipe_fd);
}

bool PipeBlazeServerStartup::IsStillAlive() {
  char c;
  return read(this->pipe_fd, &c, 1) == -1 && errno == EAGAIN;
}

void WriteSystemSpecificProcessIdentifier(const string& server_dir);

void ExecuteDaemon(const string& exe,
                   const std::vector<string>& args_vector,
                   const string& daemon_output, const string& server_dir,
                   BlazeServerStartup** server_startup) {
  int fds[2];
  if (pipe(fds)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "pipe creation failed");
  }
  CheckSingleThreaded();
  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(fds[1]);  // parent keeps only the reading side
    int unused_status;
    waitpid(child, &unused_status, 0);  // child double-forks
    *server_startup = new PipeBlazeServerStartup(fds[0]);
    return;
  } else {
    close(fds[0]);  // child keeps only the writing side
  }

  Daemonize(daemon_output);
  string pid_string = GetProcessIdAsString();
  string pid_file = blaze_util::JoinPath(server_dir, kServerPidFile);

  if (!blaze_util::WriteFile(pid_string, pid_file)) {
    // The exit code does not matter because we are already in the daemonized
    // server. The output of this operation will end up in jvm.out .
    pdie(0, "Cannot write PID file");
  }

  WriteSystemSpecificProcessIdentifier(server_dir);

  ExecuteProgram(exe, args_vector);
  pdie(0, "Cannot execute %s", exe.c_str());
}

static string RunProgram(const string& exe,
                         const std::vector<string>& args_vector) {
  int fds[2];
  if (pipe(fds)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "pipe creation failed");
  }
  int recv_socket = fds[0];
  int send_socket = fds[1];

  CheckSingleThreaded();
  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(send_socket);    // parent keeps only the reading side
    string result;
    bool success = blaze_util::ReadFrom(recv_socket, &result);
    close(recv_socket);
    if (!success) {
      pdie(blaze_exit_code::INTERNAL_ERROR, "Cannot read subprocess output");
    }
    return result;
  } else {                 // We're the child
    close(recv_socket);    // child keeps only the writing side
    // Redirect output to the writing side of the dup.
    dup2(send_socket, STDOUT_FILENO);
    dup2(send_socket, STDERR_FILENO);
    // Execute the binary
    ExecuteProgram(exe, args_vector);
    pdie(blaze_exit_code::INTERNAL_ERROR, "Failed to run %s", exe.c_str());
  }
  return string("");  //  We cannot reach here, just placate the compiler.
}

string GetJvmVersion(const string& java_exe) {
  vector<string> args;
  args.push_back("java");
  args.push_back("-version");

  string version_string = RunProgram(java_exe, args);
  return ReadJvmVersion(version_string);
}

bool CompareAbsolutePaths(const string& a, const string& b) {
  return a == b;
}

string GetHashedBaseDir(const string& root, const string& hashable) {
  unsigned char buf[blaze_util::Md5Digest::kDigestLength];
  blaze_util::Md5Digest digest;
  digest.Update(hashable.data(), hashable.size());
  digest.Finish(buf);
  return blaze_util::JoinPath(root, digest.String());
}

void CreateSecureOutputRoot(const string& path) {
  const char* root = path.c_str();
  struct stat fileinfo = {};

  if (!blaze_util::MakeDirectories(root, 0755)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "mkdir('%s')", root);
  }

  // The path already exists.
  // Check ownership and mode, and verify that it is a directory.

  if (lstat(root, &fileinfo) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "lstat('%s')", root);
  }

  if (fileinfo.st_uid != geteuid()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "'%s' is not owned by me",
        root);
  }

  if ((fileinfo.st_mode & 022) != 0) {
    int new_mode = fileinfo.st_mode & (~022);
    if (chmod(root, new_mode) < 0) {
      die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
          "'%s' has mode %o, chmod to %o failed", root,
          fileinfo.st_mode & 07777, new_mode);
    }
  }

  if (stat(root, &fileinfo) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "stat('%s')", root);
  }

  if (!S_ISDIR(fileinfo.st_mode)) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "'%s' is not a directory",
        root);
  }

  ExcludePathFromBackup(root);
}

string GetEnv(const string& name) {
  char* result = getenv(name.c_str());
  return result != NULL ? string(result) : "";
}

void SetEnv(const string& name, const string& value) {
  setenv(name.c_str(), value.c_str(), 1);
}

void UnsetEnv(const string& name) {
  unsetenv(name.c_str());
}

void SetupStdStreams() {
  // Set non-buffered output mode for stderr/stdout. The server already
  // line-buffers messages where it makes sense, so there's no need to do set
  // line-buffering here. On the other hand the server sometimes sends binary
  // output (when for example a query returns results as proto), in which case
  // we must not perform line buffering on the client side. So turn off
  // buffering here completely.
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  // Ensure we have three open fds.  Otherwise we can end up with
  // bizarre things like stdout going to the lock file, etc.
  if (fcntl(STDIN_FILENO, F_GETFL) == -1) open("/dev/null", O_RDONLY);
  if (fcntl(STDOUT_FILENO, F_GETFL) == -1) open("/dev/null", O_WRONLY);
  if (fcntl(STDERR_FILENO, F_GETFL) == -1) open("/dev/null", O_WRONLY);
}

// A signal-safe version of fprintf(stderr, ...).
//
// WARNING: any output from the blaze client may be interleaved
// with output from the blaze server.  In --curses mode,
// the Blaze server often erases the previous line of output.
// So, be sure to end each such message with TWO newlines,
// otherwise it may be erased by the next message from the
// Blaze server.
// Also, it's a good idea to start each message with a newline,
// in case the Blaze server has written a partial line.
void SigPrintf(const char *format, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, format);
  int r = vsnprintf(buf, sizeof buf, format, ap);
  va_end(ap);
  if (write(STDERR_FILENO, buf, r) <= 0) {
    // We don't care, just placate the compiler.
  }
}

uint64_t AcquireLock(const string& output_base, bool batch_mode, bool block,
                     BlazeLock* blaze_lock) {
  string lockfile = blaze_util::JoinPath(output_base, "lock");
  int lockfd = open(lockfile.c_str(), O_CREAT|O_RDWR, 0644);

  if (lockfd < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "cannot open lockfile '%s' for writing", lockfile.c_str());
  }

  // Keep server from inheriting a useless fd if we are not in batch mode
  if (!batch_mode) {
    if (fcntl(lockfd, F_SETFD, FD_CLOEXEC) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "fcntl(F_SETFD) failed for lockfile");
    }
  }

  struct flock lock;
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 0;
  // This doesn't really matter now, but allows us to subdivide the lock
  // later if that becomes meaningful.  (Ranges beyond EOF can be locked.)
  lock.l_len = 4096;

  uint64_t wait_time = 0;
  // Try to take the lock, without blocking.
  if (fcntl(lockfd, F_SETLK, &lock) == -1) {
    if (errno != EACCES && errno != EAGAIN) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "unexpected result from F_SETLK");
    }

    // We didn't get the lock.  Find out who has it.
    struct flock probe = lock;
    probe.l_pid = 0;
    if (fcntl(lockfd, F_GETLK, &probe) == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "unexpected result from F_GETLK");
    }
    if (!block) {
      die(blaze_exit_code::BAD_ARGV,
          "Another command is running (pid=%d). Exiting immediately.",
          probe.l_pid);
    }
    fprintf(stderr, "Another command is running (pid = %d).  "
            "Waiting for it to complete...", probe.l_pid);
    fflush(stderr);

    // Take a clock sample for that start of the waiting time
    uint64_t st = GetMillisecondsMonotonic();
    // Try to take the lock again (blocking).
    int r;
    do {
      r = fcntl(lockfd, F_SETLKW, &lock);
    } while (r == -1 && errno == EINTR);
    fprintf(stderr, "\n");
    if (r == -1) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
           "couldn't acquire file lock");
    }
    // Take another clock sample, calculate elapsed
    uint64_t et = GetMillisecondsMonotonic();
    wait_time = et - st;
  }

  // Identify ourselves in the lockfile.
  (void) ftruncate(lockfd, 0);
  const char *tty = ttyname(STDIN_FILENO);  // NOLINT (single-threaded)
  string msg = "owner=launcher\npid="
      + ToString(getpid()) + "\ntty=" + (tty ? tty : "") + "\n";
  // The contents are currently meant only for debugging.
  (void) write(lockfd, msg.data(), msg.size());
  blaze_lock->lockfd = lockfd;
  return wait_time;
}

void ReleaseLock(BlazeLock* blaze_lock) {
  close(blaze_lock->lockfd);
}

string GetUserName() {
  string user = GetEnv("USER");
  if (!user.empty()) {
    return user;
  }
  errno = 0;
  passwd *pwent = getpwuid(getuid());  // NOLINT (single-threaded)
  if (pwent == NULL || pwent->pw_name == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "$USER is not set, and unable to look up name of current user");
  }
  return pwent->pw_name;
}

bool IsEmacsTerminal() {
  string emacs = GetEnv("EMACS");
  string inside_emacs = GetEnv("INSIDE_EMACS");
  // GNU Emacs <25.1 (and ~all non-GNU emacsen) set EMACS=t, but >=25.1 doesn't
  // do that and instead sets INSIDE_EMACS=<stuff> (where <stuff> can look like
  // e.g. "25.1.1,comint").  So we check both variables for maximum
  // compatibility.
  return emacs == "t" || !inside_emacs.empty();
}

// Returns true iff both stdout and stderr are connected to a
// terminal, and it can support color and cursor movement
// (this is computed heuristically based on the values of
// environment variables).
bool IsStandardTerminal() {
  string term = GetEnv("TERM");
  if (term.empty() || term == "dumb" || term == "emacs" ||
      term == "xterm-mono" || term == "symbolics" || term == "9term" ||
      IsEmacsTerminal()) {
    return false;
  }
  return isatty(STDOUT_FILENO) && isatty(STDERR_FILENO);
}

// Returns the number of columns of the terminal to which stdout is
// connected, or $COLUMNS (default 80) if there is no such terminal.
int GetTerminalColumns() {
  struct winsize ws;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1) {
    return ws.ws_col;
  }
  string columns_env = GetEnv("COLUMNS");
  if (!columns_env.empty()) {
    char* endptr;
    int columns = blaze_util::strto32(columns_env.c_str(), &endptr, 10);
    if (*endptr == '\0') {  // $COLUMNS is a valid number
      return columns;
    }
  }
  return 80;  // default if not a terminal.
}

}   // namespace blaze.
