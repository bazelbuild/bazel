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

#define _WITH_DPRINTF
#include "src/main/cpp/blaze_util_platform.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>  // PATH_MAX
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cassert>
#include <cinttypes>
#include <set>
#include <string>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/global_variables.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/md5.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"

namespace blaze {

using blaze_exit_code::INTERNAL_ERROR;
using blaze_util::GetLastErrorString;

using std::set;
using std::string;
using std::vector;

namespace embedded_binaries {

class PosixDumper : public Dumper {
 public:
  static PosixDumper* Create(string* error);
  ~PosixDumper() { Finish(nullptr); }
  void Dump(const void* data, const size_t size, const string& path) override;
  bool Finish(string* error) override;

 private:
  PosixDumper() : was_error_(false) {}

  set<string> dir_cache_;
  string error_msg_;
  bool was_error_;
};

Dumper* Create(string* error) { return PosixDumper::Create(error); }

PosixDumper* PosixDumper::Create(string* error) { return new PosixDumper(); }

void PosixDumper::Dump(const void* data, const size_t size,
                       const string& path) {
  if (was_error_) {
    return;
  }

  string dirname = blaze_util::Dirname(path);
  // Performance optimization: memoize the paths we already created a
  // directory for, to spare a stat in attempting to recreate an already
  // existing directory.
  if (dir_cache_.insert(dirname).second) {
    if (!blaze_util::MakeDirectories(dirname, 0777)) {
      was_error_ = true;
      string msg = GetLastErrorString();
      error_msg_ = string("couldn't create '") + path + "': " + msg;
    }
  }

  if (was_error_) {
    return;
  }

  if (!blaze_util::WriteFile(data, size, path, 0755)) {
    was_error_ = true;
    string msg = GetLastErrorString();
    error_msg_ = string("Failed to write zipped file '") + path + "': " + msg;
  }
}

bool PosixDumper::Finish(string* error) {
  if (was_error_ && error) {
    *error = error_msg_;
  }
  return !was_error_;
}

}  // namespace embedded_binaries

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
          KillServerProcess(
              SignalHandler::Get().GetGlobals()->server_pid,
              SignalHandler::Get().GetGlobals()->options->output_base);
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

string GetJavaBinaryUnderJavabase() { return "bin/java"; }

// NB: execve() requires pointers to non-const char arrays but .c_str() returns
// a pointer to const char arrays. We could do the const_cast in this function,
// but it's better to violate const correctness as late as possible. No one
// cares about what happens just before execve() because we'll soon become a new
// binary anyway.
const char** ConvertStringVectorToArgv(const vector<string>& args) {
  const char** argv = new const char*[args.size() + 1];
  for (size_t i = 0; i < args.size(); i++) {
    argv[i] = args[i].c_str();
  }

  argv[args.size()] = NULL;

  return argv;
}

void ExecuteProgram(const string& exe, const vector<string>& args_vector) {
  BAZEL_LOG(INFO) << "Invoking binary " << exe << " in "
                  << blaze_util::GetCwd();

  const char** argv = ConvertStringVectorToArgv(args_vector);
  execv(exe.c_str(), const_cast<char**>(argv));
}

const char kListSeparator = ':';

bool SymlinkDirectories(const string &target, const string &link) {
  return symlink(target.c_str(), link.c_str()) == 0;
}

// Causes the current process to become a daemon (i.e. a child of
// init, detached from the terminal, in its own session.)  We don't
// change cwd, though.
static void Daemonize(const char* daemon_output,
                      const bool daemon_output_append) {
  // Don't call BAZEL_DIE or exit() in this function; we're already in a child
  // process so it won't work as expected.  Just don't do anything that can
  // possibly fail. :)

  signal(SIGHUP, SIG_IGN);
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
  int out_flags =
      O_WRONLY | O_CREAT | (daemon_output_append ? O_APPEND : O_TRUNC);
  if (open(daemon_output, out_flags, 0666) == -1) {
    // In a daemon, no-one can hear you scream.
    open("/dev/null", O_WRONLY);
  }
  (void) dup(STDOUT_FILENO);  // stderr (2>&1)
}

// Notifies the client about the death of the server process by keeping a socket
// open in the server. If the server dies for any reason, the socket will be
// closed, which can be detected by the client.
class SocketBlazeServerStartup : public BlazeServerStartup {
 public:
  SocketBlazeServerStartup(int pipe_fd);
  virtual ~SocketBlazeServerStartup();
  virtual bool IsStillAlive();

 private:
  int fd;
};

SocketBlazeServerStartup::SocketBlazeServerStartup(int fd)
    : fd(fd) {
}

SocketBlazeServerStartup::~SocketBlazeServerStartup() {
  close(fd);
}

bool SocketBlazeServerStartup::IsStillAlive() {
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  int result;
  do {
    result = poll(&pfd, 1, 0);
  } while (result < 0 && errno == EINTR);
  if (result == 0) {
    // Timeout, server is still alive
    return true;
  } else {
    // Whether it's an error or pfd.revents & POLLHUP > 0, we assume child is
    // dead.
    return false;
  }
}

// NB: There should only be system calls in this function. See the comment
// before ExecuteDaemon() to understand why. strerror() and strlen() are
// hopefully okay.
static void DieAfterFork(const char* message) {
  char* error_string = strerror(errno);  // strerror is hopefully okay
  write(STDERR_FILENO, message, strlen(message));  // strlen should be OK
  write(STDERR_FILENO, ": ", 2);
  write(STDERR_FILENO, error_string, strlen(error_string));
  write(STDERR_FILENO, "\n", 1);
  _exit(blaze_exit_code::INTERNAL_ERROR);
}

// NB: There should only be system calls in this function. See the comment
// before ExecuteDaemon() to understand why.
static void ReadFromFdWithRetryEintr(
    int fd, void *buf, size_t count, const char* error_message) {
  ssize_t result;
  do {
    result = read(fd, buf, count);
  } while (result < 0 && errno == EINTR);
  if (result < 0 || static_cast<size_t>(result) != count) {
    DieAfterFork(error_message);
  }
}

// NB: There should only be system calls in this function. See the comment
// before ExecuteDaemon() to understand why.
static void WriteToFdWithRetryEintr(
    int fd, void *buf, size_t count, const char* error_message) {
  ssize_t result;
  do {
    // Ideally, we'd use send(..., MSG_NOSIGNAL), but that's not available on
    // Darwin.
    result = write(fd, buf, count);
  } while (result < 0 && errno == EINTR);
  if (result < 0 || static_cast<size_t>(result) != count) {
    DieAfterFork(error_message);
  }
}

void WriteSystemSpecificProcessIdentifier(
    const string& server_dir, pid_t server_pid);

// We do a lot of seemingly-needless complications to avoid doing anything
// complex after a fork(). The reason is that forking in multi-threaded
// programs is fraught with peril.
//
// One root cause is that fork() only forks the thread it was called from. If
// another thread holds a lock, it will never be relinquished in the child
// process, and malloc() sometimes does lock. Thus, we need to avoid allocating
// any dynamic memory in the child process, which is hard if any C++ feature is
// used.
//
// We also don't know what libc does behind the scenes, so it's advisable to
// avoid anything that's not a system call. read(), write(), fork() and execv()
// aren't guaranteed to be pure system calls, either, but we can't get any
// closer to this ideal without writing logic specific to each POSIX system we
// run on.
//
// Another way to tackle this issue would be to pre-fork a child process before
// spawning any threads which is then used to fork all the other necessary
// child processes. However, then we'd need to invent a protocol that can handle
// sending stdout/stderr back and forking multiple times, which isn't trivial,
// either.
//
// Yet another fix would be not to use multiple threads before forking. However,
// we need to use gRPC to figure out if a server process is already present and
// gRPC currently cannot be convinced not to spawn any threads.
//
// Another way would be to use posix_spawn(). However, since we need to create a
// daemon process, we'd need to posix_spawn() a little child process that
// daemonizes then exec()s the actual JVM, which is also non-trivial. So I hope
// this will be good enough because for all its flaws, this solution is at least
// localized here.
int ExecuteDaemon(const string& exe,
                  const std::vector<string>& args_vector,
                  const std::map<string, EnvVarValue>& env,
                  const string& daemon_output,
                  const bool daemon_output_append,
                  const string& server_dir,
                  BlazeServerStartup** server_startup) {
  int fds[2];

  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "socket creation failed: " << GetLastErrorString();
  }

  const char* daemon_output_chars = daemon_output.c_str();
  const char** argv = ConvertStringVectorToArgv(args_vector);
  const char* exe_chars = exe.c_str();

  int child = fork();
  if (child == -1) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "fork() failed: " << GetLastErrorString();
    return -1;
  } else if (child > 0) {
    // Parent process (i.e. the client)
    close(fds[1]);  // parent keeps one side...
    int unused_status;
    waitpid(child, &unused_status, 0);  // child double-forks
    pid_t server_pid = 0;
    ReadFromFdWithRetryEintr(fds[0], &server_pid, sizeof server_pid,
                        "cannot read server PID from server");
    string pid_file = blaze_util::JoinPath(server_dir, kServerPidFile);
    if (!blaze_util::WriteFile(ToString(server_pid), pid_file)) {
      BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
          << "cannot write PID file: " << GetLastErrorString();
      return -1;
    }

    WriteSystemSpecificProcessIdentifier(server_dir, server_pid);
    char dummy = 'a';
    WriteToFdWithRetryEintr(fds[0], &dummy, 1,
                       "cannot notify server about having written PID file");
    *server_startup = new SocketBlazeServerStartup(fds[0]);
    return server_pid;
  } else {
    // Child process (i.e. the server)
    // NB: There should only be system calls in this branch. See the comment
    // before ExecuteDaemon() to understand why.
    close(fds[0]);  // ...child keeps the other.

    {
      WithEnvVars env_obj(env);
      Daemonize(daemon_output_chars, daemon_output_append);
      pid_t server_pid = getpid();
      WriteToFdWithRetryEintr(fds[1], &server_pid, sizeof server_pid,
                              "cannot communicate server PID to client");
      // We wait until the client writes the PID file so that there is no race
      // condition; the server expects the PID file to already be there so that
      // it can read it and know its own PID (see the ctor GrpcServerImpl) and
      // so that it can kill itself if the PID file is deleted (see
      // GrpcServerImpl.PidFileWatcherThread)
      char dummy;
      ReadFromFdWithRetryEintr(
          fds[1], &dummy, 1,
          "cannot get PID file write acknowledgement from client");

      execv(exe_chars, const_cast<char**>(argv));
      DieAfterFork("Cannot execute daemon");
      return -1;
    }
  }
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
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "mkdir('" << root << "'): " << GetLastErrorString();
  }

  // The path already exists.
  // Check ownership and mode, and verify that it is a directory.

  if (lstat(root, &fileinfo) < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "lstat('" << root << "'): " << GetLastErrorString();
  }

  if (fileinfo.st_uid != geteuid()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "'" << root << "' is not owned by me";
  }

  if ((fileinfo.st_mode & 022) != 0) {
    int new_mode = fileinfo.st_mode & (~022);
    if (chmod(root, new_mode) < 0) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "'" << root << "' has mode " << (fileinfo.st_mode & 07777)
          << ", chmod to " << new_mode << " failed";
    }
  }

  if (stat(root, &fileinfo) < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "stat('" << root << "'): " << GetLastErrorString();
  }

  if (!S_ISDIR(fileinfo.st_mode)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "'" << root << "' is not a directory";
  }

  ExcludePathFromBackup(root);
}

string GetEnv(const string& name) {
  char* result = getenv(name.c_str());
  return result != NULL ? string(result) : "";
}

bool ExistsEnv(const string& name) {
  return getenv(name.c_str()) != NULL;
}

void SetEnv(const string& name, const string& value) {
  setenv(name.c_str(), value.c_str(), 1);
}

void UnsetEnv(const string& name) {
  unsetenv(name.c_str());
}

bool WarnIfStartedFromDesktop() { return false; }

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

static int setlk(int fd, struct flock *lock) {
#ifdef __linux__
// If we're building with glibc <2.20, or another libc which predates
// OFD locks, define the constant ourselves.  This assumes that the libc
// and kernel definitions for struct flock are identical.
#ifndef F_OFD_SETLK
#define F_OFD_SETLK 37
#endif
#endif
#ifdef F_OFD_SETLK
  // Prefer OFD locks if available.  POSIX locks can be lost "accidentally"
  // due to any close() on the lock file, and are not reliably preserved
  // across execve() on Linux, which we need for --batch mode.
  if (fcntl(fd, F_OFD_SETLK, lock) == 0) return 0;
  if (errno != EINVAL) {
    if (errno != EACCES && errno != EAGAIN) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "unexpected result from F_OFD_SETLK: " << GetLastErrorString();
    }
    return -1;
  }
  // F_OFD_SETLK was added in Linux 3.15.  Older kernels return EINVAL.
  // Fall back to F_SETLK in that case.
#endif
  if (fcntl(fd, F_SETLK, lock) == 0) return 0;
  if (errno != EACCES && errno != EAGAIN) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "unexpected result from F_SETLK: " << GetLastErrorString();
  }
  return -1;
}

uint64_t AcquireLock(const string& output_base, bool batch_mode, bool block,
                     BlazeLock* blaze_lock) {
  string lockfile = blaze_util::JoinPath(output_base, "lock");
  int lockfd = open(lockfile.c_str(), O_CREAT|O_RDWR, 0644);

  if (lockfd < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "cannot open lockfile '" << lockfile
        << "' for writing: " << GetLastErrorString();
  }

  // Keep server from inheriting a useless fd if we are not in batch mode
  if (!batch_mode) {
    if (fcntl(lockfd, F_SETFD, FD_CLOEXEC) == -1) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "fcntl(F_SETFD) failed for lockfile: " << GetLastErrorString();
    }
  }

  struct flock lock = {};
  lock.l_type = F_WRLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 0;
  // This doesn't really matter now, but allows us to subdivide the lock
  // later if that becomes meaningful.  (Ranges beyond EOF can be locked.)
  lock.l_len = 4096;

  // Take the exclusive server lock.  If we fail, we busy-wait until the lock
  // becomes available.
  //
  // We used to rely on fcntl(F_SETLKW) to lazy-wait for the lock to become
  // available, which is theoretically fine, but doing so prevents us from
  // determining if the PID of the server holding the lock has changed under the
  // hood.  There have been multiple bug reports where users (especially macOS
  // ones) mention that the Blaze invocation hangs on a non-existent PID.  This
  // should help troubleshoot those scenarios in case there really is a bug
  // somewhere.
  bool multiple_attempts = false;
  string owner;
  const uint64_t start_time = GetMillisecondsMonotonic();
  while (setlk(lockfd, &lock) == -1) {
    string buffer(4096, 0);
    ssize_t r = pread(lockfd, &buffer[0], buffer.size(), 0);
    if (r < 0) {
      BAZEL_LOG(WARNING) << "pread() lock file: " << strerror(errno);
      r = 0;
    }
    buffer.resize(r);
    if (owner != buffer) {
      // Each time we learn a new lock owner, print it out.
      owner = buffer;
      BAZEL_LOG(USER) << "Another command holds the client lock: \n" << owner;
      if (block) {
        BAZEL_LOG(USER) << "Waiting for it to complete...";
        fflush(stderr);
      }
    }

    if (!block) {
      BAZEL_DIE(blaze_exit_code::LOCK_HELD_NOBLOCK_FOR_LOCK)
          << "Exiting because the lock is held and --noblock_for_lock was "
             "given.";
    }

    TrySleep(500);
    multiple_attempts = true;
  }
  const uint64_t end_time = GetMillisecondsMonotonic();

  // If we took the lock on the first try, force the reported wait time to 0 to
  // avoid unnecessary noise in the logs.  In this metric, we are only
  // interested in knowing how long it took for other commands to complete, not
  // how fast acquiring a lock is.
  const uint64_t wait_time = !multiple_attempts ? 0 : end_time - start_time;

  // Identify ourselves in the lockfile.
  // The contents are printed for human consumption when another client
  // fails to take the lock, but not parsed otherwise.
  (void) ftruncate(lockfd, 0);
  lseek(lockfd, 0, SEEK_SET);
  // Arguably we should ensure this fits in the 4KB we lock.  In practice no one
  // will have a cwd long enough to overflow that, and nothing currently uses
  // the rest of the lock file anyway.
  dprintf(lockfd, "pid=%d\nowner=client\n", getpid());
  string cwd = blaze_util::GetCwd();
  dprintf(lockfd, "cwd=%s\n", cwd.c_str());
  if (const char *tty = ttyname(STDIN_FILENO)) {  // NOLINT (single-threaded)
    dprintf(lockfd, "tty=%s\n", tty);
  }
  blaze_lock->lockfd = lockfd;
  return wait_time;
}

void ReleaseLock(BlazeLock* blaze_lock) {
  close(blaze_lock->lockfd);
}

bool KillServerProcess(int pid, const string& output_base) {
  // Kill the process and make sure it's dead before proceeding.
  killpg(pid, SIGKILL);
  if (!AwaitServerProcessTermination(pid, output_base,
                                     kPostKillGracePeriodSeconds)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Attempted to kill stale server process (pid=" << pid
        << ") using SIGKILL, but it did not die in a timely fashion.";
  }
  return true;
}

void TrySleep(unsigned int milliseconds) {
  time_t seconds_part = (time_t)(milliseconds / 1000);
  long nanoseconds_part = ((long)(milliseconds % 1000)) * 1000 * 1000;
  struct timespec sleeptime = {seconds_part, nanoseconds_part};
  nanosleep(&sleeptime, NULL);
}

string GetUserName() {
  string user = GetEnv("USER");
  if (!user.empty()) {
    return user;
  }
  errno = 0;
  passwd *pwent = getpwuid(getuid());  // NOLINT (single-threaded)
  if (pwent == NULL || pwent->pw_name == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "$USER is not set, and unable to look up name of current user: "
        << GetLastErrorString();
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

// Returns true if stderr is connected to a terminal, and it can support color
// and cursor movement (this is computed heuristically based on the values of
// environment variables).  The only file handle into which Blaze outputs
// control characters is stderr, so we only care for the stderr descriptor type.
bool IsStderrStandardTerminal() {
  string term = GetEnv("TERM");
  if (term.empty() || term == "dumb" || term == "emacs" ||
      term == "xterm-mono" || term == "symbolics" || term == "9term" ||
      IsEmacsTerminal()) {
    return false;
  }
  return isatty(STDERR_FILENO);
}

// Returns the number of columns of the terminal to which stderr is connected,
// or $COLUMNS (default 80) if there is no such terminal.  The only file handle
// into which Blaze outputs formatted messages is stderr, so we only care for
// width of a terminal connected to the stderr descriptor.
int GetStderrTerminalColumns() {
  struct winsize ws;
  if (ioctl(STDERR_FILENO, TIOCGWINSZ, &ws) != -1) {
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

// Raises a resource limit to the maximum allowed value.
//
// This function raises the limit of the resource given in "resource" from its
// soft limit to its hard limit. If the hard limit is unlimited and
// allow_infinity is false, uses the kernel-level limit because setting the
// soft limit to unlimited may not work.
//
// Note that this is a best-effort operation. Any failure during this process
// will result in a warning but execution will continue.
static bool UnlimitResource(const int resource, const bool allow_infinity) {
  struct rlimit rl;
  if (getrlimit(resource, &rl) == -1) {
    BAZEL_LOG(WARNING) << "failed to get resource limit " << resource << ": "
                       << strerror(errno);
    return false;
  }

  if (rl.rlim_cur == rl.rlim_max) {
    // Nothing to do. Return early to prevent triggering any warnings caused by
    // the code below. This way, we will only show warnings the first time the
    // Blaze server is started and not on each command invocation.
    return true;
  }

  rl.rlim_cur = rl.rlim_max;
  if (rl.rlim_cur == RLIM_INFINITY && !allow_infinity) {
    const rlim_t explicit_limit = GetExplicitSystemLimit(resource);
    if (explicit_limit <= 0) {
      // If not implemented (-1) or on an error (0), do nothing and try to
      // increase the soft limit to the hard one. This might fail, but it's good
      // to try anyway.
      assert(rl.rlim_cur == rl.rlim_max);
    } else {
      rl.rlim_cur = explicit_limit;
    }
  }

  if (setrlimit(resource, &rl) == -1) {
    BAZEL_LOG(WARNING) << "failed to raise resource limit " << resource
                       << " to " << static_cast<intmax_t>(rl.rlim_cur) << ": "
                       << strerror(errno);
    return false;
  }

  return true;
}

bool UnlimitResources() {
  bool success = true;
  success &= UnlimitResource(RLIMIT_NOFILE, false);
  success &= UnlimitResource(RLIMIT_NPROC, false);
  return success;
}

bool UnlimitCoredumps() {
  return UnlimitResource(RLIMIT_CORE, true);
}

void EnsurePythonPathOption(vector<string>* options) {
  // do nothing.
}

}   // namespace blaze.
