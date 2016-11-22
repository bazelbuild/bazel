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

#include "src/main/cpp/blaze_util.h"

#include <errno.h>
#include <fcntl.h>
#include <pwd.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sstream>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/util/port.h"

using blaze_util::die;
using blaze_util::pdie;

namespace blaze {

using std::string;
using std::vector;

const char kServerPidFile[] = "server.pid.txt";
const char kServerPidSymlink[] = "server.pid";

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

string MakeAbsolute(const string &path) {
  // Check if path is already absolute.
  if (path.empty() || path[0] == '/' || (isalpha(path[0]) && path[1] == ':')) {
    return path;
  }

  string cwd = blaze_util::GetCwd();

  // Determine whether the cwd ends with "/" or not.
  string separator = cwd.back() == '/' ? "" : "/";
  return cwd + separator + path;
}

// Replaces 'contents' with contents of 'fd' file descriptor.
// If `max_size` is positive, the method reads at most that many bytes; if it
// is 0, the method reads the whole file.
// Returns false on error.
bool ReadFileDescriptor(int fd, string *content, size_t max_size) {
  content->clear();
  char buf[4096];
  // OPT:  This loop generates one spurious read on regular files.
  while (int r = read(fd, buf, max_size > 0 ? std::min(max_size, sizeof buf)
                                            : sizeof buf)) {
    if (r == -1) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return false;
    }
    content->append(buf, r);
    if (max_size > 0) {
      if (max_size > static_cast<size_t>(r)) {
        max_size -= r;
      } else {
        break;
      }
    }
  }
  close(fd);
  return true;
}

// Replaces 'content' with contents of file 'filename'.
// If `max_size` is positive, the method reads at most that many bytes; if it
// is 0, the method reads the whole file.
// Returns false on error.
bool ReadFile(const string &filename, string *content, size_t max_size) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) return false;
  return ReadFileDescriptor(fd, content, max_size);
}

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const string &content, const string &filename) {
  return WriteFile(content.data(), content.size(), filename);
}

bool WriteFile(const void *data, size_t size, const std::string &filename) {
  UnlinkPath(filename);  // We don't care about the success of this.
  int fd = open(filename.c_str(), O_CREAT|O_WRONLY|O_TRUNC, 0755);  // chmod +x
  if (fd == -1) {
    return false;
  }
  int r = write(fd, data, size);
  if (r == -1) {
    return false;
  }
  int saved_errno = errno;
  if (close(fd)) {
    return false;  // Can fail on NFS.
  }
  errno = saved_errno;  // Caller should see errno from write().
  return static_cast<uint>(r) == size;
}

bool UnlinkPath(const string &file_path) {
  return unlink(file_path.c_str()) == 0;
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

const char* GetUnaryOption(const char *arg,
                           const char *next_arg,
                           const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == NULL) {
    return NULL;
  } else if (value[0] == '=') {
    return value + 1;
  } else if (value[0]) {
    return NULL;  // trailing garbage in key name
  }

  return next_arg;
}

bool GetNullaryOption(const char *arg, const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == NULL) {
    return false;
  } else if (value[0] == '=') {
    die(blaze_exit_code::BAD_ARGV,
        "In argument '%s': option '%s' does not take a value.", arg, key);
  } else if (value[0]) {
    return false;  // trailing garbage in key name
  }

  return true;
}

bool VerboseLogging() { return !GetEnv("VERBOSE_BLAZE_CLIENT").empty(); }

// Read the Jvm version from a file descriptor. The read fd
// should contains a similar output as the java -version output.
string ReadJvmVersion(const string& version_string) {
  // try to look out for 'version "'
  static const string version_pattern = "version \"";
  size_t found = version_string.find(version_pattern);
  if (found != string::npos) {
    found += version_pattern.size();
    // If we found "version \"", process until next '"'
    size_t end = version_string.find("\"", found);
    if (end == string::npos) {  // consider end of string as a '"'
      end = version_string.size();
    }
    return version_string.substr(found, end - found);
  }

  return "";
}

string GetJvmVersion(const string &java_exe) {
  vector<string> args;
  args.push_back("java");
  args.push_back("-version");

  string version_string = RunProgram(java_exe, args);
  return ReadJvmVersion(version_string);
}

bool CheckJavaVersionIsAtLeast(const string &jvm_version,
                               const string &version_spec) {
  vector<string> jvm_version_vect = blaze_util::Split(jvm_version, '.');
  int jvm_version_size = static_cast<int>(jvm_version_vect.size());
  vector<string> version_spec_vect = blaze_util::Split(version_spec, '.');
  int version_spec_size = static_cast<int>(version_spec_vect.size());
  int i;
  for (i = 0; i < jvm_version_size && i < version_spec_size; i++) {
    int jvm = blaze_util::strto32(jvm_version_vect[i].c_str(), NULL, 10);
    int spec = blaze_util::strto32(version_spec_vect[i].c_str(), NULL, 10);
    if (jvm > spec) {
      return true;
    } else if (jvm < spec) {
      return false;
    }
  }
  if (i < version_spec_size) {
    for (; i < version_spec_size; i++) {
      if (version_spec_vect[i] != "0") {
        return false;
      }
    }
  }
  return true;
}

uint64_t AcquireLock(const string& output_base, bool batch_mode, bool block,
                     BlazeLock* blaze_lock) {
  string lockfile = output_base + "/lock";
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

bool IsArg(const string& arg) {
  return blaze_util::starts_with(arg, "-") && (arg != "--help")
      && (arg != "-help") && (arg != "-h");
}

}  // namespace blaze
