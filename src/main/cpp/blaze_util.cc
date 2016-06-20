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
#include <limits.h>
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
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/util/port.h"

using blaze_util::die;
using blaze_util::pdie;
using std::vector;

namespace blaze {

string ServerPidFile() {
  return "server.pid.txt";
}

string ServerPidSymlink() {
  return "server.pid";
}

string GetUserName() {
  const char *user = getenv("USER");
  if (user && user[0] != '\0') return user;
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

  char cwdbuf[PATH_MAX];
  if (getcwd(cwdbuf, sizeof cwdbuf) == NULL) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "getcwd() failed");
  }

  // Determine whether the cwd ends with "/" or not.
  string separator = (cwdbuf[strlen(cwdbuf) - 1] == '/') ? "" : "/";
  return cwdbuf + separator + path;
}

// Runs "stat" on `path`. Returns -1 and sets errno if stat fails or
// `path` isn't a directory. If check_perms is true, this will also
// make sure that `path` is owned by the current user and has `mode`
// permissions (observing the umask). It attempts to run chmod to
// correct the mode if necessary. If `path` is a symlink, this will
// check ownership of the link, not the underlying directory.
static int GetDirectoryStat(const string& path, mode_t mode, bool check_perms) {
  struct stat filestat = {};
  if (stat(path.c_str(), &filestat) == -1) {
    return -1;
  }

  if (!S_ISDIR(filestat.st_mode)) {
    errno = ENOTDIR;
    return -1;
  }

  if (check_perms) {
    // If this is a symlink, run checks on the link. (If we did lstat above
    // then it would return false for ISDIR).
    struct stat linkstat = {};
    if (lstat(path.c_str(), &linkstat) != 0) {
      return -1;
    }
    if (linkstat.st_uid != geteuid()) {
      // The directory isn't owned by me.
      errno = EACCES;
      return -1;
    }

    mode_t mask = umask(022);
    umask(mask);
    mode = (mode & ~mask);
    if ((filestat.st_mode & 0777) != mode
        && chmod(path.c_str(), mode) == -1) {
      // errno set by chmod.
      return -1;
    }
  }
  return 0;
}

static int MakeDirectories(const string& path, mode_t mode, bool childmost) {
  if (path.empty() || path == "/") {
    errno = EACCES;
    return -1;
  }

  int retval = GetDirectoryStat(path, mode, childmost);
  if (retval == 0) {
    return 0;
  }

  if (errno == ENOENT) {
    // Path does not exist, attempt to create its parents, then it.
    string parent = blaze_util::Dirname(path);
    if (MakeDirectories(parent, mode, false) == -1) {
      // errno set by stat.
      return -1;
    }

    if (mkdir(path.c_str(), mode) == -1) {
      if (errno == EEXIST) {
        if (childmost) {
          // If there are multiple bazel calls at the same time then the
          // directory could be created between the MakeDirectories and mkdir
          // calls. This is okay, but we still have to check the permissions.
          return GetDirectoryStat(path, mode, childmost);
        } else {
          // If this isn't the childmost directory, we don't care what the
          // permissions were. If it's not even a directory then that error will
          // get caught when we attempt to create the next directory down the
          // chain.
          return 0;
        }
      }
      // errno set by mkdir.
      return -1;
    }
    return 0;
  }

  return retval;
}

// mkdir -p path. Returns 0 if the path was created or already exists and could
// be chmod-ed to exactly the given permissions. If final part of the path is a
// symlink, this ensures that the destination of the symlink has the desired
// permissions. It also checks that the directory or symlink is owned by us.
// On failure, this returns -1 and sets errno.
int MakeDirectories(const string& path, mode_t mode) {
  return MakeDirectories(path, mode, true);
}

// Replaces 'contents' with contents of 'fd' file descriptor.
// Returns false on error. Can be called from a signal handler.
bool ReadFileDescriptor(int fd, string *content) {
  content->clear();
  char buf[4096];
  // OPT:  This loop generates one spurious read on regular files.
  while (int r = read(fd, buf, sizeof buf)) {
    if (r == -1) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return false;
    }
    content->append(buf, r);
  }
  close(fd);
  return true;
}

// Replaces 'content' with contents of file 'filename'.
// Returns false on error. Can be called from a signal handler.
bool ReadFile(const string &filename, string *content) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) return false;
  return ReadFileDescriptor(fd, content);
}

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const string &content, const string &filename) {
  UnlinkPath(filename);  // We don't care about the success of this.
  int fd = open(filename.c_str(), O_CREAT|O_WRONLY|O_TRUNC, 0755);  // chmod +x
  if (fd == -1) {
    return false;
  }
  int r = write(fd, content.data(), content.size());
  if (r == -1) {
    return false;
  }
  int saved_errno = errno;
  if (close(fd)) {
    return false;  // Can fail on NFS.
  }
  errno = saved_errno;  // Caller should see errno from write().
  return static_cast<uint>(r) == content.size();
}

bool UnlinkPath(const string &file_path) {
  return unlink(file_path.c_str()) == 0;
}

// Returns true iff both stdout and stderr are connected to a
// terminal, and it can support color and cursor movement
// (this is computed heuristically based on the values of
// environment variables).
bool IsStandardTerminal() {
  string term = getenv("TERM") == nullptr ? "" : getenv("TERM");
  string emacs = getenv("EMACS") == nullptr ? "" : getenv("EMACS");
  if (term == "" || term == "dumb" || term == "emacs" || term == "xterm-mono" ||
      term == "symbolics" || term == "9term" || emacs == "t") {
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
  const char* columns_env = getenv("COLUMNS");
  if (columns_env != NULL && columns_env[0] != '\0') {
    char* endptr;
    int columns = blaze_util::strto32(columns_env, &endptr, 10);
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

bool VerboseLogging() {
  return getenv("VERBOSE_BLAZE_CLIENT") != NULL;
}

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
    uint64_t st = MonotonicClock();
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
    uint64_t et = MonotonicClock();
    wait_time = (et - st) / 1000000LL;
  }

  // Identify ourselves in the lockfile.
  ftruncate(lockfd, 0);
  const char *tty = ttyname(STDIN_FILENO);  // NOLINT (single-threaded)
  string msg = "owner=launcher\npid="
      + ToString(getpid()) + "\ntty=" + (tty ? tty : "") + "\n";
  // Don't bother checking for error, since it's unlikely and unimportant.
  // The contents are currently meant only for debugging.
  write(lockfd, msg.data(), msg.size());
  blaze_lock->lockfd = lockfd;
  return wait_time;
}

void ReleaseLock(BlazeLock* blaze_lock) {
  close(blaze_lock->lockfd);
}

}  // namespace blaze
