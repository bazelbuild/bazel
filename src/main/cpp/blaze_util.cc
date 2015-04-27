// Copyright 2014 Google Inc. All rights reserved.
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
#include <sys/xattr.h>
#include <unistd.h>

#include <sstream>

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

// Returns the given path in absolute form.  Does not change paths that are
// already absolute.
//
// If called from working directory "/bar":
//   MakeAbsolute("foo") --> "/bar/foo"
//   MakeAbsolute("/foo") ---> "/foo"
string MakeAbsolute(string path) {
  // Check if path is already absolute.
  if (path.empty() || path[0] == '/') {
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

static int MakeDirectories_(string path, int mode, bool childmost) {
  if (path.empty() || path == "/") {
    errno = EACCES;
    return -1;
  }

  struct stat filestat = {};
  if (stat(path.c_str(), &filestat) == 0) {
    if (S_ISDIR(filestat.st_mode)) {
      // Only check permissions if this is the actual directory we're trying to
      // create.
      if (childmost) {
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
        if ((filestat.st_mode & 0777) != mode
            && chmod(path.c_str(), mode) == -1) {
          // errno set by chmod.
          return -1;
        }
      }
      return 0;
    } else {
      errno = ENOTDIR;
      return -1;
    }
  }

  if (errno == ENOENT) {
    // Path does not exist, attempt to create its parents, then it.
    string parent = blaze_util::Dirname(path);
    if (MakeDirectories_(parent, mode, false) == 0
        && mkdir(path.c_str(), mode) == 0) {
      return 0;
    }
  }

  // errno set by stat.
  return -1;
}

// mkdir -p path. Returns 0 if the path was created or already exists and could
// be chmod-ed to exactly the given permissions. If final part of the path is a
// symlink, this ensures that the destination of the symlink has the desired
// permissions. It also checks that the directory or symlink is owned by us.
// On failure, this returns -1 and sets errno.
int MakeDirectories(string path, int mode) {
  return MakeDirectories_(path, mode, true);
}

// Replaces 'contents' with contents of 'fd' file descriptor.
// Returns false on error.
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
// Returns false on error.
bool ReadFile(const string &filename, string *content) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) return false;
  return ReadFileDescriptor(fd, content);
}

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const string &content, const string &filename) {
  unlink(filename.c_str());
  int fd = open(filename.c_str(), O_CREAT|O_WRONLY|O_TRUNC, 0755);  // chmod +x
  if (fd == -1) return false;
  int r = write(fd, content.data(), content.size());
  int saved_errno = errno;
  if (close(fd)) return false;  // Can fail on NFS.
  errno = saved_errno;  // Caller should see errno from write().
  return r == content.size();
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

// Replace the current process with the given program in the given working
// directory, using the given argument vector.
// This function does not return on success.
void ExecuteProgram(string exe, const vector<string>& args_vector) {
  if (VerboseLogging()) {
    string dbg;
    for (const auto& s : args_vector) {
      dbg.append(s);
      dbg.append(" ");
    }

    char cwd[PATH_MAX] = {};
    getcwd(cwd, sizeof(cwd));

    fprintf(stderr, "Invoking binary %s in %s:\n  %s\n",
            exe.c_str(), cwd, dbg.c_str());
  }

  // Copy to a char* array for execv:
  int n = args_vector.size();
  const char **argv = new const char *[n + 1];
  for (int i = 0; i < n; ++i) {
    argv[i] = args_vector[i].c_str();
  }
  argv[n] = NULL;

  execv(exe.c_str(), const_cast<char**>(argv));
}

// Re-execute the blaze command line with a different binary as argv[0].
// This function does not return on success.
void ReExecute(const string &executable, int argc, const char *argv[]) {
  vector<string> args;
  args.push_back(executable);
  for (int i = 1; i < argc; i++) {
    args.push_back(argv[i]);
  }
  ExecuteProgram(args[0], args);
}

const char* GetUnaryOption(const char *arg, const char *next_arg,
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

bool CheckValidPort(const string &str, const string &option, string *error) {
  int number;
  if (blaze_util::safe_strto32(str, &number) && number > 0 && number < 65536) {
    return true;
  }

  blaze_util::StringPrintf(error,
      "Invalid argument to %s: '%s' (must be a valid port number).",
      option.c_str(), str.c_str());
  return false;
}

bool VerboseLogging() {
  return getenv("VERBOSE_BLAZE_CLIENT") != NULL;
}

// Read the Jvm version from a file descriptor. The read fd
// should contains a similar output as the java -version output.
string ReadJvmVersion(int fd) {
  string version_string;
  if (ReadFileDescriptor(fd, &version_string)) {
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
  }
  return "";
}

string GetJvmVersion(string java_exe) {
  vector<string> args;
  args.push_back("java");
  args.push_back("-version");

  int fds[2];
  if (pipe(fds)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "pipe creation failed");
  }

  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(fds[1]);         // parent keeps only the reading side
    return ReadJvmVersion(fds[0]);
  } else {
    close(fds[0]);  // child keeps only the writing side
    // Redirect output to the writing side of the dup.
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    // Execute java -version
    ExecuteProgram(java_exe, args);
    pdie(blaze_exit_code::INTERNAL_ERROR, "Failed to run java -version");
  }
}

bool CheckJavaVersionIsAtLeast(string jvm_version, string version_spec) {
  vector<string> jvm_version_vect = blaze_util::Split(jvm_version, '.');
  vector<string> version_spec_vect = blaze_util::Split(version_spec, '.');
  int i;
  for (i = 0; i < jvm_version_vect.size() && i < version_spec_vect.size();
       i++) {
    int jvm = blaze_util::strto32(jvm_version_vect[i].c_str(), NULL, 10);
    int spec = blaze_util::strto32(version_spec_vect[i].c_str(), NULL, 10);
    if (jvm > spec) {
      return true;
    } else if (jvm < spec) {
      return false;
    }
  }
  if (i < version_spec_vect.size()) {
    for (; i < version_spec_vect.size(); i++) {
      if (version_spec_vect[i] != "0") {
        return false;
      }
    }
  }
  return true;
}

}  // namespace blaze
