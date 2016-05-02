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

#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;

using std::string;
using std::vector;

void ExecuteProgram(const string &exe, const vector<string> &args_vector) {
  if (VerboseLogging()) {
    string dbg;
    for (const auto &s : args_vector) {
      dbg.append(s);
      dbg.append(" ");
    }

    char cwd[PATH_MAX] = {};
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
      pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "getcwd() failed");
    }

    fprintf(stderr, "Invoking binary %s in %s:\n  %s\n", exe.c_str(), cwd,
            dbg.c_str());
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

std::string ListSeparator() { return ":"; }

bool SymlinkDirectories(const string &target, const string &link) {
  return symlink(target.c_str(), link.c_str()) == 0;
}

// Causes the current process to become a daemon (i.e. a child of
// init, detached from the terminal, in its own session.)  We don't
// change cwd, though.
static void Daemonize(const string& daemon_output) {
  // Don't call die() or exit() in this function; we're already in a
  // child process so it won't work as expected.  Just don't do
  // anything that can possibly fail. :)

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
  if (open(daemon_output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666) == -1) {
    // In a daemon, no-one can hear you scream.
    open("/dev/null", O_WRONLY);
  }
  dup(STDOUT_FILENO);  // stderr (2>&1)
}

class PipeBlazeServerStartup : public BlazeServerStartup {
 public:
  PipeBlazeServerStartup(int pipe_fd);
  virtual ~PipeBlazeServerStartup();
  bool IsStillAlive() override;

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
  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(fds[1]);  // parent keeps only the reading side
    *server_startup = new PipeBlazeServerStartup(fds[0]);
    return;
  } else {
    close(fds[0]);  // child keeps only the writing side
  }

  Daemonize(daemon_output);
  string pid_string = ToString(getpid());
  string pid_file = blaze_util::JoinPath(server_dir, ServerPidFile());
  string pid_symlink_file =
      blaze_util::JoinPath(server_dir, ServerPidSymlink());

  if (!WriteFile(pid_string, pid_file)) {
    // The exit code does not matter because we are already in the daemonized
    // server. The output of this operation will end up in jvm.out .
    pdie(0, "Cannot write PID file");
  }

  UnlinkPath(pid_symlink_file.c_str());
  if (symlink(pid_string.c_str(), pid_symlink_file.c_str()) < 0) {
    pdie(0, "Cannot write PID symlink");
  }

  WriteSystemSpecificProcessIdentifier(server_dir);

  ExecuteProgram(exe, args_vector);
  pdie(0, "Cannot execute %s", exe.c_str());
}

string RunProgram(const string& exe, const std::vector<string>& args_vector) {
  int fds[2];
  if (pipe(fds)) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "pipe creation failed");
  }

  int child = fork();
  if (child == -1) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "fork() failed");
  } else if (child > 0) {  // we're the parent
    close(fds[1]);         // parent keeps only the reading side
    string result;
    if (!ReadFileDescriptor(fds[0], &result)) {
      pdie(blaze_exit_code::INTERNAL_ERROR, "Cannot read subprocess output");
    }

    return result;
  } else {          // We're the child
    close(fds[0]);  // child keeps only the writing side
    // Redirect output to the writing side of the dup.
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    // Execute the binary
    ExecuteProgram(exe, args_vector);
    pdie(blaze_exit_code::INTERNAL_ERROR, "Failed to run %s", exe.c_str());
  }
}

bool ReadDirectorySymlink(const string &name, string* result) {
  char buf[PATH_MAX + 1];
  int len = readlink(name.c_str(), buf, PATH_MAX);
  if (len < 0) {
    return false;
  }

  buf[len] = 0;
  *result = buf;
  return true;
}

bool CompareAbsolutePaths(const string& a, const string& b) {
  return a == b;
}

}   // namespace blaze.
