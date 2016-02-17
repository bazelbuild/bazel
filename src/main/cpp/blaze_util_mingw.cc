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

#include <errno.h>  // errno, ENAMETOOLONG
#include <limits.h>
#include <string.h>  // strerror
#include <sys/cygwin.h>
#include <sys/socket.h>
#include <sys/statfs.h>
#include <unistd.h>

#include <windows.h>

#include <cstdlib>
#include <cstdio>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using blaze_util::die;
using blaze_util::pdie;
using std::string;
using std::vector;

void WarnFilesystemType(const string& output_base) {
}

string GetSelfPath() {
  char buffer[PATH_MAX] = {};
  if (!GetModuleFileName(0, buffer, sizeof(buffer))) {
    pdie(255, "Error %u getting executable file name\n", GetLastError());
  }
  return string(buffer);
}

string GetOutputRoot() {
  return "/var/tmp";
}

pid_t GetPeerProcessId(int socket) {
  struct ucred creds = {};
  socklen_t len = sizeof creds;
  if (getsockopt(socket, SOL_SOCKET, SO_PEERCRED, &creds, &len) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "can't get server pid from connection");
  }
  return creds.pid;
}

uint64_t MonotonicClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

uint64_t ProcessClock() {
  struct timespec ts = {};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // TODO(bazel-team): There should be a similar function on Windows.
}

string GetProcessCWD(int pid) {
  char server_cwd[PATH_MAX] = {};
  if (readlink(
          ("/proc/" + ToString(pid) + "/cwd").c_str(),
          server_cwd, sizeof(server_cwd)) < 0) {
    return "";
  }

  return string(server_cwd);
}

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".dll");
}

string GetDefaultHostJavabase() {
  const char *javahome = getenv("JAVA_HOME");
  if (javahome == NULL) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Error: JAVA_HOME not set.");
  }
  return javahome;
}

namespace {
void ReplaceAll(
        std::string* s, const std::string& pattern, const std::string with) {
  size_t pos = 0;
  while (true) {
    size_t pos = s->find(pattern, pos);
    if (pos == std::string::npos) return;
    *s = s->replace(pos, pattern.length(), with);
    pos += with.length();
  }
}
}  // namespace

// Replace the current process with the given program in the given working
// directory, using the given argument vector.
// This function does not return on success.
void ExecuteProgram(const string& exe, const vector<string>& args_vector) {
  if (VerboseLogging()) {
    string dbg;
    for (const auto& s : args_vector) {
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

  // Build full command line.
  string cmdline;
  bool first = true;
  for (const auto& s : args_vector) {
    if (first) {
      first = false;
      // Skip first argument, instead use quoted executable name with ".exe"
      // suffix.
      cmdline.append("\"");
      cmdline.append(exe);
      cmdline.append(".exe");
      cmdline.append("\"");
      continue;
    } else {
      cmdline.append(" ");
    }

    string arg = s;
    // Quote quotes.
    if (s.find("\"") != string::npos) {
      ReplaceAll(&arg, "\"", "\\\"");
    }

    // Quotize spaces.
    if (arg.find(" ") != string::npos) {
      cmdline.append("\"");
      cmdline.append(arg);
      cmdline.append("\"");
    } else {
      cmdline.append(arg);
    }
  }

  // Copy command line into a mutable buffer.
  // CreateProcess is allowed to mutate its command line argument.
  // Max command line length is per CreateProcess documentation
  // (https://msdn.microsoft.com/en-us/library/ms682425(VS.85).aspx)
  static const int kMaxCmdLineLength = 32768;
  char actual_line[kMaxCmdLineLength];
  if (cmdline.length() >= kMaxCmdLineLength) {
    pdie(255, "Command line too long: %s", cmdline.c_str());
  }
  strncpy(actual_line, cmdline.c_str(), kMaxCmdLineLength);
  // Add trailing '\0' to be sure.
  actual_line[kMaxCmdLineLength - 1] = '\0';

  // Execute program.
  STARTUPINFO startupinfo = {0};
  PROCESS_INFORMATION pi = {0};

  // Propagate BAZEL_SH environment variable to a sub-process.
  // todo(dslomov): More principled approach to propagating
  // environment variables.
  SetEnvironmentVariable("BAZEL_SH", getenv("BAZEL_SH"));

  bool success = CreateProcess(
      nullptr,       // _In_opt_    LPCTSTR               lpApplicationName,
      actual_line,   // _Inout_opt_ LPTSTR                lpCommandLine,
      nullptr,       // _In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
      nullptr,       // _In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
      true,          // _In_        BOOL                  bInheritHandles,
      0,             // _In_        DWORD                 dwCreationFlags,
      nullptr,       // _In_opt_    LPVOID                lpEnvironment,
      nullptr,       // _In_opt_    LPCTSTR               lpCurrentDirectory,
      &startupinfo,  // _In_        LPSTARTUPINFO         lpStartupInfo,
      &pi);          // _Out_       LPPROCESS_INFORMATION lpProcessInformation

  if (!success) {
    pdie(255, "Error %u executing: %s\n", GetLastError(), actual_line);
  }
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  // Emulate execv.
  exit(exit_code);
}

string ListSeparator() { return ";"; }

string ConvertPath(const string& path) {
  char* wpath = static_cast<char*>(cygwin_create_path(
      CCP_POSIX_TO_WIN_A, static_cast<const void*>(path.c_str())));
  string result(wpath);
  free(wpath);
  return result;
}

}  // namespace blaze
