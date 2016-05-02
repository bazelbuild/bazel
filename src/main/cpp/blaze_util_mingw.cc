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

static void PrintError(const string& op) {
    DWORD last_error = ::GetLastError();
    if (last_error == 0) {
        return;
    }

    char* message_buffer;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER
            | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        last_error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR) &message_buffer,
        0,
        NULL);

    fprintf(stderr, "ERROR: %s: %s (%d)\n",
            op.c_str(), message_buffer, last_error);
    LocalFree(message_buffer);
}

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
  char* tmpdir = getenv("TMPDIR");
  if (tmpdir == 0 || strlen(tmpdir) == 0) {
    return "/var/tmp";
  } else {
    return string(tmpdir);
  }
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

// Max command line length is per CreateProcess documentation
// (https://msdn.microsoft.com/en-us/library/ms682425(VS.85).aspx)
static const int MAX_CMDLINE_LENGTH = 32768;

struct CmdLine {
  char cmdline[MAX_CMDLINE_LENGTH];
};
static void CreateCommandLine(CmdLine* result, const string& exe,
                              const vector<string>& args_vector) {
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

  if (cmdline.size() >= MAX_CMDLINE_LENGTH) {
    pdie(blaze_exit_code::INTERNAL_ERROR,
         "Command line too long: %s", cmdline.c_str());
  }

  // Copy command line into a mutable buffer.
  // CreateProcess is allowed to mutate its command line argument.
  strncpy(result->cmdline, cmdline.c_str(), MAX_CMDLINE_LENGTH - 1);
  result->cmdline[MAX_CMDLINE_LENGTH - 1] = 0;
}

}  // namespace

string RunProgram(
    const string& exe, const vector<string>& args_vector) {
  SECURITY_ATTRIBUTES sa = {0};

  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE pipe_read, pipe_write;
  if (!CreatePipe(&pipe_read, &pipe_write, &sa, 0)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "CreatePipe");
  }

  if (!SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "SetHandleInformation");
  }

  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFO startupInfo = {0};

  startupInfo.hStdError = pipe_write;
  startupInfo.hStdOutput = pipe_write;
  startupInfo.dwFlags |= STARTF_USESTDHANDLES;
  CmdLine cmdline;
  CreateCommandLine(&cmdline, exe, args_vector);

  bool ok = CreateProcess(
      NULL,           // _In_opt_    LPCTSTR               lpApplicationName,
      //                 _Inout_opt_ LPTSTR                lpCommandLine,
      cmdline.cmdline,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
      true,           // _In_        BOOL                  bInheritHandles,
      0,              // _In_        DWORD                 dwCreationFlags,
      NULL,           // _In_opt_    LPVOID                lpEnvironment,
      NULL,           // _In_opt_    LPCTSTR               lpCurrentDirectory,
      &startupInfo,   // _In_        LPSTARTUPINFO         lpStartupInfo,
      &processInfo);  // _Out_       LPPROCESS_INFORMATION lpProcessInformation

  if (!ok) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "CreateProcess");
  }

  CloseHandle(pipe_write);
  std::string result = "";
  DWORD bytes_read;
  CHAR buf[1024];

  for (;;) {
    ok = ::ReadFile(pipe_read, buf, 1023, &bytes_read, NULL);
    if (!ok || bytes_read == 0) {
      break;
    }
    buf[bytes_read] = 0;
    result = result + buf;
  }

  CloseHandle(pipe_read);
  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);

  return result;
}

// If we pass DETACHED_PROCESS to CreateProcess(), cmd.exe appropriately
// returns the command prompt when the client terminates. msys2, however, in
// its infinite wisdom, waits until the *server* terminates and cannot be
// convinced otherwise.
//
// So, we first pretend to be a POSIX daemon so that msys2 knows about our
// intentions and *then* we call CreateProcess(). Life ain't easy.
static bool DaemonizeOnWindows() {
  if (fork() > 0) {
    // We are the original client process.
    return true;
  }

  if (fork() > 0) {
    // We are the child of the original client process. Terminate so that the
    // actual server is not a child process of the client.
    exit(0);
  }

  setsid();
  // Contrary to the POSIX version, we are not closing the three standard file
  // descriptors here. CreateProcess() will take care of that and it's useful
  // to see the error messages in ExecuteDaemon() on the console of the client.
  return false;
}

// Keeping an eye on the server process on Windows is not implemented yet.
// TODO(lberki): Implement this, because otherwise if we can't start up a server
// process, the client will hang until it times out.
class DummyBlazeServerStartup : public BlazeServerStartup {
 public:
  DummyBlazeServerStartup() {}
  virtual ~DummyBlazeServerStartup() {}
  bool IsStillAlive() override { return true; }
};

void ExecuteDaemon(const string& exe, const std::vector<string>& args_vector,
                   const string& daemon_output, const string& server_dir,
                   BlazeServerStartup** server_startup) {
  if (DaemonizeOnWindows()) {
    // We are the client process
    *server_startup = new DummyBlazeServerStartup();
    return;
  }

  SECURITY_ATTRIBUTES sa;

  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE output_file;

  if (!CreateFile(
      daemon_output.c_str(),
      GENERIC_READ | GENERIC_WRITE,
      0,
      &sa,
      CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      NULL)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "CreateFile");
  }

  HANDLE pipe_read, pipe_write;
  if (!CreatePipe(&pipe_read, &pipe_write, &sa, 0)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "CreatePipe");
  }

  if (!SetHandleInformation(pipe_write, HANDLE_FLAG_INHERIT, 0)) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "SetHandleInformation");
  }

  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFO startupInfo = {0};

  startupInfo.hStdInput = pipe_read;
  startupInfo.hStdError = output_file;
  startupInfo.hStdOutput = output_file;
  startupInfo.dwFlags |= STARTF_USESTDHANDLES;
  CmdLine cmdline;
  CreateCommandLine(&cmdline, exe, args_vector);

  // Propagate BAZEL_SH environment variable to a sub-process.
  // todo(dslomov): More principled approach to propagating
  // environment variables.
  SetEnvironmentVariable("BAZEL_SH", getenv("BAZEL_SH"));

  bool ok = CreateProcess(
      NULL,           // _In_opt_    LPCTSTR               lpApplicationName,
      //                 _Inout_opt_ LPTSTR                lpCommandLine,
      cmdline.cmdline,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
      TRUE,           // _In_        BOOL                  bInheritHandles,
      //                 _In_        DWORD                 dwCreationFlags,
      DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
      NULL,           // _In_opt_    LPVOID                lpEnvironment,
      NULL,           // _In_opt_    LPCTSTR               lpCurrentDirectory,
      &startupInfo,   // _In_        LPSTARTUPINFO         lpStartupInfo,
      &processInfo);  // _Out_       LPPROCESS_INFORMATION lpProcessInformation

  if (!ok) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "CreateProcess");
  }

  CloseHandle(output_file);
  CloseHandle(pipe_write);
  CloseHandle(pipe_read);

  string pid_string = ToString(getpid());
  string pid_file = blaze_util::JoinPath(server_dir, ServerPidFile());
  if (!WriteFile(pid_string, pid_file)) {
    // Not a lot we can do if this fails
    fprintf(stderr, "Cannot write PID file %s\n", pid_file.c_str());
  }

  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);

  exit(0);
}

// Run the given program in the current working directory,
// using the given argument vector.
void ExecuteProgram(
    const string& exe, const vector<string>& args_vector) {
  CmdLine cmdline;
  CreateCommandLine(&cmdline, exe, args_vector);

  STARTUPINFO startupInfo = {0};
  PROCESS_INFORMATION processInfo = {0};

  // Propagate BAZEL_SH environment variable to a sub-process.
  // todo(dslomov): More principled approach to propagating
  // environment variables.
  SetEnvironmentVariable("BAZEL_SH", getenv("BAZEL_SH"));

  bool success = CreateProcess(
      NULL,           // _In_opt_    LPCTSTR               lpApplicationName,
      //                 _Inout_opt_ LPTSTR                lpCommandLine,
      cmdline.cmdline,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpProcessAttributes,
      NULL,           // _In_opt_    LPSECURITY_ATTRIBUTES lpThreadAttributes,
      true,           // _In_        BOOL                  bInheritHandles,
      0,              // _In_        DWORD                 dwCreationFlags,
      NULL,           // _In_opt_    LPVOID                lpEnvironment,
      NULL,           // _In_opt_    LPCTSTR               lpCurrentDirectory,
      &startupInfo,   // _In_        LPSTARTUPINFO         lpStartupInfo,
      &processInfo);  // _Out_       LPPROCESS_INFORMATION lpProcessInformation

  if (!success) {
    pdie(255, "Error %u executing: %s\n", GetLastError(), cmdline);
  }
  // The output base lock is held while waiting
  WaitForSingleObject(processInfo.hProcess, INFINITE);
  DWORD exit_code;
  GetExitCodeProcess(processInfo.hProcess, &exit_code);
  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);
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

string ConvertPathToPosix(const string& win_path) {
  char* posix_path = static_cast<char*>(cygwin_create_path(
      CCP_WIN_A_TO_POSIX, static_cast<const void*>(win_path.c_str())));
  string result(posix_path);
  free(posix_path);
  return result;
}

// Cribbed from ntifs.h, not present in windows.h

#define REPARSE_MOUNTPOINT_HEADER_SIZE   8

typedef struct {
  DWORD ReparseTag;
  WORD ReparseDataLength;
  WORD Reserved;
  WORD SubstituteNameOffset;
  WORD SubstituteNameLength;
  WORD PrintNameOffset;
  WORD PrintNameLength;
  WCHAR PathBuffer[ANYSIZE_ARRAY];
} REPARSE_MOUNTPOINT_DATA_BUFFER, *PREPARSE_MOUNTPOINT_DATA_BUFFER;

HANDLE OpenDirectory(const string& path, bool readWrite) {
  HANDLE result = ::CreateFile(
      path.c_str(),
      readWrite ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ,
      0,
      NULL,
      OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
      NULL);
  if (result == INVALID_HANDLE_VALUE) {
    PrintError("CreateFile(" + path + ")");
  }

  return result;
}

bool SymlinkDirectories(const string &posix_target, const string &posix_name) {
  string target = ConvertPath(posix_target);
  string name = ConvertPath(posix_name);

  // Junctions are directories, so create one
  if (!::CreateDirectory(name.c_str(), NULL)) {
    PrintError("CreateDirectory(" + name + ")");
    return false;
  }

  HANDLE directory = OpenDirectory(name, true);
  if (directory == INVALID_HANDLE_VALUE) {
    return false;
  }

  char reparse_buffer_bytes[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
  REPARSE_MOUNTPOINT_DATA_BUFFER* reparse_buffer =
      reinterpret_cast<REPARSE_MOUNTPOINT_DATA_BUFFER *>(reparse_buffer_bytes);
  memset(reparse_buffer_bytes, 0, MAXIMUM_REPARSE_DATA_BUFFER_SIZE);

  // non-parsed path prefix. Required for junction targets.
  string prefixed_target = "\\??\\" + target;
  int prefixed_target_length = ::MultiByteToWideChar(
      CP_ACP,
      0,
      prefixed_target.c_str(),
      -1,
      reparse_buffer->PathBuffer,
      MAX_PATH);
  if (prefixed_target_length == 0) {
    PrintError("MultiByteToWideChar(" + prefixed_target + ")");
    CloseHandle(directory);
    return false;
  }

  // In addition to their target, junctions also have another string which
  // tells which target to show to the user. mklink cuts of the \??\ part, so
  // that's what we do, too.
  int target_length = ::MultiByteToWideChar(
      CP_UTF8,
      0,
      target.c_str(),
      -1,
      reparse_buffer->PathBuffer + prefixed_target_length,
      MAX_PATH);
  if (target_length == 0) {
    PrintError("MultiByteToWideChar(" + target + ")");
    CloseHandle(directory);
    return false;
  }

  reparse_buffer->ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
  reparse_buffer->PrintNameOffset = prefixed_target_length * sizeof(WCHAR);
  reparse_buffer->PrintNameLength = (target_length - 1) * sizeof(WCHAR);
  reparse_buffer->SubstituteNameLength =
      (prefixed_target_length - 1) * sizeof(WCHAR);
  reparse_buffer->SubstituteNameOffset = 0;
  reparse_buffer->Reserved = 0;
  reparse_buffer->ReparseDataLength =
       reparse_buffer->SubstituteNameLength +
       reparse_buffer->PrintNameLength + 12;

  DWORD bytes_returned;
  bool result = ::DeviceIoControl(
      directory,
      FSCTL_SET_REPARSE_POINT,
      reparse_buffer,
      reparse_buffer->ReparseDataLength + REPARSE_MOUNTPOINT_HEADER_SIZE,
      NULL,
      0,
      &bytes_returned,
      NULL);
  if (!result) {
    PrintError("DeviceIoControl(FSCTL_SET_REPARSE_POINT, " + name + ")");
  }
  CloseHandle(directory);
  return result;
}

bool ReadDirectorySymlink(const string &posix_name, string* result) {
  string name = ConvertPath(posix_name);
  HANDLE directory = OpenDirectory(name, false);
  if (directory == INVALID_HANDLE_VALUE) {
    return false;
  }

  char reparse_buffer_bytes[MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
  REPARSE_MOUNTPOINT_DATA_BUFFER* reparse_buffer =
      reinterpret_cast<REPARSE_MOUNTPOINT_DATA_BUFFER *>(reparse_buffer_bytes);
  memset(reparse_buffer_bytes, 0, MAXIMUM_REPARSE_DATA_BUFFER_SIZE);

  reparse_buffer->ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
  DWORD bytes_returned;
  bool ok = ::DeviceIoControl(
      directory,
      FSCTL_GET_REPARSE_POINT,
      NULL,
      0,
      reparse_buffer,
      MAXIMUM_REPARSE_DATA_BUFFER_SIZE,
      &bytes_returned,
      NULL);
  if (!ok) {
    PrintError("DeviceIoControl(FSCTL_GET_REPARSE_POINT, " + name + ")");
  }

  CloseHandle(directory);
  if (!ok) {
    return false;
  }

  char print_name[MAX_PATH];
  int count = ::WideCharToMultiByte(
      CP_UTF8,
      0,
      reparse_buffer->PathBuffer +
         (reparse_buffer->PrintNameOffset / sizeof(WCHAR)),
      reparse_buffer->PrintNameLength,
      print_name,
      MAX_PATH,
      NULL,
      NULL);
  if (count == 0) {
    PrintError("WideCharToMultiByte()");
    *result = "";
    return false;
  } else {
    *result = ConvertPathToPosix(print_name);
    return true;
  }
}

static bool IsAbsoluteWindowsPath(const string& p) {
  if (p.size() < 3) {
    return false;
  }

  if (p.substr(1, 2) == ":/") {
    return true;
  }

  if (p.substr(1, 2) == ":\\") {
    return true;
  }

  return false;
}

bool CompareAbsolutePaths(const string& a, const string& b) {
  string a_real = IsAbsoluteWindowsPath(a) ? ConvertPathToPosix(a) : a;
  string b_real = IsAbsoluteWindowsPath(b) ? ConvertPathToPosix(b) : b;
  return a_real == b_real;
}

void KillServerProcess(
    int pid, const string& output_base, const string& install_base) {
  // Not implemented yet. TerminateProcess should work.
}

}  // namespace blaze
