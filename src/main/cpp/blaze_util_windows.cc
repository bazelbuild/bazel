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

#include "src/main/cpp/blaze_util_platform.h"

#include <fcntl.h>
#include <stdarg.h>  // va_start, va_end, va_list

#include <windows.h>
#include <lmcons.h>          // UNLEN
#include <versionhelpers.h>  // IsWindows8OrGreater

#include <io.h>            // _open
#include <knownfolders.h>  // FOLDERID_Profile
#include <objbase.h>       // CoTaskMemFree
#include <shlobj.h>        // SHGetKnownFolderPath

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <sstream>
#include <thread>       // NOLINT (to silence Google-internal linter)
#include <type_traits>  // static_assert
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/global_variables.h"
#include "src/main/cpp/startup_options.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/md5.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"

namespace blaze {

// Ensure we can safely cast (const) wchar_t* to LP(C)WSTR.
// This is true with MSVC but usually not with GCC.
static_assert(sizeof(wchar_t) == sizeof(WCHAR),
              "wchar_t and WCHAR should be the same size");

// When using widechar Win32 API functions the maximum path length is 32K.
// Add 4 characters for potential UNC prefix and a couple more for safety.
static const size_t kWindowsPathBufferSize = 0x8010;

using bazel::windows::AutoAttributeList;
using bazel::windows::AutoHandle;
using bazel::windows::CreateJunction;
using bazel::windows::CreateJunctionResult;

// TODO(bazel-team): stop using BAZEL_DIE, handle errors on the caller side.
// BAZEL_DIE calls exit(exitcode), which makes it difficult to follow the
// control flow and does not call destructors on local variables on the call
// stack.
using blaze_util::GetLastErrorString;

using std::string;
using std::unique_ptr;
using std::wstring;

namespace embedded_binaries {

class WindowsDumper : public Dumper {
 public:
  static WindowsDumper* Create(string* error);
  ~WindowsDumper() { Finish(nullptr); }
  void Dump(const void* data, const size_t size, const string& path) override;
  bool Finish(string* error) override;

 private:
  WindowsDumper() : threadpool_(NULL), cleanup_group_(NULL) {}

  PTP_POOL threadpool_;
  PTP_CLEANUP_GROUP cleanup_group_;
  TP_CALLBACK_ENVIRON threadpool_env_;

  std::mutex dir_cache_lock_;
  std::set<string> dir_cache_;

  std::mutex error_lock_;
  string error_msg_;
};

namespace {

class DumpContext {
 public:
  DumpContext(unique_ptr<uint8_t[]> data, const size_t size, const string path,
              std::mutex* dir_cache_lock, std::set<string>* dir_cache,
              std::mutex* error_lock_, string* error_msg);
  void Run();

 private:
  void MaybeSignalError(const string& msg);

  unique_ptr<uint8_t[]> data_;
  const size_t size_;
  const string path_;

  std::mutex* dir_cache_lock_;
  std::set<string>* dir_cache_;

  std::mutex* error_lock_;
  string* error_msg_;
};

VOID CALLBACK WorkCallback(_Inout_ PTP_CALLBACK_INSTANCE Instance,
                           _Inout_opt_ PVOID Context, _Inout_ PTP_WORK Work);

}  // namespace

Dumper* Create(string* error) { return WindowsDumper::Create(error); }

WindowsDumper* WindowsDumper::Create(string* error) {
  unique_ptr<WindowsDumper> result(new WindowsDumper());

  result->threadpool_ = CreateThreadpool(NULL);
  if (result->threadpool_ == NULL) {
    if (error) {
      string msg = GetLastErrorString();
      *error = "CreateThreadpool failed: " + msg;
    }
    return nullptr;
  }

  result->cleanup_group_ = CreateThreadpoolCleanupGroup();
  if (result->cleanup_group_ == NULL) {
    string msg = GetLastErrorString();
    CloseThreadpool(result->threadpool_);
    if (error) {
      string msg = GetLastErrorString();
      *error = "CreateThreadpoolCleanupGroup failed: " + msg;
    }
    return nullptr;
  }

  // I (@laszlocsomor) experimented with different thread counts and found that
  // 8 threads provide a significant advantage over 1 thread, but adding more
  // threads provides only marginal speedup.
  SetThreadpoolThreadMaximum(result->threadpool_, 16);
  SetThreadpoolThreadMinimum(result->threadpool_, 8);

  InitializeThreadpoolEnvironment(&result->threadpool_env_);
  SetThreadpoolCallbackPool(&result->threadpool_env_, result->threadpool_);
  SetThreadpoolCallbackCleanupGroup(&result->threadpool_env_,
                                    result->cleanup_group_, NULL);

  return result.release();  // release pointer ownership
}

void WindowsDumper::Dump(const void* data, const size_t size,
                         const string& path) {
  {
    std::lock_guard<std::mutex> g(error_lock_);
    if (!error_msg_.empty()) {
      return;
    }
  }

  unique_ptr<uint8_t[]> data_copy(new uint8_t[size]);
  memcpy(data_copy.get(), data, size);
  unique_ptr<DumpContext> ctx(new DumpContext(std::move(data_copy), size, path,
                                              &dir_cache_lock_, &dir_cache_,
                                              &error_lock_, &error_msg_));
  PTP_WORK w = CreateThreadpoolWork(WorkCallback, ctx.get(), &threadpool_env_);
  if (w == NULL) {
    string err = GetLastErrorString();
    err = string("WindowsDumper::Dump() couldn't submit work: ") + err;

    std::lock_guard<std::mutex> g(error_lock_);
    error_msg_ = err;
  } else {
    ctx.release();  // release pointer ownership
    SubmitThreadpoolWork(w);
  }
}

bool WindowsDumper::Finish(string* error) {
  if (threadpool_ == NULL) {
    return true;
  }
  CloseThreadpoolCleanupGroupMembers(cleanup_group_, FALSE, NULL);
  CloseThreadpoolCleanupGroup(cleanup_group_);
  CloseThreadpool(threadpool_);
  threadpool_ = NULL;
  cleanup_group_ = NULL;

  std::lock_guard<std::mutex> g(error_lock_);
  if (!error_msg_.empty() && error) {
    *error = error_msg_;
  }
  return error_msg_.empty();
}

namespace {

DumpContext::DumpContext(unique_ptr<uint8_t[]> data, const size_t size,
                         const string path, std::mutex* dir_cache_lock,
                         std::set<string>* dir_cache, std::mutex* error_lock_,
                         string* error_msg)
    : data_(std::move(data)),
      size_(size),
      path_(path),
      dir_cache_lock_(dir_cache_lock),
      dir_cache_(dir_cache),
      error_msg_(error_msg) {}

void DumpContext::Run() {
  string dirname = blaze_util::Dirname(path_);

  bool success = true;
  // Performance optimization: memoize the paths we already created a
  // directory for, to spare a stat in attempting to recreate an already
  // existing directory. This optimization alone shaves off seconds from the
  // extraction time on Windows.
  {
    std::lock_guard<std::mutex> guard(*dir_cache_lock_);
    if (dir_cache_->insert(dirname).second) {
      success = blaze_util::MakeDirectories(dirname, 0777);
    }
  }

  if (!success) {
    MaybeSignalError(string("Couldn't create directory '") + dirname + "'");
    return;
  }

  if (!blaze_util::WriteFile(data_.get(), size_, path_, 0755)) {
    MaybeSignalError(string("Failed to write zipped file '") + path_ + "'");
  }
}

void DumpContext::MaybeSignalError(const string& msg) {
  std::lock_guard<std::mutex> g(*error_lock_);
  *error_msg_ = msg;
}

VOID CALLBACK WorkCallback(_Inout_ PTP_CALLBACK_INSTANCE Instance,
                           _Inout_opt_ PVOID Context, _Inout_ PTP_WORK Work) {
  unique_ptr<DumpContext> ctx(reinterpret_cast<DumpContext*>(Context));
  ctx->Run();
}

}  // namespace

}  // namespace embedded_binaries

SignalHandler SignalHandler::INSTANCE;

class WindowsClock {
 public:
  uint64_t GetMilliseconds() const;

  static const WindowsClock INSTANCE;

 private:
  // Clock frequency per seconds.
  // It's safe to cache this because (from QueryPerformanceFrequency on MSDN):
  // "The frequency of the performance counter is fixed at system boot and is
  // consistent across all processors. Therefore, the frequency need only be
  // queried upon application initialization, and the result can be cached."
  const LARGE_INTEGER kFrequency;

  WindowsClock();

  static LARGE_INTEGER GetFrequency();
  static LARGE_INTEGER GetMillisecondsAsLargeInt(const LARGE_INTEGER& freq);
};

BOOL WINAPI ConsoleCtrlHandler(_In_ DWORD ctrlType) {
  static volatile int sigint_count = 0;
  switch (ctrlType) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
      if (++sigint_count >= 3) {
        SigPrintf(
            "\n%s caught third Ctrl+C handler signal; killed.\n\n",
            SignalHandler::Get().GetGlobals()->options->product_name.c_str());
        if (SignalHandler::Get().GetGlobals()->server_pid != -1) {
          KillServerProcess(
              SignalHandler::Get().GetGlobals()->server_pid,
              SignalHandler::Get().GetGlobals()->options->output_base);
        }
        _exit(1);
      }
      SigPrintf(
          "\n%s Ctrl+C handler; shutting down.\n\n",
          SignalHandler::Get().GetGlobals()->options->product_name.c_str());
      SignalHandler::Get().CancelServer();
      return TRUE;

    case CTRL_CLOSE_EVENT:
      SignalHandler::Get().CancelServer();
      return TRUE;
  }
  return false;
}

void SignalHandler::Install(GlobalVariables* globals,
                            SignalHandler::Callback cancel_server) {
  _globals = globals;
  _cancel_server = cancel_server;
  ::SetConsoleCtrlHandler(&ConsoleCtrlHandler, TRUE);
}

ATTRIBUTE_NORETURN void SignalHandler::PropagateSignalOrExit(int exit_code) {
  // We do not handle signals on Windows; always exit with exit_code.
  exit(exit_code);
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
  int stderr_fileno = _fileno(stderr);
  char buf[1024];
  va_list ap;
  va_start(ap, format);
  int r = vsnprintf(buf, sizeof buf, format, ap);
  va_end(ap);
  if (write(stderr_fileno, buf, r) <= 0) {
    // We don't care, just placate the compiler.
  }
}

static void PrintErrorW(const wstring& op) {
  DWORD last_error = ::GetLastError();
  if (last_error == 0) {
    return;
  }

  WCHAR* message_buffer;
  FormatMessageW(
      /* dwFlags */ FORMAT_MESSAGE_ALLOCATE_BUFFER |
          FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      /* lpSource */ nullptr,
      /* dwMessageId */ last_error,
      /* dwLanguageId */ MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      /* lpBuffer */ message_buffer,
      /* nSize */ 0,
      /* Arguments */ nullptr);

  fwprintf(stderr, L"ERROR: %s: %s (%d)\n", op.c_str(), message_buffer,
           last_error);
  LocalFree(message_buffer);
}

void WarnFilesystemType(const string& output_base) {
}

string GetProcessIdAsString() {
  return ToString(GetCurrentProcessId());
}

string GetSelfPath() {
  WCHAR buffer[kWindowsPathBufferSize] = {0};
  if (!GetModuleFileNameW(0, buffer, kWindowsPathBufferSize)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "GetSelfPath: GetModuleFileNameW: " << GetLastErrorString();
  }
  return string(blaze_util::WstringToCstring(buffer).get());
}

string GetOutputRoot() {
  string home = GetHomeDir();
  if (home.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Cannot find a good output root.\n"
           "Set the USERPROFILE or the HOME environment variable.\n"
           "Example (in cmd.exe):\n"
           "    set USERPROFILE=c:\\_bazel\\<YOUR-USERNAME>\n"
           "or:\n"
           "    set HOME=c:\\_bazel\\<YOUR-USERNAME>";
  }
  return home;
}

string GetHomeDir() {
  if (IsRunningWithinTest()) {
    // Bazel is running inside of a test. Respect $HOME that the test setup has
    // set instead of using the actual home directory of the current user.
    return GetPathEnv("HOME");
  }

  PWSTR wpath;
  // Look up the user's home directory. The default value of "FOLDERID_Profile"
  // is the same as %USERPROFILE%, but it does not require the envvar to be set.
  if (SUCCEEDED(::SHGetKnownFolderPath(FOLDERID_Profile, KF_FLAG_DEFAULT, NULL,
                                       &wpath))) {
    string result = string(blaze_util::WstringToCstring(wpath).get());
    ::CoTaskMemFree(wpath);
    return result;
  }

  // On Windows 2016 Server, Nano server: FOLDERID_Profile is unknown but
  // %USERPROFILE% is set. See https://github.com/bazelbuild/bazel/issues/6701
  string userprofile = GetPathEnv("USERPROFILE");
  if (!userprofile.empty()) {
    return userprofile;
  }

  return GetPathEnv("HOME");  // only defined in MSYS/Cygwin
}

string FindSystemWideBlazerc() {
  // TODO(bazel-team): figure out a good path to return here.
  return "";
}

string GetJavaBinaryUnderJavabase() { return "bin/java.exe"; }

uint64_t GetMillisecondsMonotonic() {
  return WindowsClock::INSTANCE.GetMilliseconds();
}

void SetScheduling(bool batch_cpu_scheduling, int io_nice_level) {
  // TODO(bazel-team): There should be a similar function on Windows.
}

string GetProcessCWD(int pid) {
  // TODO(bazel-team) 2016-11-18: decide whether we need this on Windows and
  // implement or delete.
  return "";
}

bool IsSharedLibrary(const string &filename) {
  return blaze_util::ends_with(filename, ".dll");
}

string GetSystemJavabase() {
  string javahome(GetPathEnv("JAVA_HOME"));
  if (!javahome.empty()) {
    string javac = blaze_util::JoinPath(javahome, "bin/javac.exe");
    if (blaze_util::PathExists(javac.c_str())) {
      return javahome;
    }
    BAZEL_LOG(WARNING)
        << "Ignoring JAVA_HOME, because it must point to a JDK, not a JRE.";
  }

  return "";
}

namespace {

// Max command line length is per CreateProcess documentation
// (https://msdn.microsoft.com/en-us/library/ms682425(VS.85).aspx)
//
// Quoting rules are described here:
// https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/everyone-quotes-command-line-arguments-the-wrong-way/

static const int MAX_CMDLINE_LENGTH = 32768;

struct CmdLine {
  char cmdline[MAX_CMDLINE_LENGTH];
};
static void CreateCommandLine(CmdLine* result, const string& exe,
                              const std::vector<string>& args_vector) {
  std::ostringstream cmdline;
  string short_exe;
  string error;
  if (!blaze_util::AsShortWindowsPath(exe, &short_exe, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "CreateCommandLine: AsShortWindowsPath(" << exe << "): " << error;
  }
  bool first = true;
  for (const auto& s : args_vector) {
    if (first) {
      first = false;
      // Skip first argument, instead use quoted executable name.
      cmdline << '\"' << short_exe << '\"';
      continue;
    } else {
      cmdline << ' ';
    }

    bool has_space = s.find(" ") != string::npos;

    if (has_space) {
      cmdline << '\"';
    }

    std::string::const_iterator it = s.begin();
    while (it != s.end()) {
      char ch = *it++;
      switch (ch) {
        case '"':
          // Escape double quotes
          cmdline << "\\\"";
          break;

        case '\\':
          if (it == s.end()) {
            // Backslashes at the end of the string are quoted if we add quotes
            cmdline << (has_space ? "\\\\" : "\\");
          } else {
            // Backslashes everywhere else are quoted if they are followed by a
            // quote or a backslash
            cmdline << (*it == '"' || *it == '\\' ? "\\\\" : "\\");
          }
          break;

        default:
          cmdline << ch;
      }
    }

    if (has_space) {
      cmdline << '\"';
    }
  }

  string cmdline_str = cmdline.str();
  if (cmdline_str.size() >= MAX_CMDLINE_LENGTH) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "Command line too long (" << cmdline_str.size() << " > "
        << MAX_CMDLINE_LENGTH << "): " << cmdline_str;
  }

  // Copy command line into a mutable buffer.
  // CreateProcess is allowed to mutate its command line argument.
  strncpy(result->cmdline, cmdline_str.c_str(), MAX_CMDLINE_LENGTH - 1);
  result->cmdline[MAX_CMDLINE_LENGTH - 1] = 0;
}

}  // namespace

static bool GetProcessStartupTime(HANDLE process, uint64_t* result) {
  FILETIME creation_time, dummy1, dummy2, dummy3;
  // GetProcessTimes cannot handle NULL arguments.
  if (process == INVALID_HANDLE_VALUE ||
      !::GetProcessTimes(process, &creation_time, &dummy1, &dummy2, &dummy3)) {
    return false;
  }
  *result = static_cast<uint64_t>(creation_time.dwHighDateTime) << 32 |
            creation_time.dwLowDateTime;
  return true;
}

static void WriteProcessStartupTime(const string& server_dir, HANDLE process) {
  uint64_t start_time = 0;
  if (!GetProcessStartupTime(process, &start_time)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WriteProcessStartupTime(" << server_dir
        << "): GetProcessStartupTime failed: " << GetLastErrorString();
  }

  string start_time_file = blaze_util::JoinPath(server_dir, "server.starttime");
  if (!blaze_util::WriteFile(ToString(start_time), start_time_file)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WriteProcessStartupTime(" << server_dir << "): WriteFile("
        << start_time_file << ") failed: " << GetLastErrorString();
  }
}

static HANDLE CreateJvmOutputFile(const wstring& path, LPSECURITY_ATTRIBUTES sa,
                                  bool daemon_out_append) {
  // If the previous server process was asked to be shut down (but not killed),
  // it takes a while for it to comply, so wait until the JVM output file that
  // it held open is closed. There seems to be no better way to wait for a file
  // to be closed on Windows.
  static const unsigned int timeout_sec = 60;
  for (unsigned int waited = 0; waited < timeout_sec; ++waited) {
    HANDLE handle = ::CreateFileW(
        /* lpFileName */ path.c_str(),
        /* dwDesiredAccess */ GENERIC_READ | GENERIC_WRITE,
        /* dwShareMode */ FILE_SHARE_READ,
        /* lpSecurityAttributes */ sa,
        /* dwCreationDisposition */
            daemon_out_append ? OPEN_ALWAYS : CREATE_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);
    if (handle != INVALID_HANDLE_VALUE) {
      if (daemon_out_append
          && !SetFilePointerEx(handle, {0}, NULL, FILE_END)) {
        fprintf(stderr, "Could not seek to end of file (%ls)\n", path.c_str());
        return INVALID_HANDLE_VALUE;
      }
      return handle;
    }
    if (GetLastError() != ERROR_SHARING_VIOLATION &&
        GetLastError() != ERROR_LOCK_VIOLATION) {
      // Some other error occurred than the file being open; bail out.
      break;
    }

    // The file is still held open, the server is shutting down. There's a
    // chance that another process holds it open, we don't know; in that case
    // we just exit after the timeout expires.
    if (waited == 5 || waited == 10 || waited == 30) {
      fprintf(stderr,
              "Waiting for previous Bazel server's log file to close "
              "(waited %d seconds, waiting at most %d)\n",
              waited, timeout_sec);
    }
    Sleep(1000);
  }
  return INVALID_HANDLE_VALUE;
}

class ProcessHandleBlazeServerStartup : public BlazeServerStartup {
 public:
  ProcessHandleBlazeServerStartup(HANDLE _proc) : proc(_proc) {}

  bool IsStillAlive() override {
    FILETIME dummy1, exit_time, dummy2, dummy3;
    return GetProcessTimes(proc, &dummy1, &exit_time, &dummy2, &dummy3) &&
           exit_time.dwHighDateTime == 0 && exit_time.dwLowDateTime == 0;
  }

 private:
  AutoHandle proc;
};


int ExecuteDaemon(const string& exe,
                  const std::vector<string>& args_vector,
                  const std::map<string, EnvVarValue>& env,
                  const string& daemon_output,
                  const bool daemon_out_append,
                  const string& binaries_dir,
                  const string& server_dir,
                  BlazeServerStartup** server_startup) {
  wstring wdaemon_output;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(daemon_output, &wdaemon_output,
                                         &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): AsAbsoluteWindowsPath("
        << daemon_output << ") failed: " << error;
  }

  SECURITY_ATTRIBUTES inheritable_handle_sa = {sizeof(SECURITY_ATTRIBUTES),
                                               NULL, TRUE};

  AutoHandle devnull(::CreateFileW(
      L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
      &inheritable_handle_sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
  if (!devnull.IsValid()) {
    error = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): CreateFileA(NUL) failed: " << error;
  }

  AutoHandle stdout_file(CreateJvmOutputFile(
      wdaemon_output.c_str(), &inheritable_handle_sa, daemon_out_append));
  if (!stdout_file.IsValid()) {
    error = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): CreateJvmOutputFile("
        << blaze_util::WstringToString(wdaemon_output) << ") failed: " << error;
  }
  HANDLE stderr_handle;
  // We must duplicate the handle to stdout, otherwise "bazel clean --expunge"
  // won't work, because when it tries to close stdout then stderr, the former
  // will succeed but the latter will appear to be valid yet still fail to
  // close.
  if (!DuplicateHandle(
          /* hSourceProcessHandle */ GetCurrentProcess(),
          /* hSourceHandle */ stdout_file,
          /* hTargetProcessHandle */ GetCurrentProcess(),
          /* lpTargetHandle */ &stderr_handle,
          /* dwDesiredAccess */ 0,
          /* bInheritHandle */ TRUE,
          /* dwOptions */ DUPLICATE_SAME_ACCESS)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): DuplicateHandle("
        << blaze_util::WstringToString(wdaemon_output)
        << ") failed: " << GetLastErrorString();
  }
  AutoHandle stderr_file(stderr_handle);

  // Create an attribute list.
  wstring werror;
  std::unique_ptr<AutoAttributeList> lpAttributeList;
  if (!AutoAttributeList::Create(devnull, stdout_file, stderr_handle,
                                 &lpAttributeList, &werror)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): attribute list creation failed: "
        << blaze_util::WstringToString(werror);
  }

  PROCESS_INFORMATION processInfo = {0};
  STARTUPINFOEXA startupInfoEx = {0};
  lpAttributeList->InitStartupInfoExA(&startupInfoEx);

  CmdLine cmdline;
  CreateCommandLine(&cmdline, exe, args_vector);

  BOOL ok;
  {
    WithEnvVars env_obj(env);

    ok = CreateProcessA(
        /* lpApplicationName */ NULL,
        /* lpCommandLine */ cmdline.cmdline,
        /* lpProcessAttributes */ NULL,
        /* lpThreadAttributes */ NULL,
        /* bInheritHandles */ TRUE,
        /* dwCreationFlags */ DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
        EXTENDED_STARTUPINFO_PRESENT,
        /* lpEnvironment */ NULL,
        /* lpCurrentDirectory */ NULL,
        /* lpStartupInfo */ &startupInfoEx.StartupInfo,
        /* lpProcessInformation */ &processInfo);
  }

  if (!ok) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteDaemon(" << exe << "): CreateProcess(" << cmdline.cmdline
        << ") failed: " << GetLastErrorString();
  }

  WriteProcessStartupTime(server_dir, processInfo.hProcess);

  // Pass ownership of processInfo.hProcess
  *server_startup = new ProcessHandleBlazeServerStartup(processInfo.hProcess);

  string pid_string = ToString(processInfo.dwProcessId);
  string pid_file = blaze_util::JoinPath(server_dir, kServerPidFile);
  if (!blaze_util::WriteFile(pid_string, pid_file)) {
    // Not a lot we can do if this fails
    fprintf(stderr, "Cannot write PID file %s\n", pid_file.c_str());
  }

  // Don't close processInfo.hProcess here, it's now owned by the
  // ProcessHandleBlazeServerStartup instance.
  CloseHandle(processInfo.hThread);

  return processInfo.dwProcessId;
}

// Returns whether nested jobs are not available on the current system.
static bool NestedJobsSupported() {
  // Nested jobs are supported from Windows 8
  return IsWindows8OrGreater();
}

// Run the given program in the current working directory, using the given
// argument vector, wait for it to finish, then exit ourselves with the exitcode
// of that program.
void ExecuteProgram(const string& exe, const std::vector<string>& args_vector) {
  CmdLine cmdline;
  CreateCommandLine(&cmdline, exe, args_vector);

  STARTUPINFOA startupInfo = {0};
  startupInfo.cb = sizeof(STARTUPINFOA);

  PROCESS_INFORMATION processInfo = {0};

  HANDLE job = INVALID_HANDLE_VALUE;
  if (NestedJobsSupported()) {
    job = CreateJobObject(NULL, NULL);
    if (job == NULL) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "ExecuteProgram(" << exe
          << "): CreateJobObject failed: " << GetLastErrorString();
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};
    job_info.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                 &job_info, sizeof(job_info))) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "ExecuteProgram(" << exe
          << "): SetInformationJobObject failed: " << GetLastErrorString();
    }
  }

  BOOL success = CreateProcessA(
      /* lpApplicationName */ NULL,
      /* lpCommandLine */ cmdline.cmdline,
      /* lpProcessAttributes */ NULL,
      /* lpThreadAttributes */ NULL,
      /* bInheritHandles */ TRUE,
      /* dwCreationFlags */ CREATE_SUSPENDED,
      /* lpEnvironment */ NULL,
      /* lpCurrentDirectory */ NULL,
      /* lpStartupInfo */ &startupInfo,
      /* lpProcessInformation */ &processInfo);

  if (!success) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteProgram(" << exe << "): CreateProcess(" << cmdline.cmdline
        << ") failed: " << GetLastErrorString();
  }

  // On Windows versions that support nested jobs (Windows 8 and above), we
  // assign the Bazel server to a job object. Every process that Bazel creates,
  // as well as all their child processes, will be assigned to this job object.
  // When the Bazel server terminates the OS can reliably kill the entire
  // process tree under it. On Windows versions that don't support nested jobs
  // (Windows 7), we don't assign the Bazel server to a big job object. Instead,
  // when Bazel creates new processes, it does so using the JNI library. The
  // library assigns individual job objects to each subprocess. This way when
  // these processes terminate, the OS can kill all their subprocesses. Bazel's
  // own subprocesses are not in a job object though, so we only create
  // subprocesses via the JNI library.
  if (job != INVALID_HANDLE_VALUE) {
    if (!AssignProcessToJobObject(job, processInfo.hProcess)) {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "ExecuteProgram(" << exe
          << "): AssignProcessToJobObject failed: " << GetLastErrorString();
    }
  }
  // Now that we potentially put the process into a new job object, we can start
  // running it.
  if (ResumeThread(processInfo.hThread) == -1) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "ExecuteProgram(" << exe
        << "): ResumeThread failed: " << GetLastErrorString();
  }

  WaitForSingleObject(processInfo.hProcess, INFINITE);
  DWORD exit_code;
  GetExitCodeProcess(processInfo.hProcess, &exit_code);
  CloseHandle(processInfo.hProcess);
  CloseHandle(processInfo.hThread);
  exit(exit_code);
}

const char kListSeparator = ';';

bool SymlinkDirectories(const string &posix_target, const string &posix_name) {
  wstring name;
  wstring target;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(posix_name, &name, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "SymlinkDirectories(" << posix_target << ", " << posix_name
        << "): AsAbsoluteWindowsPath(" << posix_target << ") failed: " << error;
    return false;
  }
  if (!blaze_util::AsAbsoluteWindowsPath(posix_target, &target, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "SymlinkDirectories(" << posix_target << ", " << posix_name
        << "): AsAbsoluteWindowsPath(" << posix_name << ") failed: " << error;
    return false;
  }
  wstring werror;
  if (CreateJunction(name, target, &werror) != CreateJunctionResult::kSuccess) {
    string error(blaze_util::WstringToCstring(werror.c_str()).get());
    BAZEL_LOG(ERROR) << "SymlinkDirectories(" << posix_target << ", "
                     << posix_name << "): CreateJunction: " << error;
    return false;
  }
  return true;
}


#ifndef STILL_ACTIVE
#define STILL_ACTIVE (259)  // From MSDN about GetExitCodeProcess.
#endif

// On Windows (and Linux) we use a combination of PID and start time to identify
// the server process. That is supposed to be unique unless one can start more
// processes than there are PIDs available within a single jiffy.
bool VerifyServerProcess(int pid, const string& output_base) {
  AutoHandle process(
      ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid));
  if (!process.IsValid()) {
    // Cannot find the server process. Can happen if the PID file is stale.
    return false;
  }

  DWORD exit_code = 0;
  uint64_t start_time = 0;
  if (!::GetExitCodeProcess(process, &exit_code) || exit_code != STILL_ACTIVE ||
      !GetProcessStartupTime(process, &start_time)) {
    // Process doesn't exist or died meantime, all is good. No stale server is
    // present.
    return false;
  }

  string recorded_start_time;
  bool file_present = blaze_util::ReadFile(
      blaze_util::JoinPath(output_base, "server/server.starttime"),
      &recorded_start_time);

  // If start time file got deleted, but PID file didn't, assume that this is an
  // old Bazel process that doesn't know how to write start time files yet.
  return !file_present || recorded_start_time == ToString(start_time);
}

bool KillServerProcess(int pid, const string& output_base) {
  AutoHandle process(::OpenProcess(
      PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid));
  DWORD exitcode = 0;
  if (!process.IsValid() || !::GetExitCodeProcess(process, &exitcode) ||
      exitcode != STILL_ACTIVE) {
    // Cannot find the server process (can happen if the PID file is stale) or
    // it already exited.
    return false;
  }

  BOOL result = TerminateProcess(process, /*uExitCode*/ 0);
  if (!result || !AwaitServerProcessTermination(pid, output_base,
                                                kPostKillGracePeriodSeconds)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Cannot terminate server process with PID " << pid
        << ", output_base=(" << output_base << "): " << GetLastErrorString();
  }
  return result;
}

void TrySleep(unsigned int milliseconds) {
  Sleep(milliseconds);
}

// Not supported.
void ExcludePathFromBackup(const string &path) {
}

string GetHashedBaseDir(const string& root, const string& hashable) {
  // Builds a shorter output base dir name for Windows.

  // We create a path name representing the 128 bits of MD5 digest. To avoid
  // platform incompatibilities we restrict the alphabet to ASCII letters and
  // numbers. Windows paths are case-insensitive, so use only lower-case
  // letters. These constraints yield a 5-bit alphabet.
  // Since we only need 6 digits, ignore 0 and 1 because they look like
  // upper-case "O" and lower-case "l".
  static const char* alphabet = "abcdefghijklmnopqrstuvwxyz234567";

  // 128 bits of data in base-32 require 128/5 = 25 digits with 3 bits lost.
  // Maximum path length on Windows is only 259 characters, so we'll only use
  // a few characters characters (base-32 digits) to represent the digest.
  // Using only 8 characters we represent 40 bits of the original 128.
  // Since the mapping is lossy and collisions are unlikely in practice, we'll
  // keep the mapping simple and just use the lower 5 bits of the first 8 bytes.
  static const unsigned char kLower5BitsMask = 0x1F;
  static const int filename_length = 8;
  unsigned char md5[blaze_util::Md5Digest::kDigestLength];
  char coded_name[filename_length + 1];
  blaze_util::Md5Digest digest;
  digest.Update(hashable.data(), hashable.size());
  digest.Finish(md5);
  for (int i = 0; i < filename_length; ++i) {
    coded_name[i] = alphabet[md5[i] & kLower5BitsMask];
  }
  coded_name[filename_length] = '\0';
  return blaze_util::JoinPath(root, string(coded_name));
}

void CreateSecureOutputRoot(const string& path) {
  // TODO(bazel-team): implement this properly, by mimicing whatever the POSIX
  // implementation does.
  const char* root = path.c_str();
  if (!blaze_util::MakeDirectories(path, 0755)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "MakeDirectories(" << root << ") failed: " << GetLastErrorString();
  }

  if (!blaze_util::IsDirectory(path)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "'" << root << "' is not a directory";
  }

  ExcludePathFromBackup(root);
}

string GetEnv(const string& name) {
  DWORD size = ::GetEnvironmentVariableA(name.c_str(), NULL, 0);
  if (size == 0) {
    return string();  // unset or empty envvar
  }

  unique_ptr<char[]> value(new char[size]);
  ::GetEnvironmentVariableA(name.c_str(), value.get(), size);
  return string(value.get());
}

string GetPathEnv(const string& name) {
  string value = GetEnv(name);
  if (value.empty()) {
    return value;
  }
  if (bazel::windows::HasUncPrefix(value.c_str())) {
    value = value.substr(4);
  }
  string wpath, error;
  if (!blaze_util::AsWindowsPath(value, &wpath, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Invalid path in envvar \"" << name << "\": " << error;
  }
  // Callers of GetPathEnv expect a path with forward slashes.
  std::replace(wpath.begin(), wpath.end(), '\\', '/');
  return wpath;
}

bool ExistsEnv(const string& name) {
  return ::GetEnvironmentVariableA(name.c_str(), NULL, 0) != 0;
}

void SetEnv(const string& name, const string& value) {
  // _putenv_s both calls ::SetEnvionmentVariableA and updates environ(5).
  _putenv_s(name.c_str(), value.c_str());
}

void UnsetEnv(const string& name) { SetEnv(name, ""); }

bool WarnIfStartedFromDesktop() {
  // GetConsoleProcessList returns:
  //   0, if no console attached (Bazel runs as a subprocess)
  //   1, if Bazel was started by clicking on its icon
  //   2, if Bazel was started from the command line (even if its output is
  //      redirected)
  DWORD dummy[2] = {0};
  if (GetConsoleProcessList(dummy, 2) != 1) {
    return false;
  }
  printf(
      "Bazel is a command line tool.\n\n"
      "Try opening a console, such as the Windows Command Prompt (cmd.exe) "
      "or PowerShell, and running \"bazel help\".\n\n"
      "Press Enter to close this window...");
  ReadFile(GetStdHandle(STD_INPUT_HANDLE), dummy, 1, dummy, NULL);
  return true;
}

#ifndef ENABLE_PROCESSED_OUTPUT
// From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
#define ENABLE_PROCESSED_OUTPUT 0x0001
#endif  // not ENABLE_PROCESSED_OUTPUT

#ifndef ENABLE_WRAP_AT_EOL_OUTPUT
// From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
#define ENABLE_WRAP_AT_EOL_OUTPUT 0x0002
#endif  // not ENABLE_WRAP_AT_EOL_OUTPUT

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
// From MSDN about BOOL SetConsoleMode(HANDLE, DWORD).
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif  // not ENABLE_VIRTUAL_TERMINAL_PROCESSING

void SetupStdStreams() {
  static const DWORD stdhandles[] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
                                     STD_ERROR_HANDLE};
  for (int i = 0; i <= 2; ++i) {
    HANDLE handle = ::GetStdHandle(stdhandles[i]);
    if (handle == INVALID_HANDLE_VALUE || handle == NULL) {
      // Ensure we have open fds to each std* stream. Otherwise we can end up
      // with bizarre things like stdout going to the lock file, etc.
      _open("NUL", (i == 0) ? _O_RDONLY : _O_WRONLY);
    }
    DWORD mode = 0;
    if (i > 0 && handle != INVALID_HANDLE_VALUE && handle != NULL &&
        ::GetConsoleMode(handle, &mode)) {
      DWORD newmode = mode | ENABLE_PROCESSED_OUTPUT |
                      ENABLE_WRAP_AT_EOL_OUTPUT |
                      ENABLE_VIRTUAL_TERMINAL_PROCESSING;
      if (mode != newmode) {
        // We don't care about the success of this. Worst that can happen if
        // this method fails is that the console won't understand control
        // characters like color change or carriage return.
        ::SetConsoleMode(handle, newmode);
      }
    }
  }
}

LARGE_INTEGER WindowsClock::GetFrequency() {
  LARGE_INTEGER result;
  if (!QueryPerformanceFrequency(&result)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WindowsClock::GetFrequency: QueryPerformanceFrequency failed: "
        << GetLastErrorString();
  }

  // On ancient Windows versions (pre-XP) and specific hardware the result may
  // be 0. Since this is pre-XP, we don't handle that, just error out.
  if (result.QuadPart <= 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WindowsClock::GetFrequency: QueryPerformanceFrequency returned "
           "invalid result ("
        << result.QuadPart << "): " << GetLastErrorString();
  }

  return result;
}

LARGE_INTEGER WindowsClock::GetMillisecondsAsLargeInt(
    const LARGE_INTEGER& freq) {
  LARGE_INTEGER counter;
  if (!QueryPerformanceCounter(&counter)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "WindowsClock::GetMillisecondsAsLargeInt: QueryPerformanceCounter "
           "failed: "
        << GetLastErrorString();
  }

  LARGE_INTEGER result;
  result.QuadPart =
      // seconds
      (counter.QuadPart / freq.QuadPart) * 1000LL +
      // milliseconds
      (((counter.QuadPart % freq.QuadPart) * 1000LL) / freq.QuadPart);

  return result;
}

const WindowsClock WindowsClock::INSTANCE;

WindowsClock::WindowsClock()
    : kFrequency(GetFrequency()) {}

uint64_t WindowsClock::GetMilliseconds() const {
  return GetMillisecondsAsLargeInt(kFrequency).QuadPart;
}

uint64_t AcquireLock(const string& output_base, bool batch_mode, bool block,
                     BlazeLock* blaze_lock) {
  string lockfile = blaze_util::JoinPath(output_base, "lock");
  wstring wlockfile;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(lockfile, &wlockfile, &error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "AcquireLock(" << output_base << "): AsAbsoluteWindowsPath("
        << lockfile << ") failed: " << error;
  }

  blaze_lock->handle = INVALID_HANDLE_VALUE;
  bool first_lock_attempt = true;
  uint64_t st = GetMillisecondsMonotonic();
  while (true) {
    blaze_lock->handle = ::CreateFileW(
        /* lpFileName */ wlockfile.c_str(),
        /* dwDesiredAccess */ GENERIC_READ | GENERIC_WRITE,
        /* dwShareMode */ FILE_SHARE_READ,
        /* lpSecurityAttributes */ NULL,
        /* dwCreationDisposition */ CREATE_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);
    if (blaze_lock->handle != INVALID_HANDLE_VALUE) {
      // We could open the file, so noone else holds a lock on it.
      break;
    }
    if (GetLastError() == ERROR_SHARING_VIOLATION) {
      // Someone else has the lock.
      BAZEL_LOG(USER) << "Another command holds the client lock";
      if (!block) {
        BAZEL_DIE(blaze_exit_code::LOCK_HELD_NOBLOCK_FOR_LOCK)
            << "Exiting because the lock is held and --noblock_for_lock was "
               "given.";
      }
      if (first_lock_attempt) {
        first_lock_attempt = false;
        BAZEL_LOG(USER) << "Waiting for it to complete...";
        fflush(stderr);
      }
      Sleep(/* dwMilliseconds */ 200);
    } else {
      BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
          << "AcquireLock(" << lockfile << "): CreateFileW("
          << blaze_util::WstringToString(wlockfile)
          << ") failed: " << GetLastErrorString();
    }
  }
  uint64_t wait_time = GetMillisecondsMonotonic() - st;

  // We have the lock.
  OVERLAPPED overlapped = {0};
  if (!LockFileEx(
          /* hFile */ blaze_lock->handle,
          /* dwFlags */ LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
          /* dwReserved */ 0,
          /* nNumberOfBytesToLockLow */ 1,
          /* nNumberOfBytesToLockHigh */ 0,
          /* lpOverlapped */ &overlapped)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "AcquireLock(" << lockfile << "): LockFileEx("
        << blaze_util::WstringToString(wlockfile)
        << ") failed: " << GetLastErrorString();
  }
  // On other platforms we write some info about this process into the lock file
  // such as the server PID. On Windows we don't do that because the file is
  // locked exclusively, meaning other processes may not open the file even for
  // reading.

  return wait_time;
}

void ReleaseLock(BlazeLock* blaze_lock) {
  OVERLAPPED overlapped = {0};
  UnlockFileEx(blaze_lock->handle, 0, 1, 0, &overlapped);
  CloseHandle(blaze_lock->handle);
}

#ifdef GetUserName
// By including <windows.h>, we have GetUserName defined either as
// GetUserNameA or GetUserNameW.
#undef GetUserName
#endif

string GetUserName() {
  WCHAR buffer[UNLEN + 1];
  DWORD len = UNLEN + 1;
  if (!::GetUserNameW(buffer, &len)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "GetUserNameW failed: " << GetLastErrorString();
  }
  return string(blaze_util::WstringToCstring(buffer).get());
}

bool IsEmacsTerminal() {
  string emacs = GetEnv("EMACS");
  // GNU Emacs <25.1 (and ~all non-GNU emacsen) set EMACS=t, but >=25.1 doesn't
  // do that and instead sets INSIDE_EMACS=<stuff> (where <stuff> can look like
  // e.g. "25.1.1,comint").  So we check both variables for maximum
  // compatibility.
  return emacs == "t" || ExistsEnv("INSIDE_EMACS");
}

// Returns true if stderr is connected to a terminal, and it can support color
// and cursor movement (this is computed heuristically based on the values of
// environment variables).  Blaze only outputs control characters to stderr,
// so we only care for the stderr descriptor type.
bool IsStderrStandardTerminal() {
  DWORD mode = 0;
  HANDLE handle = ::GetStdHandle(STD_ERROR_HANDLE);
  // handle may be invalid when stderr is redirected
  if (handle == INVALID_HANDLE_VALUE || !::GetConsoleMode(handle, &mode) ||
      !(mode & ENABLE_PROCESSED_OUTPUT) ||
      !(mode & ENABLE_WRAP_AT_EOL_OUTPUT) ||
      !(mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
    return false;
  }
  return true;
}

// Returns the number of columns of the terminal to which stderr is connected,
// or $COLUMNS (default 80) if there is no such terminal.  Blaze only outputs
// formatted messages to stderr, so we only care for width of a terminal
// connected to the stderr descriptor.
int GetStderrTerminalColumns() {
  string columns_env = GetEnv("COLUMNS");
  if (!columns_env.empty()) {
    char* endptr;
    int columns = blaze_util::strto32(columns_env.c_str(), &endptr, 10);
    if (*endptr == '\0') {  // $COLUMNS is a valid number
      return columns;
    }
  }

  HANDLE stdout_handle = ::GetStdHandle(STD_ERROR_HANDLE);
  if (stdout_handle != INVALID_HANDLE_VALUE) {
    // stdout_handle may be invalid when stdout is redirected.
    CONSOLE_SCREEN_BUFFER_INFO screen_info;
    if (GetConsoleScreenBufferInfo(stdout_handle, &screen_info)) {
      int width = 1 + screen_info.srWindow.Right - screen_info.srWindow.Left;
      if (width > 1) {
        return width;
      }
    }
  }

  return 80;  // default if not a terminal.
}

bool UnlimitResources() {
  return true;  // Nothing to do so assume success.
}

bool UnlimitCoredumps() {
  return true;  // Nothing to do so assume success.
}

static const int MAX_KEY_LENGTH = 255;
// We do not care about registry values longer than MAX_PATH
static const int REG_VALUE_BUFFER_SIZE = MAX_PATH;

// Implements heuristics to discover msys2 installation.
static string GetMsysBash() {
  HKEY h_uninstall;

  // MSYS2 installer writes its registry into HKCU, although documentation
  // (https://msdn.microsoft.com/en-us/library/ms954376.aspx)
  // clearly states that it should go to HKLM.
  static constexpr const char key[] =
      "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall";
  if (RegOpenKeyExA(HKEY_CURRENT_USER,  // _In_     HKEY    hKey,
                    key,                // _In_opt_ LPCTSTR lpSubKey,
                    0,                  // _In_     DWORD   ulOptions,
                    KEY_ENUMERATE_SUB_KEYS |
                        KEY_QUERY_VALUE,  // _In_     REGSAM  samDesired,
                    &h_uninstall          // _Out_    PHKEY   phkResult
                    )) {
    BAZEL_LOG(INFO) << "Cannot open HKCU\\" << key;
    return string();
  }
  AutoHandle auto_uninstall(h_uninstall);

  // Since MSYS2 decided to generate a new product key for each installation,
  // we enumerate all keys under
  // HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall and find the first
  // with MSYS2 64bit display name.
  static constexpr const char msys_display_name[] = "MSYS2 64bit";
  DWORD n_subkeys;

  if (RegQueryInfoKey(h_uninstall,  // _In_        HKEY      hKey,
                      0,            // _Out_opt_   LPTSTR    lpClass,
                      0,            // _Inout_opt_ LPDWORD   lpcClass,
                      0,            // _Reserved_  LPDWORD   lpReserved,
                      &n_subkeys,   // _Out_opt_   LPDWORD   lpcSubKeys,
                      0,            // _Out_opt_   LPDWORD   lpcMaxSubKeyLen,
                      0,            // _Out_opt_   LPDWORD   lpcMaxClassLen,
                      0,            // _Out_opt_   LPDWORD   lpcValues,
                      0,            // _Out_opt_   LPDWORD   lpcMaxValueNameLen,
                      0,            // _Out_opt_   LPDWORD   lpcMaxValueLen,
                      0,  // _Out_opt_   LPDWORD   lpcbSecurityDescriptor,
                      0   // _Out_opt_   PFILETIME lpftLastWriteTime
                      )) {
    BAZEL_LOG(INFO) << "Cannot query HKCU\\" << key;
    return string();
  }

  for (DWORD key_index = 0; key_index < n_subkeys; key_index++) {
    char subkey_name[MAX_KEY_LENGTH];
    if (RegEnumKeyA(h_uninstall,         // _In_  HKEY   hKey,
                    key_index,           // _In_  DWORD  dwIndex,
                    subkey_name,         // _Out_ LPTSTR lpName,
                    sizeof(subkey_name)  // _In_  DWORD  cchName
                    )) {
      BAZEL_LOG(INFO) << "Cannot get " << key_index << " subkey of HKCU\\"
                      << key;
      continue;  // try next subkey
    }

    HKEY h_subkey;
    if (RegOpenKeyEx(h_uninstall,      // _In_     HKEY    hKey,
                     subkey_name,      // _In_opt_ LPCTSTR lpSubKey,
                     0,                // _In_     DWORD   ulOptions,
                     KEY_QUERY_VALUE,  // _In_     REGSAM  samDesired,
                     &h_subkey         // _Out_    PHKEY   phkResult
                     )) {
      BAZEL_LOG(ERROR) << "Failed to open subkey HKCU\\" << key << "\\"
                       << subkey_name;
      continue;  // try next subkey
    }
    AutoHandle auto_subkey(h_subkey);

    BYTE value[REG_VALUE_BUFFER_SIZE];
    DWORD value_length = sizeof(value);
    DWORD value_type;

    if (RegQueryValueEx(h_subkey,       // _In_        HKEY    hKey,
                        "DisplayName",  // _In_opt_    LPCTSTR lpValueName,
                        0,              // _Reserved_  LPDWORD lpReserved,
                        &value_type,    // _Out_opt_   LPDWORD lpType,
                        value,          // _Out_opt_   LPBYTE  lpData,
                        &value_length   // _Inout_opt_ LPDWORD lpcbData
                        )) {
      // This registry key has no DisplayName subkey, so it cannot be MSYS2, or
      // it cannot be a version of MSYS2 that we are looking for.
      continue;
    }

    if (value_type == REG_SZ &&
        0 == memcmp(msys_display_name, value, sizeof(msys_display_name))) {
      BAZEL_LOG(INFO) << "Getting install location of HKCU\\" << key << "\\"
                      << subkey_name;
      BYTE path[REG_VALUE_BUFFER_SIZE];
      DWORD path_length = sizeof(path);
      DWORD path_type;
      if (RegQueryValueEx(
              h_subkey,           // _In_        HKEY    hKey,
              "InstallLocation",  // _In_opt_    LPCTSTR lpValueName,
              0,                  // _Reserved_  LPDWORD lpReserved,
              &path_type,         // _Out_opt_   LPDWORD lpType,
              path,               // _Out_opt_   LPBYTE  lpData,
              &path_length        // _Inout_opt_ LPDWORD lpcbData
              )) {
        // This version of MSYS2 does not seem to create a "InstallLocation"
        // subkey. Let's ignore this registry key to avoid picking up an MSYS2
        // version that may be different from what Bazel expects.
        continue;  // try next subkey
      }

      if (path_length == 0 || path_type != REG_SZ) {
        // This version of MSYS2 seem to have recorded an empty installation
        // location, or the registry key was modified. Either way, let's ignore
        // this registry key and keep looking at the next subkey.
        continue;
      }

      BAZEL_LOG(INFO) << "Install location of HKCU\\" << key << "\\"
                      << subkey_name << " is " << path;
      string path_as_string(path, path + path_length - 1);
      string bash_exe = path_as_string + "\\usr\\bin\\bash.exe";
      if (!blaze_util::PathExists(bash_exe)) {
        // The supposed bash.exe does not exist. Maybe MSYS2 was deleted but not
        // uninstalled? We can't tell, but for all we care, this MSYS2 path is
        // not what we need, so ignore this registry key.
        continue;
      }

      BAZEL_LOG(INFO) << "Detected MSYS2 Bash at " << bash_exe.c_str();
      return bash_exe;
    }
  }
  return string();
}

static string GetBinaryFromPath(const string& binary_name) {
  char found[MAX_PATH];
  string path_list = blaze::GetPathEnv("PATH");

  // We do not fully replicate all the quirks of search in PATH.
  // There is no system function to do so, and that way lies madness.
  size_t start = 0;
  do {
    // This ignores possibly quoted semicolons in PATH etc.
    size_t end = path_list.find_first_of(";", start);
    string path = path_list.substr(
        start, end != string::npos ? end - start : string::npos);
    // Handle one typical way of quoting (where.exe does not handle this, but
    // CreateProcess does).
    if (path.size() > 1 && path[0] == '"' && path[path.size() - 1] == '"') {
      path = path.substr(1, path.size() - 2);
    }
    if (SearchPathA(path.c_str(),         // _In_opt_  LPCTSTR lpPath,
                    binary_name.c_str(),  // _In_      LPCTSTR lpFileName,
                    0,                    // LPCTSTR lpExtension,
                    sizeof(found),        // DWORD   nBufferLength,
                    found,                // _Out_     LPTSTR  lpBuffer,
                    0                     // _Out_opt_ LPTSTR  *lpFilePart
                    )) {
      BAZEL_LOG(INFO) << binary_name.c_str() << " found on PATH: " << found;
      return string(found);
    }
    if (end == string::npos) {
      break;
    }
    start = end + 1;
  } while (true);

  return string();
}

static string LocateBashMaybe() {
  string msys_bash = GetMsysBash();
  return msys_bash.empty() ? GetBinaryFromPath("bash.exe") : msys_bash;
}

string DetectBashAndExportBazelSh() {
  string bash = blaze::GetPathEnv("BAZEL_SH");
  if (!bash.empty()) {
    return bash;
  }

  uint64_t start = blaze::GetMillisecondsMonotonic();

  bash = LocateBashMaybe();
  uint64_t end = blaze::GetMillisecondsMonotonic();
  if (bash.empty()) {
    BAZEL_LOG(INFO) << "BAZEL_SH detection took " << end - start
                    << " msec, not found";
  } else {
    BAZEL_LOG(INFO) << "BAZEL_SH detection took " << end - start
                    << " msec, found " << bash.c_str();
    // Set process environment variable.
    blaze::SetEnv("BAZEL_SH", bash);
  }

  return bash;
}

void EnsurePythonPathOption(std::vector<string>* options) {
  string python_path = GetBinaryFromPath("python.exe");
  if (!python_path.empty()) {
    // Provide python path as coming from the least important rc file.
    std::replace(python_path.begin(), python_path.end(), '\\', '/');
    options->push_back(string("--default_override=0:build=--python_path=") +
                       python_path);
  }
}

}  // namespace blaze
