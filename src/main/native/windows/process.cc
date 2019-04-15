// Copyright 2019 The Bazel Authors. All rights reserved.
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

#include "src/main/native/windows/process.h"

#include <memory>
#include <sstream>

namespace bazel {
namespace windows {

template <typename T>
static std::wstring ToString(const T& e) {
  std::wstringstream s;
  s << e;
  return s.str();
}

static bool NestedJobsSupported() {
  OSVERSIONINFOEX version_info;
  version_info.dwOSVersionInfoSize = sizeof(version_info);
  if (!GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&version_info))) {
    return false;
  }

  return version_info.dwMajorVersion > 6 ||
         (version_info.dwMajorVersion == 6 &&
          version_info.dwMinorVersion >= 2);
}

WaitableProcess::WaitableProcess() : exit_code_(STILL_ACTIVE) { }

bool WaitableProcess::Create(
    const std::wstring& wpath, const std::wstring& argv_rest,
    const std::wstring& wcwd, void* env, HANDLE child_stdin,
    HANDLE child_stdout, HANDLE child_stderr, std::wstring* error) {
  std::wstring argv0;
  std::wstring error_msg(
      bazel::windows::AsExecutablePathForCreateProcess(wpath, &argv0));
  if (!error_msg.empty()) {
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, error_msg);
    return false;
  }

  std::wstring commandline = argv0 + L" " + argv_rest;
  std::wstring cwd;
  error_msg = bazel::windows::AsShortPath(wcwd, &cwd);
  if (!error_msg.empty()) {
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, error_msg);
    return false;
  }

  std::unique_ptr<WCHAR[]> mutable_commandline(
      new WCHAR[commandline.size() + 1]);
  wcsncpy(mutable_commandline.get(), commandline.c_str(),
          commandline.size() + 1);

  // Standard file handles are closed even if the process was successfully
  // created. If this was not so, operations on these file handles would not
  // return immediately if the process is terminated.
  // Therefore we make these handles auto-closing (by using AutoHandle).
  bazel::windows::AutoHandle thread;
  PROCESS_INFORMATION process_info = {0};
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};

  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  job_ = CreateJobObject(NULL, NULL);
  if (!job_.IsValid()) {
    DWORD err_code = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return false;
  }

  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job_, JobObjectExtendedLimitInformation,
                               &job_info, sizeof(job_info))) {
    DWORD err_code = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return false;
  }

  ioport_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
  if (!ioport_.IsValid()) {
    DWORD err_code = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return false;
  }
  JOBOBJECT_ASSOCIATE_COMPLETION_PORT port;
  port.CompletionKey = job_;
  port.CompletionPort = ioport_;
  if (!SetInformationJobObject(job_,
                               JobObjectAssociateCompletionPortInformation,
                               &port, sizeof(port))) {
    DWORD err_code = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return false;
  }

  std::unique_ptr<bazel::windows::AutoAttributeList> attr_list;
  if (!bazel::windows::AutoAttributeList::Create(
          child_stdin, child_stdout, child_stderr, &attr_list,
          &error_msg)) {
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", L"", error_msg);
    return false;
  }

  // kMaxCmdline value: see lpCommandLine parameter of CreateProcessW.
  static constexpr size_t kMaxCmdline = 32767;

  std::wstring cmd_sample = mutable_commandline.get();
  if (cmd_sample.size() > 200) {
    cmd_sample = cmd_sample.substr(0, 195) + L"(...)";
  }
  if (wcsnlen_s(mutable_commandline.get(), kMaxCmdline) == kMaxCmdline) {
    std::wstringstream error_msg;
    error_msg << L"command is longer than CreateProcessW's limit ("
              << kMaxCmdline << L" characters)";
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"CreateProcessWithExplicitHandles",
        cmd_sample, error_msg.str().c_str());
    return false;
  }

  STARTUPINFOEXW info;
  attr_list->InitStartupInfoExW(&info);
  if (!CreateProcessW(
          /* lpApplicationName */ NULL,
          /* lpCommandLine */ mutable_commandline.get(),
          /* lpProcessAttributes */ NULL,
          /* lpThreadAttributes */ NULL,
          /* bInheritHandles */ TRUE,
          /* dwCreationFlags */ CREATE_NO_WINDOW  // Don't create console
                                                  // window
              |
              CREATE_NEW_PROCESS_GROUP  // So that Ctrl-Break isn't propagated
              | CREATE_SUSPENDED  // So that it doesn't start a new job itself
              | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
          /* lpEnvironment */ env,
          /* lpCurrentDirectory */ cwd.empty() ? NULL : cwd.c_str(),
          /* lpStartupInfo */ &info.StartupInfo,
          /* lpProcessInformation */ &process_info)) {
    DWORD err = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"CreateProcessW", cmd_sample, err);
    return false;
  }

  pid_ = process_info.dwProcessId;
  process_ = process_info.hProcess;
  thread = process_info.hThread;

  if (!AssignProcessToJobObject(job_, process_)) {
    BOOL is_in_job = false;
    if (IsProcessInJob(process_, NULL, &is_in_job) && is_in_job &&
        !NestedJobsSupported()) {
      // We are on a pre-Windows 8 system and the Bazel is already in a job.
      // We can't create nested jobs, so just revert to TerminateProcess() and
      // hope for the best. In batch mode, the launcher puts Bazel in a job so
      // that will take care of cleanup once the command finishes.
      job_ = INVALID_HANDLE_VALUE;
      ioport_ = INVALID_HANDLE_VALUE;
    } else {
      DWORD err_code = GetLastError();
      *error = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return false;
    }
  }

  // Now that we put the process in a new job object, we can start executing
  // it
  if (ResumeThread(thread) == -1) {
    DWORD err_code = GetLastError();
    *error = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return false;
  }

  *error = L"";
  return true;
}

// These are the possible return values from the NativeProcess::WaitFor()
// method.
static const int kWaitSuccess = 0;
static const int kWaitTimeout = 1;
static const int kWaitError = 2;

int WaitableProcess::Wait(long timeout_msec, std::wstring* error) {
  DWORD win32_timeout = timeout_msec < 0 ? INFINITE : timeout_msec;
  int result;
  switch (WaitForSingleObject(process_, win32_timeout)) {
    case WAIT_OBJECT_0:
      result = kWaitSuccess;
      break;

    case WAIT_TIMEOUT:
      result = kWaitTimeout;
      break;

    // Any other case is an error and should be reported back to Bazel.
    default:
      DWORD err_code = GetLastError();
      *error = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"NativeProcess:WaitFor",
                                                ToString(pid_), err_code);
      return kWaitError;
  }

//  if (stdin_ != INVALID_HANDLE_VALUE) {
//    CloseHandle(stdin_);
//    stdin_ = INVALID_HANDLE_VALUE;
//  }

  // Ensure that the process is really terminated (if WaitForSingleObject
  // above timed out, we have to explicitly kill it) and that it doesn't
  // leave behind any subprocesses.
  if (!Terminate(error)) {
    return kWaitError;
  }

  if (job_.IsValid()) {
    // Wait for the job object to complete, signalling that all subprocesses
    // have exited.
    DWORD CompletionCode;
    ULONG_PTR CompletionKey;
    LPOVERLAPPED Overlapped;
    while (GetQueuedCompletionStatus(ioport_, &CompletionCode, &CompletionKey,
                                     &Overlapped, INFINITE) &&
           !((HANDLE)CompletionKey == job_ &&
             CompletionCode == JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO)) {
      // Still waiting...
    }

    job_ = INVALID_HANDLE_VALUE;
    ioport_ = INVALID_HANDLE_VALUE;
  }

  // Fetch and store the exit code in case Bazel asks us for it later,
  // because we cannot do this anymore after we closed the handle.
  GetExitCode(error);

  if (process_.IsValid()) {
    process_ = INVALID_HANDLE_VALUE;
  }

  return result;
}

bool WaitableProcess::Terminate(std::wstring* error) {
  static const UINT exit_code = 130;  // 128 + SIGINT, like on Linux

  if (job_.IsValid()) {
    if (!TerminateJobObject(job_, exit_code)) {
      DWORD err_code = GetLastError();
      *error = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"NativeProcess::Terminate",
                                                ToString(pid_), err_code);
      return false;
    }
  } else if (process_.IsValid()) {
    if (!TerminateProcess(process_, exit_code)) {
      DWORD err_code = GetLastError();
      std::wstring our_error = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"NativeProcess::Terminate",
          ToString(pid_), err_code);

      // If the process exited, despite TerminateProcess having failed, we're
      // still happy and just ignore the error. It might have been a race
      // where the process exited by itself just before we tried to kill it.
      // However, if the process is *still* running at this point (evidenced
      // by its exit code still being STILL_ACTIVE) then something went
      // really unexpectedly wrong and we should report that error.
      if (GetExitCode(error) == STILL_ACTIVE) {
        // Restore the error message from TerminateProcess - it will be much
        // more helpful for debugging in case something goes wrong here.
        *error = our_error;
        return false;
      }
    }

    if (WaitForSingleObject(process_, INFINITE) != WAIT_OBJECT_0) {
      DWORD err_code = GetLastError();
      *error = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"NativeProcess::Terminate",
                                                ToString(pid_), err_code);
      return false;
    }
  }

  *error = L"";
  return true;
}

int WaitableProcess::GetExitCode(std::wstring* error) {
  if (exit_code_ == STILL_ACTIVE) {
    if (!GetExitCodeProcess(process_, &exit_code_)) {
      DWORD err_code = GetLastError();
      *error = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"NativeProcess::GetExitCode",
                                                ToString(pid_), err_code);
      return -1;
    }
  }

  return exit_code_;
}

}  // namespace windows
}  // namespace bazel
