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

#include <VersionHelpers.h>

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

bool WaitableProcess::Create(const std::wstring& argv0,
                             const std::wstring& argv_rest, void* env,
                             const std::wstring& wcwd, HANDLE stdin_process,
                             HANDLE stdout_process, HANDLE stderr_process,
                             AutoHandle* out_job, AutoHandle* out_ioport,
                             AutoHandle* out_process, DWORD* out_pid,
                             std::wstring* error) {
  std::wstring cwd;
  std::wstring error_msg(AsShortPath(wcwd, &cwd));
  if (!error_msg.empty()) {
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, error_msg);
    return false;
  }

  std::wstring argv0short;
  error_msg = AsExecutablePathForCreateProcess(argv0, &argv0short);
  if (!error_msg.empty()) {
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, error_msg);
    return false;
  }

  std::wstring commandline =
      argv_rest.empty() ? argv0short : (argv0short + L" " + argv_rest);
  std::unique_ptr<WCHAR[]> mutable_commandline(
      new WCHAR[commandline.size() + 1]);
  wcsncpy(mutable_commandline.get(), commandline.c_str(),
          commandline.size() + 1);
  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  *out_job = CreateJobObject(NULL, NULL);
  if (!out_job->IsValid()) {
    DWORD err_code = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, err_code);
    return false;
  }

  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};
  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(*out_job, JobObjectExtendedLimitInformation,
                               &job_info, sizeof(job_info))) {
    DWORD err_code = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, err_code);
    return false;
  }

  *out_ioport = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
  if (!out_ioport->IsValid()) {
    DWORD err_code = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, err_code);
    return false;
  }
  JOBOBJECT_ASSOCIATE_COMPLETION_PORT port;
  port.CompletionKey = *out_job;
  port.CompletionPort = *out_ioport;
  if (!SetInformationJobObject(*out_job,
                               JobObjectAssociateCompletionPortInformation,
                               &port, sizeof(port))) {
    DWORD err_code = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, err_code);
    return false;
  }

  std::unique_ptr<AutoAttributeList> attr_list;
  if (!AutoAttributeList::Create(stdin_process, stdout_process, stderr_process,
                                 &attr_list, &error_msg)) {
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", L"", error_msg);
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
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"CreateProcessWithExplicitHandles", cmd_sample,
                              error_msg.str().c_str());
    return false;
  }

  PROCESS_INFORMATION process_info = {0};
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
              | CREATE_NEW_PROCESS_GROUP  // So that Ctrl-Break isn't propagated
              | CREATE_SUSPENDED  // So that it doesn't start a new job itself
              | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
          /* lpEnvironment */ env,
          /* lpCurrentDirectory */ cwd.empty() ? NULL : cwd.c_str(),
          /* lpStartupInfo */ &info.StartupInfo,
          /* lpProcessInformation */ &process_info)) {
    DWORD err = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__, L"CreateProcessW",
                              cmd_sample, err);
    return false;
  }

  *out_pid = process_info.dwProcessId;
  *out_process = process_info.hProcess;
  AutoHandle thread(process_info.hThread);

  if (!AssignProcessToJobObject(*out_job, *out_process)) {
    BOOL is_in_job = false;
    if (IsProcessInJob(*out_process, NULL, &is_in_job) && is_in_job &&
        !IsWindows8OrGreater()) {
      // Pre-Windows 8 systems don't support nested jobs, and Bazel is already
      // in a job.  We can't create nested jobs, so just revert to
      // TerminateProcess() and hope for the best. In batch mode, the launcher
      // puts Bazel in a job so that will take care of cleanup once the
      // command finishes.
      *out_job = INVALID_HANDLE_VALUE;
      *out_ioport = INVALID_HANDLE_VALUE;
    } else {
      DWORD err_code = GetLastError();
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"WaitableProcess::Create", argv0, err_code);
      return false;
    }
  }

  // Now that we put the process in a new job object, we can start executing
  // it
  if (ResumeThread(thread) == -1) {
    DWORD err_code = GetLastError();
    *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                              L"WaitableProcess::Create", argv0, err_code);
    return false;
  }

  *error = L"";
  return true;
}

int WaitableProcess::WaitFor(int64_t timeout_msec, DWORD pid,
                             AutoHandle* in_out_job, AutoHandle* in_out_ioport,
                             AutoHandle* in_out_process, DWORD* out_exit_code,
                             std::wstring* error) {
  DWORD win32_timeout = timeout_msec < 0 ? INFINITE : timeout_msec;
  int result;
  switch (WaitForSingleObject(*in_out_process, win32_timeout)) {
    case WAIT_OBJECT_0:
      result = kWaitSuccess;
      break;

    case WAIT_TIMEOUT:
      result = kWaitTimeout;
      break;

    // Any other case is an error and should be reported back to Bazel.
    default:
      DWORD err_code = GetLastError();
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"WaitableProcess::WaitFor", ToString(pid),
                                err_code);
      return kWaitError;
  }

  // Ensure that the process is really terminated (if WaitForSingleObject
  // above timed out, we have to explicitly kill it) and that it doesn't
  // leave behind any subprocesses.
  if (!Terminate(*in_out_job, *in_out_process, pid, out_exit_code, error)) {
    return kWaitError;
  }

  if (in_out_job->IsValid()) {
    // Wait for the job object to complete, signalling that all subprocesses
    // have exited.
    DWORD CompletionCode;
    ULONG_PTR CompletionKey;
    LPOVERLAPPED Overlapped;
    while (GetQueuedCompletionStatus(*in_out_ioport, &CompletionCode,
                                     &CompletionKey, &Overlapped, INFINITE) &&
           !((HANDLE)CompletionKey == (HANDLE)*in_out_job &&
             CompletionCode == JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO)) {
      // Still waiting...
    }

    *in_out_job = INVALID_HANDLE_VALUE;
    *in_out_ioport = INVALID_HANDLE_VALUE;
  }

  // Fetch and store the exit code in case Bazel asks us for it later,
  // because we cannot do this anymore after we closed the handle.
  GetExitCode(*in_out_process, pid, out_exit_code, error);

  if (in_out_process->IsValid()) {
    *in_out_process = INVALID_HANDLE_VALUE;
  }

  return result;
}

int WaitableProcess::GetExitCode(const AutoHandle& process, DWORD pid,
                                 DWORD* out_exit_code, std::wstring* error) {
  if (*out_exit_code == STILL_ACTIVE) {
    if (!GetExitCodeProcess(process, out_exit_code)) {
      DWORD err_code = GetLastError();
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"WaitableProcess::GetExitCode", ToString(pid),
                                err_code);
      return -1;
    }
  }

  return *out_exit_code;
}

bool WaitableProcess::Terminate(const AutoHandle& job,
                                const AutoHandle& process, DWORD pid,
                                DWORD* out_exit_code, std::wstring* error) {
  static constexpr UINT exit_code = 130;  // 128 + SIGINT, like on Linux

  if (job.IsValid()) {
    if (!TerminateJobObject(job, exit_code)) {
      DWORD err_code = GetLastError();
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"WaitableProcess::Terminate", ToString(pid),
                                err_code);
      return false;
    }
  } else if (process.IsValid()) {
    if (!TerminateProcess(process, exit_code)) {
      DWORD err_code = GetLastError();
      std::wstring our_error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"WaitableProcess::Terminate",
                                                ToString(pid), err_code);

      // If the process exited, despite TerminateProcess having failed, we're
      // still happy and just ignore the error. It might have been a race
      // where the process exited by itself just before we tried to kill it.
      // However, if the process is *still* running at this point (evidenced
      // by its exit code still being STILL_ACTIVE) then something went
      // really unexpectedly wrong and we should report that error.
      if (GetExitCode(process, pid, out_exit_code, error) == STILL_ACTIVE) {
        // Restore the error message from TerminateProcess - it will be much
        // more helpful for debugging in case something goes wrong here.
        *error = our_error;
        return false;
      }
    }

    if (WaitForSingleObject(process, INFINITE) != WAIT_OBJECT_0) {
      DWORD err_code = GetLastError();
      *error = MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                L"WaitableProcess::Terminate", ToString(pid),
                                err_code);
      return false;
    }
  }

  *error = L"";
  return true;
}

}  // namespace windows
}  // namespace bazel
