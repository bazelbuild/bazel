// Copyright 2016 The Bazel Authors. All rights reserved.
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

#define WIN32_LEAN_AND_MEAN

#include <wchar.h>
#include <windows.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>  // static_assert

#include "src/main/native/jni.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/jni-util.h"
#include "src/main/native/windows/util.h"

template <typename T>
static std::wstring ToString(const T& e) {
  std::wstringstream s;
  s << e;
  return s.str();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetpid(
    JNIEnv* env, jclass clazz) {
  return GetCurrentProcessId();
}

struct NativeOutputStream {
  HANDLE handle_;
  std::wstring error_;
  std::atomic<bool> closed_;
  NativeOutputStream()
      : handle_(INVALID_HANDLE_VALUE), error_(L""), closed_(false) {}

  void close() {
    closed_.store(true);
    if (handle_ == INVALID_HANDLE_VALUE) {
      return;
    }

    // CancelIoEx only cancels I/O operations in the current process.
    // https://msdn.microsoft.com/en-us/library/windows/desktop/aa363792(v=vs.85).aspx
    //
    // Therefore if this process bequested `handle_` to a child process, we
    // cannot cancel I/O in the child process.
    CancelIoEx(handle_, NULL);
    CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
  }

  // Return the last error as a human-readable string and clear it.
  jstring getLastErrorAsString(JNIEnv* env) {
    jstring result = env->NewString(reinterpret_cast<const jchar*>(
        error_.c_str()), error_.size());
    error_ = L"";
    return result;
  }
};

class JavaByteArray {
 public:
  JavaByteArray(JNIEnv* env, jbyteArray java_array)
      : env_(env),
        array_(java_array),
        size_(java_array != nullptr ? env->GetArrayLength(java_array) : 0),
        ptr_(java_array != nullptr ? env->GetByteArrayElements(java_array, NULL)
                                   : nullptr) {}

  ~JavaByteArray() {
    if (array_ != nullptr) {
      env_->ReleaseByteArrayElements(array_, ptr_, 0);
      array_ = nullptr;
      size_ = 0;
      ptr_ = nullptr;
    }
  }

  jsize size() { return size_; }
  jbyte* ptr() { return ptr_; }

 private:
  JNIEnv* env_;
  jbyteArray array_;
  jsize size_;
  jbyte* ptr_;
};

class NativeProcess {
public:
  NativeProcess()
      : stdin_(INVALID_HANDLE_VALUE),
        stdout_(),
        stderr_(),
        process_(INVALID_HANDLE_VALUE),
        job_(INVALID_HANDLE_VALUE),
        error_(L"") {}

  ~NativeProcess() {
    if (stdin_ != INVALID_HANDLE_VALUE) {
      CloseHandle(stdin_);
    }

    stdout_.close();
    stderr_.close();

    if (process_ != INVALID_HANDLE_VALUE) {
      CloseHandle(process_);
    }

    if (job_ != INVALID_HANDLE_VALUE) {
      CloseHandle(job_);
    }
  }

  jint WaitFor(jlong timeout_msec) {
    HANDLE handles[1] = {process_};
    DWORD win32_timeout = timeout_msec < 0 ? INFINITE : timeout_msec;
    jint result;
    switch (WaitForMultipleObjects(1, handles, FALSE, win32_timeout)) {
      case 0:
        result = 0;
        break;

      case WAIT_TIMEOUT:
        result = 1;
        break;

      case WAIT_FAILED:
        result = 2;
        break;

      default:
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeWaitFor", ToString(pid_),
            err_code);
        break;
    }

    if (stdin_ != INVALID_HANDLE_VALUE) {
      CloseHandle(stdin_);
      stdin_ = INVALID_HANDLE_VALUE;
    }

    return result;
  }

  jint GetPid() {
    // MSDN says that GetProcessId cannot fail.
    error_ = L"";
    return GetProcessId(process_);
  }

  // Terminates this process (and subprocesses, if job objects are available).
  jboolean Terminate() {
    static const UINT exit_code = 130;  // 128 + SIGINT, like on Linux

    if (job_ != INVALID_HANDLE_VALUE) {
      if (!TerminateJobObject(job_, exit_code)) {
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"NativeProcess::Terminate",
            ToString(pid_), GetLastError());
        return JNI_FALSE;
      }
    } else if (process_ != INVALID_HANDLE_VALUE) {
      if (!TerminateProcess(process_, exit_code)) {
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"NativeProcess::Terminate",
            ToString(pid_), GetLastError());
        return JNI_FALSE;
      }
    }

    error_ = L"";
    return JNI_TRUE;
  }

  // Return the last error as a human-readable string and clear it.
  jstring getLastErrorAsString(JNIEnv *env) {
    jstring result = env->NewString(reinterpret_cast<const jchar*>(
        error_.c_str()), error_.size());
    error_ = L"";
    return result;
  }

  HANDLE stdin_;
  NativeOutputStream stdout_;
  NativeOutputStream stderr_;
  HANDLE process_;
  HANDLE job_;
  DWORD pid_;
  std::wstring error_;
};

static bool NestedJobsSupported() {
  OSVERSIONINFOEX version_info;
  version_info.dwOSVersionInfoSize = sizeof(version_info);
  if (!GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&version_info))) {
    return false;
  }

  return version_info.dwMajorVersion > 6 ||
         (version_info.dwMajorVersion == 6 && version_info.dwMinorVersion >= 2);
}

// Ensure we can safely cast jlong to void*.
static_assert(sizeof(jlong) == sizeof(void*),
              "jlong and void* should be the same size");

static_assert(sizeof(jchar) == sizeof(WCHAR),
              "jchar and WCHAR should be the same size");

static jlong PtrAsJlong(void* p) { return reinterpret_cast<jlong>(p); }

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeCreateProcess(
    JNIEnv* env, jclass clazz, jstring java_argv0, jstring java_argv_rest,
    jbyteArray java_env, jstring java_cwd, jstring java_stdout_redirect,
    jstring java_stderr_redirect, jboolean redirectErrorStream) {
  NativeProcess* result = new NativeProcess();

  std::wstring argv0;
  std::wstring wpath(bazel::windows::GetJavaWpath(env, java_argv0));
  std::wstring error_msg(
      bazel::windows::AsExecutablePathForCreateProcess(wpath, &argv0));
  if (!error_msg.empty()) {
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, error_msg);
    return PtrAsJlong(result);
  }

  std::wstring commandline =
      argv0 + L" " + bazel::windows::GetJavaWstring(env, java_argv_rest);
  std::wstring stdout_redirect = bazel::windows::AddUncPrefixMaybe(
      bazel::windows::GetJavaWpath(env, java_stdout_redirect));
  std::wstring stderr_redirect = bazel::windows::AddUncPrefixMaybe(
      bazel::windows::GetJavaWpath(env, java_stderr_redirect));
  std::wstring cwd;
  std::wstring wcwd(bazel::windows::GetJavaWpath(env, java_cwd));
  error_msg = bazel::windows::AsShortPath(wcwd, &cwd);
  if (!error_msg.empty()) {
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, error_msg);
    return PtrAsJlong(result);
  }

  const bool stdout_is_stream = stdout_redirect.empty();
  const bool stderr_is_stream =
      redirectErrorStream ? stdout_is_stream : stderr_redirect.empty();
  const bool stderr_same_handle_as_stdout =
      redirectErrorStream ||
      (!stderr_redirect.empty() &&
       stderr_redirect.size() == stdout_redirect.size() &&
       _wcsnicmp(stderr_redirect.c_str(), stdout_redirect.c_str(),
                 stderr_redirect.size()) == 0);

  std::unique_ptr<WCHAR[]> mutable_commandline(
      new WCHAR[commandline.size() + 1]);
  wcsncpy(mutable_commandline.get(), commandline.c_str(),
          commandline.size() + 1);

  SECURITY_ATTRIBUTES sa = {0};
  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;

  // Standard file handles are closed even if the process was successfully
  // created. If this was not so, operations on these file handles would not
  // return immediately if the process is terminated.
  // Therefore we make these handles auto-closing (by using AutoHandle).
  bazel::windows::AutoHandle stdin_process;
  bazel::windows::AutoHandle stdout_process;
  bazel::windows::AutoHandle stderr_process;
  bazel::windows::AutoHandle thread;
  PROCESS_INFORMATION process_info = {0};
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};

  JavaByteArray env_map(env, java_env);
  if (env_map.ptr() != nullptr) {
    if (env_map.size() < 4) {
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath,
          std::wstring(L"the environment must be at least 4 bytes long, was ") +
              ToString(env_map.size()) + L" bytes");
      return PtrAsJlong(result);
    } else if (env_map.ptr()[env_map.size() - 1] != 0 ||
               env_map.ptr()[env_map.size() - 2] != 0 ||
               env_map.ptr()[env_map.size() - 3] != 0 ||
               env_map.ptr()[env_map.size() - 4] != 0) {
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath,
          L"environment array must end with 4 null bytes");
      return PtrAsJlong(result);
    }
  }

  {
    HANDLE pipe_read_h, pipe_write_h;
    if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return PtrAsJlong(result);
    }
    stdin_process = pipe_read_h;
    result->stdin_ = pipe_write_h;
  }

  if (!stdout_is_stream) {
    result->stdout_.close();

    stdout_process = CreateFileW(
        /* lpFileName */ stdout_redirect.c_str(),
        /* dwDesiredAccess */ GENERIC_WRITE,
        /* dwShareMode */ 0,
        /* lpSecurityAttributes */ &sa,
        /* dwCreationDisposition */ OPEN_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);

    if (!stdout_process.IsValid()) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", stdout_redirect,
          err_code);
      return PtrAsJlong(result);
    }
    if (!SetFilePointerEx(stdout_process, {0}, NULL, FILE_END)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", stdout_redirect,
          err_code);
      return PtrAsJlong(result);
    }
  } else {
    HANDLE pipe_read_h, pipe_write_h;
    if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return PtrAsJlong(result);
    }
    result->stdout_.handle_ = pipe_read_h;
    stdout_process = pipe_write_h;
  }

  if (stderr_same_handle_as_stdout) {
    HANDLE stdout_process_dup_h;
    if (!DuplicateHandle(GetCurrentProcess(), stdout_process,
                         GetCurrentProcess(), &stdout_process_dup_h, 0, TRUE,
                         DUPLICATE_SAME_ACCESS)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return PtrAsJlong(result);
    }
    if (!stderr_is_stream) {
      result->stderr_.close();
    }

    stderr_process = stdout_process_dup_h;
  } else if (!stderr_redirect.empty()) {
    result->stderr_.close();
    stderr_process = CreateFileW(
        /* lpFileName */ stderr_redirect.c_str(),
        /* dwDesiredAccess */ GENERIC_WRITE,
        /* dwShareMode */ 0,
        /* lpSecurityAttributes */ &sa,
        /* dwCreationDisposition */ OPEN_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);

    if (!stderr_process.IsValid()) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", stderr_redirect,
          err_code);
      return PtrAsJlong(result);
    }
    if (!SetFilePointerEx(stderr_process, {0}, NULL, FILE_END)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", stderr_redirect,
          err_code);
      return PtrAsJlong(result);
    }
  } else {
    HANDLE pipe_read_h, pipe_write_h;
    if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return PtrAsJlong(result);
    }
    result->stderr_.handle_ = pipe_read_h;
    stderr_process = pipe_write_h;
  }

  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  HANDLE job = CreateJobObject(NULL, NULL);
  if (job == NULL) {
    DWORD err_code = GetLastError();
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return PtrAsJlong(result);
  }

  result->job_ = job;

  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                               &job_info, sizeof(job_info))) {
    DWORD err_code = GetLastError();
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return PtrAsJlong(result);
  }

  std::unique_ptr<bazel::windows::AutoAttributeList> attr_list;
  if (!bazel::windows::AutoAttributeList::Create(stdin_process, stdout_process,
                                                 stderr_process, &attr_list,
                                                 &error_msg)) {
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", L"", error_msg);
    return PtrAsJlong(result);
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
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"CreateProcessWithExplicitHandles",
        cmd_sample, error_msg.str().c_str());
    return PtrAsJlong(result);
  }

  STARTUPINFOEXW info;
  attr_list->InitStartupInfoExW(&info);
  if (!CreateProcessW(
          /* lpApplicationName */ NULL,
          /* lpCommandLine */ mutable_commandline.get(),
          /* lpProcessAttributes */ NULL,
          /* lpThreadAttributes */ NULL,
          /* bInheritHandles */ TRUE,
          /* dwCreationFlags */ CREATE_NO_WINDOW  // Don't create console window
              | CREATE_NEW_PROCESS_GROUP  // So that Ctrl-Break isn't propagated
              | CREATE_SUSPENDED  // So that it doesn't start a new job itself
              | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
          /* lpEnvironment */ env_map.ptr(),
          /* lpCurrentDirectory */ cwd.empty() ? NULL : cwd.c_str(),
          /* lpStartupInfo */ &info.StartupInfo,
          /* lpProcessInformation */ &process_info)) {
    DWORD err = GetLastError();
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"CreateProcessW", cmd_sample, err);
    return PtrAsJlong(result);
  }

  result->pid_ = process_info.dwProcessId;
  result->process_ = process_info.hProcess;
  thread = process_info.hThread;

  if (!AssignProcessToJobObject(result->job_, result->process_)) {
    BOOL is_in_job = false;
    if (IsProcessInJob(result->process_, NULL, &is_in_job) && is_in_job &&
        !NestedJobsSupported()) {
      // We are on a pre-Windows 8 system and the Bazel is already in a job.
      // We can't create nested jobs, so just revert to TerminateProcess() and
      // hope for the best. In batch mode, the launcher puts Bazel in a job so
      // that will take care of cleanup once the command finishes.
      CloseHandle(result->job_);
      result->job_ = INVALID_HANDLE_VALUE;
    } else {
      DWORD err_code = GetLastError();
      result->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
      return PtrAsJlong(result);
    }
  }

  // Now that we put the process in a new job object, we can start executing it
  if (ResumeThread(thread) == -1) {
    DWORD err_code = GetLastError();
    result->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
    return PtrAsJlong(result);
  }

  result->error_ = L"";
  return PtrAsJlong(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeWriteStdin(
    JNIEnv* env, jclass clazz, jlong process_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  JavaByteArray bytes(env, java_bytes);
  if (offset < 0 || length <= 0 || offset > bytes.size() - length) {
    process->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeWriteStdin", ToString(process->pid_),
        L"Array index out of bounds");
    return -1;
  }

  DWORD bytes_written;

  if (!::WriteFile(process->stdin_, bytes.ptr() + offset, length,
                   &bytes_written, NULL)) {
    DWORD err_code = GetLastError();
    process->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeWriteStdin", ToString(process->pid_),
        err_code);
    bytes_written = -1;
  }

  process->error_ = L"";
  return bytes_written;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetStdout(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return PtrAsJlong(&process->stdout_);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetStderr(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return PtrAsJlong(&process->stderr_);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeReadStream(
    JNIEnv* env, jclass clazz, jlong stream_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);

  JavaByteArray bytes(env, java_bytes);
  if (offset < 0 || length <= 0 || offset > bytes.size() - length) {
    stream->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeReadStream", L"",
        L"Array index out of bounds");
    return -1;
  }

  if (stream->handle_ == INVALID_HANDLE_VALUE || stream->closed_.load()) {
    stream->error_ = L"";
    return 0;
  }

  DWORD bytes_read;
  if (!::ReadFile(stream->handle_, bytes.ptr() + offset, length, &bytes_read,
                  NULL)) {
    // Check if either the other end closed the pipe or we did it with
    // NativeOutputStream.close() . In the latter case, we'll get a "system
    // call interrupted" error.
    if (GetLastError() == ERROR_BROKEN_PIPE || stream->closed_.load()) {
      // End of file.
      stream->error_ = L"";
      bytes_read = 0;
    } else {
      DWORD err_code = GetLastError();
      stream->error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"nativeReadStream", L"", err_code);
      bytes_read = -1;
    }
  } else {
    stream->error_ = L"";
  }

  return bytes_read;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetExitCode(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  DWORD exit_code;
  if (!GetExitCodeProcess(process->process_, &exit_code)) {
    DWORD err_code = GetLastError();
    process->error_ = bazel::windows::MakeErrorMessage(
        WSTR(__FILE__), __LINE__, L"nativeGetExitCode", ToString(process->pid_),
        err_code);
    return -1;
  }

  return exit_code;
}

// return values:
// 0: Wait completed successfully
// 1: Timeout
// 2: Wait returned with an error
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeWaitFor(
    JNIEnv* env, jclass clazz, jlong process_long, jlong java_timeout) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->WaitFor(java_timeout);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetProcessPid(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->GetPid();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeTerminate(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->Terminate();
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeDeleteProcess(
    JNIEnv* env, jclass clazz, jlong process_long) {
  delete reinterpret_cast<NativeProcess*>(process_long);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeCloseStream(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  stream->close();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeProcessGetLastError(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->getLastErrorAsString(env);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeStreamGetLastError(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  return stream->getLastErrorAsString(env);
}
