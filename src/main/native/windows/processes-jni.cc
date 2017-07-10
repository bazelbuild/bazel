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
#define WINVER 0x0601
#define _WIN32_WINNT 0x0601

#include <jni.h>
#include <windows.h>

#include <atomic>
#include <memory>
#include <string>

#include "src/main/native/windows/jni-util.h"
#include "src/main/native/windows/util.h"

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetpid(
    JNIEnv* env, jclass clazz) {
  return GetCurrentProcessId();
}

struct NativeOutputStream {
  HANDLE handle_;
  std::string error_;
  std::atomic<bool> closed_;
  NativeOutputStream()
      : handle_(INVALID_HANDLE_VALUE), error_(""), closed_(false) {}

  void close() {
    closed_.store(true);
    if (handle_ == INVALID_HANDLE_VALUE) {
      return;
    }

    CancelIoEx(handle_, NULL);
    CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
  }
};

static std::wstring AddUncPrefixMaybe(const std::wstring& path) {
  return (path.size() >= MAX_PATH) ? (std::wstring(L"\\\\?\\") + path) : path;
}

struct NativeProcess {
  HANDLE stdin_;
  NativeOutputStream stdout_;
  NativeOutputStream stderr_;
  HANDLE process_;
  HANDLE job_;
  DWORD pid_;
  std::string error_;

  NativeProcess()
      : stdin_(INVALID_HANDLE_VALUE),
        stdout_(),
        stderr_(),
        process_(INVALID_HANDLE_VALUE),
        job_(INVALID_HANDLE_VALUE),
        error_("") {}
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

static bool NestedJobsSupported() {
  OSVERSIONINFOEX version_info;
  version_info.dwOSVersionInfoSize = sizeof(version_info);
  if (!GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&version_info))) {
    return false;
  }

  return version_info.dwMajorVersion > 6 ||
         version_info.dwMajorVersion == 6 && version_info.dwMinorVersion >= 2;
}

static jlong PtrAsJlong(void* p) { return reinterpret_cast<jlong>(p); }

static std::string AsExecutableForCreateProcess(JNIEnv* env, jstring path,
                                                std::string* result) {
  return bazel::windows::AsExecutablePathForCreateProcess(
      bazel::windows::GetJavaUTFString(env, path),
      [env, path]() { return bazel::windows::GetJavaWstring(env, path); },
      result);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeCreateProcess(
    JNIEnv* env, jclass clazz, jstring java_argv0, jstring java_argv_rest,
    jbyteArray java_env, jstring java_cwd, jstring java_stdout_redirect,
    jstring java_stderr_redirect) {
  NativeProcess* result = new NativeProcess();

  std::string argv0;
  std::string error_msg(AsExecutableForCreateProcess(env, java_argv0, &argv0));
  if (!error_msg.empty()) {
    result->error_ = error_msg;
    return PtrAsJlong(result);
  }

  std::string commandline =
      argv0 + " " + bazel::windows::GetJavaUTFString(env, java_argv_rest);
  std::wstring stdout_redirect = AddUncPrefixMaybe(
      bazel::windows::GetJavaWstring(env, java_stdout_redirect));
  std::wstring stderr_redirect = AddUncPrefixMaybe(
      bazel::windows::GetJavaWstring(env, java_stderr_redirect));
  std::string cwd;
  error_msg = bazel::windows::AsShortPath(
      bazel::windows::GetJavaUTFString(env, java_cwd),
      [env, java_cwd]() {
        return bazel::windows::GetJavaWstring(env, java_cwd);
      },
      &cwd);
  if (!error_msg.empty()) {
    result->error_ = error_msg;
    return PtrAsJlong(result);
  }

  std::unique_ptr<char[]> mutable_commandline(new char[commandline.size() + 1]);
  strncpy(mutable_commandline.get(), commandline.c_str(),
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
  STARTUPINFOA startup_info = {0};
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};

  JavaByteArray env_map(env, java_env);
  if (env_map.ptr() != nullptr) {
    if (env_map.size() < 2) {
      result->error_ = "The environment must be at least two bytes long";
      return PtrAsJlong(result);
    } else if (env_map.ptr()[env_map.size() - 1] != 0 ||
               env_map.ptr()[env_map.size() - 2] != 0) {
      result->error_ = "Environment array must end with two null bytes";
      return PtrAsJlong(result);
    }
  }

  if (!CreatePipe(&stdin_process.handle, &result->stdin_, &sa, 0)) {
    result->error_ = bazel::windows::GetLastErrorString("CreatePipe(stdin)");
    return PtrAsJlong(result);
  }

  if (!stdout_redirect.empty()) {
    result->stdout_.close();

    stdout_process.handle = CreateFileW(
        /* lpFileName */ stdout_redirect.c_str(),
        /* dwDesiredAccess */ FILE_APPEND_DATA,
        /* dwShareMode */ 0,
        /* lpSecurityAttributes */ &sa,
        /* dwCreationDisposition */ OPEN_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);

    if (!stdout_process.IsValid()) {
      result->error_ = bazel::windows::GetLastErrorString("CreateFile(stdout)");
      return PtrAsJlong(result);
    }
  } else {
    if (!CreatePipe(&result->stdout_.handle_, &stdout_process.handle, &sa, 0)) {
      result->error_ = bazel::windows::GetLastErrorString("CreatePipe(stdout)");
      return PtrAsJlong(result);
    }
  }

  // The value of the stderr HANDLE.
  // If stdout and stderr are redirected to the same file, this will be equal to
  // stdout_process.handle and stderr_process.handle will be
  // INVALID_HANDLE_VALUE, so the two AutoHandle objects' d'tors will not
  // attempt to close stdout's handle twice.
  // If stdout != stderr, then stderr_handle = stderr_process.handle, and these
  // are distinct from stdout_process.handle, so again the d'tors will do the
  // right thing, closing the handles.
  // In both cases, we DO NOT close stderr_handle, since it's either
  // stdout_process's or stderr_process's d'tor doing so.
  HANDLE stderr_handle = INVALID_HANDLE_VALUE;

  if (!stderr_redirect.empty()) {
    result->stderr_.close();
    if (stdout_redirect == stderr_redirect) {
      stderr_handle = stdout_process.handle;
      // do not set stderr_process.handle; it equals stdout_process.handle and
      // the AutoHandle d'tor would attempt to close it again
    } else {
      stderr_handle = CreateFileW(
          /* lpFileName */ stderr_redirect.c_str(),
          /* dwDesiredAccess */ FILE_APPEND_DATA,
          /* dwShareMode */ 0,
          /* lpSecurityAttributes */ &sa,
          /* dwCreationDisposition */ OPEN_ALWAYS,
          /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
          /* hTemplateFile */ NULL);

      if (stderr_handle == INVALID_HANDLE_VALUE) {
        result->error_ =
            bazel::windows::GetLastErrorString("CreateFile(stderr)");
        return PtrAsJlong(result);
      }
      // stderr_process != stdout_process, so set its handle, so the AutoHandle
      // d'tor will close it
      stderr_process.handle = stderr_handle;
    }
  } else {
    if (!CreatePipe(&result->stderr_.handle_, &stderr_handle, &sa, 0)) {
      result->error_ = bazel::windows::GetLastErrorString("CreatePipe(stderr)");
      return PtrAsJlong(result);
    }
    stderr_process.handle = stderr_handle;
  }

  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  HANDLE job = CreateJobObject(NULL, NULL);
  if (job == NULL) {
    result->error_ = bazel::windows::GetLastErrorString("CreateJobObject()");
    return PtrAsJlong(result);
  }

  result->job_ = job;

  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                               &job_info, sizeof(job_info))) {
    result->error_ =
        bazel::windows::GetLastErrorString("SetInformationJobObject()");
    return PtrAsJlong(result);
  }

  startup_info.hStdInput = stdin_process.handle;
  startup_info.hStdOutput = stdout_process.handle;
  startup_info.hStdError = stderr_handle;
  startup_info.dwFlags |= STARTF_USESTDHANDLES;

  BOOL ok = CreateProcessA(
      /* lpApplicationName */ NULL,
      /* lpCommandLine */ mutable_commandline.get(),
      /* lpProcessAttributes */ NULL,
      /* lpThreadAttributes */ NULL,
      /* bInheritHandles */ TRUE,
      /* dwCreationFlags */ CREATE_NO_WINDOW  // Don't create a console window
          | CREATE_NEW_PROCESS_GROUP  // So that Ctrl-Break is not propagated
          | CREATE_SUSPENDED,  // So that it doesn't start a new job itself
      /* lpEnvironment */ env_map.ptr(),
      /* lpCurrentDirectory */ cwd.empty() ? nullptr : cwd.c_str(),
      /* lpStartupInfo */ &startup_info,
      /* lpProcessInformation */ &process_info);

  if (!ok) {
    result->error_ = bazel::windows::GetLastErrorString("CreateProcess()");
    return PtrAsJlong(result);
  }

  result->pid_ = process_info.dwProcessId;
  result->process_ = process_info.hProcess;
  thread.handle = process_info.hThread;

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
      result->error_ =
          bazel::windows::GetLastErrorString("AssignProcessToJobObject()");
      return PtrAsJlong(result);
    }
  }

  // Now that we put the process in a new job object, we can start executing it
  if (ResumeThread(thread) == -1) {
    result->error_ = bazel::windows::GetLastErrorString("ResumeThread()");
    return PtrAsJlong(result);
  }

  result->error_ = "";
  return PtrAsJlong(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeWriteStdin(
    JNIEnv* env, jclass clazz, jlong process_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  JavaByteArray bytes(env, java_bytes);
  if (offset < 0 || length <= 0 || offset > bytes.size() - length) {
    process->error_ = "Array index out of bounds";
    return -1;
  }

  DWORD bytes_written;

  if (!::WriteFile(process->stdin_, bytes.ptr() + offset, length,
                   &bytes_written, NULL)) {
    process->error_ = bazel::windows::GetLastErrorString("WriteFile()");
    bytes_written = -1;
  }

  process->error_ = "";
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
    stream->error_ = "Array index out of bounds";
    return -1;
  }

  if (stream->handle_ == INVALID_HANDLE_VALUE || stream->closed_.load()) {
    stream->error_ = "";
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
      stream->error_ = "";
      bytes_read = 0;
    } else {
      stream->error_ = bazel::windows::GetLastErrorString("ReadFile()");
      bytes_read = -1;
    }
  } else {
    stream->error_ = "";
  }

  return bytes_read;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetExitCode(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  DWORD exit_code;
  if (!GetExitCodeProcess(process->process_, &exit_code)) {
    process->error_ =
        bazel::windows::GetLastErrorString("GetExitCodeProcess()");
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
  HANDLE handles[1] = {process->process_};
  DWORD win32_timeout = java_timeout < 0 ? INFINITE : java_timeout;
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
      process->error_ = "WaitForMultipleObjects() returned unknown result";
      result = 2;
      break;
  }

  // Close the pipe handles so that any pending nativeReadStream() calls
  // return. This will call CancelIoEx() on the file handles in order to make
  // ReadFile() in nativeReadStream() return; otherwise, CloseHandle() would
  // hang.
  //
  // This protects against a subprocess being created, it passing the write
  // side of the stdout/stderr pipes to a subprocess, then dying. In that case,
  // if we didn't do this, the Java side of the code would hang waiting for the
  // streams to finish.
  //
  // An alternative implementation would be to rely on job control terminating
  // the subprocesses, but we don't want to assume that it's always available.
  process->stdout_.close();
  process->stderr_.close();

  if (process->stdin_ != INVALID_HANDLE_VALUE) {
    CloseHandle(process->stdin_);
    process->stdin_ = INVALID_HANDLE_VALUE;
  }

  return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetProcessPid(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  process->error_ = "";
  return GetProcessId(process->process_);  // MSDN says that this cannot fail
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeTerminate(
    JNIEnv* env, jclass clazz, jlong process_long) {
  static const UINT exit_code = 130;  // 128 + SIGINT, like on Linux
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  if (process->job_ != INVALID_HANDLE_VALUE) {
    if (!TerminateJobObject(process->job_, exit_code)) {
      process->error_ =
          bazel::windows::GetLastErrorString("TerminateJobObject()");
      return JNI_FALSE;
    }
  } else if (process->process_ != INVALID_HANDLE_VALUE) {
    if (!TerminateProcess(process->process_, exit_code)) {
      process->error_ =
          bazel::windows::GetLastErrorString("TerminateProcess()");
      return JNI_FALSE;
    }
  }

  process->error_ = "";
  return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeDeleteProcess(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  if (process->stdin_ != INVALID_HANDLE_VALUE) {
    CloseHandle(process->stdin_);
  }

  process->stdout_.close();
  process->stderr_.close();

  if (process->process_ != INVALID_HANDLE_VALUE) {
    CloseHandle(process->process_);
  }

  if (process->job_ != INVALID_HANDLE_VALUE) {
    CloseHandle(process->job_);
  }

  delete process;
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
  jstring result = env->NewStringUTF(process->error_.c_str());
  process->error_ = "";
  return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeStreamGetLastError(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  jstring result = env->NewStringUTF(stream->error_.c_str());
  stream->error_ = "";
  return result;
}
