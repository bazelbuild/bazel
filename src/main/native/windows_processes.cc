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

#define WINVER 0x0601
#define _WIN32_WINNT 0x0601

#include <jni.h>
#include <string.h>
#include <windows.h>

#include <atomic>
#include <memory>
#include <string>
#include <type_traits>  // static_assert

#include "src/main/native/windows_util.h"

// Ensure we can safely cast (const) jchar* to (const) WCHAR* and LP(C)WSTR.
// This is true with MSVC but not always with GCC.
static_assert(sizeof(jchar) == sizeof(WCHAR),
              "jchar and WCHAR should be the same size");

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetpid(
    JNIEnv* env, jclass clazz) {
  return GetCurrentProcessId();
}

struct NativeOutputStream {
  HANDLE handle_;
  std::string error_;
  std::atomic<bool> closed_;
  NativeOutputStream()
      : handle_(INVALID_HANDLE_VALUE),
        error_(""),
        closed_(false) {}

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

static bool NestedJobsSupported() {
  OSVERSIONINFOEX version_info;
  version_info.dwOSVersionInfoSize = sizeof(version_info);
  if (!GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&version_info))) {
    return false;
  }

  return version_info.dwMajorVersion > 6 ||
    version_info.dwMajorVersion == 6 && version_info.dwMinorVersion >= 2;
}

static std::string GetJavaUTFString(JNIEnv* env, jstring str) {
  std::string result;
  if (str != nullptr) {
    const char* jstr = env->GetStringUTFChars(str, nullptr);
    result.assign(jstr);
    env->ReleaseStringUTFChars(str, jstr);
  }
  return result;
}

static std::wstring GetJavaWstring(JNIEnv* env, jstring str) {
  std::wstring result;
  if (str != nullptr) {
    const jchar* jstr = env->GetStringChars(str, nullptr);
    // We can safely reinterpret_cast because of the static_assert checking that
    // sizeof(jchar) = sizeof(WCHAR).
    result.assign(reinterpret_cast<const WCHAR*>(jstr));
    env->ReleaseStringChars(str, jstr);
  }
  return result;
}

static std::wstring AddUncPrefixMaybe(const std::wstring& path) {
  return (path.size() > MAX_PATH) ? (std::wstring(L"\\\\?\\") + path) : path;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeCreateProcess(
    JNIEnv* env, jclass clazz, jstring java_argv0, jstring java_argv_rest,
    jbyteArray java_env, jstring java_cwd, jstring java_stdout_redirect,
    jstring java_stderr_redirect) {
  // TODO(laszlocsomor): compute the 8dot3 path for java_argv0.
  std::string commandline = GetJavaUTFString(env, java_argv0) + " " +
                            GetJavaUTFString(env, java_argv_rest);
  std::wstring stdout_redirect =
      AddUncPrefixMaybe(GetJavaWstring(env, java_stdout_redirect));
  std::wstring stderr_redirect =
      AddUncPrefixMaybe(GetJavaWstring(env, java_stderr_redirect));
  std::string cwd = GetJavaUTFString(env, java_cwd);

  jsize env_size = -1;
  jbyte* env_bytes = NULL;

  std::unique_ptr<char[]> mutable_commandline(new char[commandline.size() + 1]);
  strncpy(mutable_commandline.get(), commandline.c_str(),
          commandline.size() + 1);

  NativeProcess* result = new NativeProcess();

  SECURITY_ATTRIBUTES sa = {0};
  sa.nLength = sizeof(SECURITY_ATTRIBUTES);
  sa.bInheritHandle = TRUE;

  HANDLE stdin_process = INVALID_HANDLE_VALUE;
  HANDLE stdout_process = INVALID_HANDLE_VALUE;
  HANDLE stderr_process = INVALID_HANDLE_VALUE;
  HANDLE thread = INVALID_HANDLE_VALUE;
  HANDLE event = INVALID_HANDLE_VALUE;
  PROCESS_INFORMATION process_info = {0};
  STARTUPINFOA startup_info = {0};
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = {0};

  if (java_env != NULL) {
    env_size = env->GetArrayLength(java_env);
    env_bytes = env->GetByteArrayElements(java_env, NULL);

    if (env_size < 2) {
      result->error_ = "The environment must be at least two bytes long";
      goto cleanup;
    } else if (env_bytes[env_size - 1] != 0 || env_bytes[env_size - 2] != 0) {
      result->error_ = "Environment array must end with two null bytes";
      goto cleanup;
    }
  }

  if (!CreatePipe(&stdin_process, &result->stdin_, &sa, 0)) {
    result->error_ = windows_util::GetLastErrorString("CreatePipe(stdin)");
    goto cleanup;
  }

  if (!stdout_redirect.empty()) {
    result->stdout_.close();

    stdout_process = CreateFileW(
        /* lpFileName */ stdout_redirect.c_str(),
        /* dwDesiredAccess */ FILE_APPEND_DATA,
        /* dwShareMode */ 0,
        /* lpSecurityAttributes */ &sa,
        /* dwCreationDisposition */ OPEN_ALWAYS,
        /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
        /* hTemplateFile */ NULL);

    if (stdout_process == INVALID_HANDLE_VALUE) {
      result->error_ = windows_util::GetLastErrorString("CreateFile(stdout)");
      goto cleanup;
    }
  } else {
    if (!CreatePipe(&result->stdout_.handle_, &stdout_process, &sa, 0)) {
      result->error_ = windows_util::GetLastErrorString("CreatePipe(stdout)");
      goto cleanup;
    }
  }

  if (!stderr_redirect.empty()) {
    result->stderr_.close();
    if (stdout_redirect == stderr_redirect) {
      stderr_process = stdout_process;
    } else {
      stderr_process = CreateFileW(
          /* lpFileName */ stderr_redirect.c_str(),
          /* dwDesiredAccess */ FILE_APPEND_DATA,
          /* dwShareMode */ 0,
          /* lpSecurityAttributes */ &sa,
          /* dwCreationDisposition */ OPEN_ALWAYS,
          /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
          /* hTemplateFile */ NULL);

      if (stderr_process == INVALID_HANDLE_VALUE) {
        result->error_ = windows_util::GetLastErrorString("CreateFile(stderr)");
        goto cleanup;
      }
    }
  } else {
    if (!CreatePipe(&result->stderr_.handle_, &stderr_process, &sa, 0)) {
      result->error_ = windows_util::GetLastErrorString("CreatePipe(stderr)");
      goto cleanup;
    }
  }

  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  HANDLE job = CreateJobObject(NULL, NULL);
  if (job == NULL) {
    result->error_ = windows_util::GetLastErrorString("CreateJobObject()");
    goto cleanup;
  }

  result->job_ = job;

  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(
      job,
      JobObjectExtendedLimitInformation,
      &job_info,
      sizeof(job_info))) {
    result->error_ =
        windows_util::GetLastErrorString("SetInformationJobObject()");
    goto cleanup;
  }

  startup_info.hStdInput = stdin_process;
  startup_info.hStdOutput = stdout_process;
  startup_info.hStdError = stderr_process;
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
      /* lpEnvironment */ env_bytes,
      /* lpCurrentDirectory */ cwd.empty() ? nullptr : cwd.c_str(),
      /* lpStartupInfo */ &startup_info,
      /* lpProcessInformation */ &process_info);

  if (!ok) {
    result->error_ = windows_util::GetLastErrorString("CreateProcess()");
    goto cleanup;
  }

  result->pid_ = process_info.dwProcessId;
  result->process_ = process_info.hProcess;
  thread = process_info.hThread;

  if (!AssignProcessToJobObject(result->job_, result->process_)) {
    BOOL is_in_job = false;
    if (IsProcessInJob(result->process_, NULL, &is_in_job)
        && is_in_job
        && !NestedJobsSupported()) {
      // We are on a pre-Windows 8 system and the Bazel is already in a job.
      // We can't create nested jobs, so just revert to TerminateProcess() and
      // hope for the best. In batch mode, the launcher puts Bazel in a job so
      // that will take care of cleanup once the command finishes.
      CloseHandle(result->job_);
      result->job_ = INVALID_HANDLE_VALUE;
    } else {
      result->error_ =
          windows_util::GetLastErrorString("AssignProcessToJobObject()");
      goto cleanup;
    }
  }

  // Now that we put the process in a new job object, we can start executing it
  if (ResumeThread(thread) == -1) {
    result->error_ = windows_util::GetLastErrorString("ResumeThread()");
    goto cleanup;
  }

  result->error_ = "";

cleanup:
  // Standard file handles are closed even if the process was successfully
  // created. If this was not so, operations on these file handles would not
  // return immediately if the process is terminated.
  if (stdin_process != INVALID_HANDLE_VALUE) {
    CloseHandle(stdin_process);
  }

  if (stdout_process != INVALID_HANDLE_VALUE) {
    CloseHandle(stdout_process);
  }

  if (stderr_process != INVALID_HANDLE_VALUE
      && stderr_process != stdout_process) {
    CloseHandle(stderr_process);
  }

  if (thread != INVALID_HANDLE_VALUE) {
    CloseHandle(thread);
  }

  if (env_bytes != NULL) {
    env->ReleaseByteArrayElements(java_env, env_bytes, 0);
  }

  return reinterpret_cast<jlong>(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeWriteStdin(
    JNIEnv *env, jclass clazz, jlong process_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  jsize array_size = env->GetArrayLength(java_bytes);
  if (offset < 0 || length <= 0 || offset > array_size - length) {
    process->error_ = "Array index out of bounds";
    return -1;
  }

  jbyte* bytes = env->GetByteArrayElements(java_bytes, NULL);
  DWORD bytes_written;

  if (!::WriteFile(process->stdin_, bytes + offset, length, &bytes_written,
                   NULL)) {
    process->error_ = windows_util::GetLastErrorString("WriteFile()");
    bytes_written = -1;
  }

  env->ReleaseByteArrayElements(java_bytes, bytes, 0);
  process->error_ = "";
  return bytes_written;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetStdout(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return reinterpret_cast<jlong>(&process->stdout_);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetStderr(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return reinterpret_cast<jlong>(&process->stderr_);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeReadStream(
    JNIEnv* env, jclass clazz, jlong stream_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);

  jsize array_size = env->GetArrayLength(java_bytes);
  if (offset < 0 || length <= 0 || offset > array_size - length) {
    stream->error_ = "Array index out of bounds";
    return -1;
  }

  if (stream->handle_ == INVALID_HANDLE_VALUE || stream->closed_.load()) {
    stream->error_ = "";
    return 0;
  }

  jbyte* bytes = env->GetByteArrayElements(java_bytes, NULL);
  DWORD bytes_read;
  if (!::ReadFile(stream->handle_, bytes + offset, length, &bytes_read, NULL)) {
    // Check if either the other end closed the pipe or we did it with
    // NativeOutputStream.close() . In the latter case, we'll get a "system
    // call interrupted" error.
    if (GetLastError() == ERROR_BROKEN_PIPE || stream->closed_.load()) {
      // End of file.
      stream->error_ = "";
      bytes_read = 0;
    } else {
      stream->error_ = windows_util::GetLastErrorString("ReadFile()");
      bytes_read = -1;
    }
  } else {
    stream->error_ = "";
  }

  env->ReleaseByteArrayElements(java_bytes, bytes, 0);
  return bytes_read;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetExitCode(
    JNIEnv *env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  DWORD exit_code;
  if (!GetExitCodeProcess(process->process_, &exit_code)) {
    process->error_ = windows_util::GetLastErrorString("GetExitCodeProcess()");
    return -1;
  }

  return exit_code;
}

// return values:
// 0: Wait completed successfully
// 1: Timeout
// 2: Wait returned with an error
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeWaitFor(
    JNIEnv *env, jclass clazz, jlong process_long, jlong java_timeout) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  HANDLE handles[1] = { process->process_ };
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
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetProcessPid(
    JNIEnv *env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  process->error_ = "";
  return GetProcessId(process->process_);  // MSDN says that this cannot fail
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeTerminate(
    JNIEnv *env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);

  if (process->job_ != INVALID_HANDLE_VALUE) {
    // In theory, CloseHandle() on process->job_ would work, too, since we set
    // KILL_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, but this is a little more explicit.
    if (!TerminateJobObject(process->job_, 0)) {
      process->error_ =
          windows_util::GetLastErrorString("TerminateJobObject()");
      return JNI_FALSE;
    }
  } else if (process->process_ != INVALID_HANDLE_VALUE) {
    if (!TerminateProcess(process->process_, 1)) {
      process->error_ = windows_util::GetLastErrorString("TerminateProcess()");
      return JNI_FALSE;
    }
  }

  process->error_ = "";
  return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeDeleteProcess(
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
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeCloseStream(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  stream->close();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeProcessGetLastError(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  jstring result = env->NewStringUTF(process->error_.c_str());
  process->error_ = "";
  return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeStreamGetLastError(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  jstring result = env->NewStringUTF(stream->error_.c_str());
  stream->error_ = "";
  return result;
}
