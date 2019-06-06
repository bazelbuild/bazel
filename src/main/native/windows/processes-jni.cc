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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <stdint.h>
#include <wchar.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>  // static_assert

#include "src/main/native/jni.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/jni-util.h"
#include "src/main/native/windows/process.h"
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

class NativeOutputStream {
 public:
  NativeOutputStream()
      : handle_(INVALID_HANDLE_VALUE), error_(L""), closed_(false) {}

  void Close() {
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

  void SetHandle(HANDLE handle) { handle_ = handle; }

  jint ReadStream(JNIEnv* env, jbyteArray java_bytes, jint offset,
                  jint length) {
    JavaByteArray bytes(env, java_bytes);
    if (offset < 0 || length <= 0 || offset > bytes.size() - length) {
      error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"nativeReadStream", L"",
                                                L"Array index out of bounds");
      return -1;
    }

    if (handle_ == INVALID_HANDLE_VALUE || closed_.load()) {
      error_ = L"";
      return 0;
    }

    DWORD bytes_read;
    if (!::ReadFile(handle_, bytes.ptr() + offset, length, &bytes_read, NULL)) {
      // Check if either the other end closed the pipe or we did it with
      // NativeOutputStream.Close() . In the latter case, we'll get a "system
      // call interrupted" error.
      if (GetLastError() == ERROR_BROKEN_PIPE || closed_.load()) {
        // End of file.
        error_ = L"";
        bytes_read = 0;
      } else {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeReadStream", L"", err_code);
        bytes_read = -1;
      }
    } else {
      error_ = L"";
    }

    return bytes_read;
  }

  // Return the last error as a human-readable string and clear it.
  jstring GetLastErrorAsString(JNIEnv* env) {
    jstring result = env->NewString(
        reinterpret_cast<const jchar*>(error_.c_str()), error_.size());
    error_ = L"";
    return result;
  }

 private:
  HANDLE handle_;
  std::wstring error_;
  std::atomic<bool> closed_;
};

class NativeProcess {
 public:
  NativeProcess() : stdout_(), stderr_(), error_(L"") {}

  ~NativeProcess() {
    stdout_.Close();
    stderr_.Close();
  }

  jboolean Create(JNIEnv* env, jstring java_argv0, jstring java_argv_rest,
                  jbyteArray java_env, jstring java_cwd,
                  jstring java_stdout_redirect, jstring java_stderr_redirect,
                  jboolean redirectErrorStream) {
    std::wstring wpath(bazel::windows::GetJavaWpath(env, java_argv0));

    std::wstring stdout_redirect = bazel::windows::AddUncPrefixMaybe(
        bazel::windows::GetJavaWpath(env, java_stdout_redirect));
    std::wstring stderr_redirect = bazel::windows::AddUncPrefixMaybe(
        bazel::windows::GetJavaWpath(env, java_stderr_redirect));

    const bool stdout_is_stream = stdout_redirect.empty();
    const bool stderr_is_stream =
        redirectErrorStream ? stdout_is_stream : stderr_redirect.empty();
    const bool stderr_same_handle_as_stdout =
        redirectErrorStream ||
        (!stderr_redirect.empty() &&
         stderr_redirect.size() == stdout_redirect.size() &&
         _wcsnicmp(stderr_redirect.c_str(), stdout_redirect.c_str(),
                   stderr_redirect.size()) == 0);

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

    JavaByteArray env_map(env, java_env);
    if (env_map.ptr() != nullptr) {
      if (env_map.size() < 4) {
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath,
            std::wstring(
                L"the environment must be at least 4 bytes long, was ") +
                ToString(env_map.size()) + L" bytes");
        return false;
      } else if (env_map.ptr()[env_map.size() - 1] != 0 ||
                 env_map.ptr()[env_map.size() - 2] != 0 ||
                 env_map.ptr()[env_map.size() - 3] != 0 ||
                 env_map.ptr()[env_map.size() - 4] != 0) {
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath,
            L"environment array must end with 4 null bytes");
        return false;
      }
    }

    {
      HANDLE pipe_read_h, pipe_write_h;
      if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
        return false;
      }
      stdin_process = pipe_read_h;
      stdin_ = pipe_write_h;
    }

    if (!stdout_is_stream) {
      stdout_.Close();

      stdout_process = CreateFileW(
          /* lpFileName */ stdout_redirect.c_str(),
          /* dwDesiredAccess */ GENERIC_WRITE,
          // Must share for reading, otherwise symlink-following file existence
          // checks (e.g. java.nio.file.Files.exists()) fail.
          /* dwShareMode */ FILE_SHARE_READ,
          /* lpSecurityAttributes */ &sa,
          /* dwCreationDisposition */ OPEN_ALWAYS,
          /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
          /* hTemplateFile */ NULL);

      if (!stdout_process.IsValid()) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                  L"nativeCreateProcess",
                                                  stdout_redirect, err_code);
        return false;
      }
      if (!SetFilePointerEx(stdout_process, {0}, NULL, FILE_END)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                  L"nativeCreateProcess",
                                                  stdout_redirect, err_code);
        return false;
      }
    } else {
      HANDLE pipe_read_h, pipe_write_h;
      if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
        return false;
      }
      stdout_.SetHandle(pipe_read_h);
      stdout_process = pipe_write_h;
    }

    if (stderr_same_handle_as_stdout) {
      HANDLE stdout_process_dup_h;
      if (!DuplicateHandle(GetCurrentProcess(), stdout_process,
                           GetCurrentProcess(), &stdout_process_dup_h, 0, TRUE,
                           DUPLICATE_SAME_ACCESS)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
        return false;
      }
      if (!stderr_is_stream) {
        stderr_.Close();
      }

      stderr_process = stdout_process_dup_h;
    } else if (!stderr_redirect.empty()) {
      stderr_.Close();
      stderr_process = CreateFileW(
          /* lpFileName */ stderr_redirect.c_str(),
          /* dwDesiredAccess */ GENERIC_WRITE,
          // Must share for reading, otherwise symlink-following file existence
          // checks (e.g. java.nio.file.Files.exists()) fail.
          /* dwShareMode */ FILE_SHARE_READ,
          /* lpSecurityAttributes */ &sa,
          /* dwCreationDisposition */ OPEN_ALWAYS,
          /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
          /* hTemplateFile */ NULL);

      if (!stderr_process.IsValid()) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                  L"nativeCreateProcess",
                                                  stderr_redirect, err_code);
        return false;
      }
      if (!SetFilePointerEx(stderr_process, {0}, NULL, FILE_END)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                  L"nativeCreateProcess",
                                                  stderr_redirect, err_code);
        return false;
      }
    } else {
      HANDLE pipe_read_h, pipe_write_h;
      if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0)) {
        DWORD err_code = GetLastError();
        error_ = bazel::windows::MakeErrorMessage(
            WSTR(__FILE__), __LINE__, L"nativeCreateProcess", wpath, err_code);
        return false;
      }
      stderr_.SetHandle(pipe_read_h);
      stderr_process = pipe_write_h;
    }
    return proc_.Create(
        wpath, bazel::windows::GetJavaWstring(env, java_argv_rest),
        env_map.ptr(), bazel::windows::GetJavaWpath(env, java_cwd),
        stdin_process, stdout_process, stderr_process, nullptr, &error_);
  }

  void CloseStdin() {
    if (stdin_.IsValid()) {
      stdin_ = INVALID_HANDLE_VALUE;
    }
  }

  // Wait for this process to exit (or timeout).
  int WaitFor(int64_t timeout_msec) {
    return proc_.WaitFor(timeout_msec, nullptr, &error_);
  }

  // Returns the exit code of the process if it has already exited. If the
  // process is still running, returns STILL_ACTIVE (= 259).
  int GetExitCode() { return proc_.GetExitCode(&error_); }

  DWORD GetPid() const { return proc_.GetPid(); }

  jint WriteStdin(JNIEnv* env, jbyteArray java_bytes, jint offset,
                  jint length) {
    JavaByteArray bytes(env, java_bytes);
    if (offset < 0 || length <= 0 || offset > bytes.size() - length) {
      error_ = bazel::windows::MakeErrorMessage(
          WSTR(__FILE__), __LINE__, L"NativeProcess:WriteStdin",
          ToString(GetPid()), L"Array index out of bounds");
      return -1;
    }

    DWORD bytes_written;

    if (!::WriteFile(stdin_, bytes.ptr() + offset, length, &bytes_written,
                     NULL)) {
      DWORD err_code = GetLastError();
      error_ = bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                                L"NativeProcess:WriteStdin",
                                                ToString(GetPid()), err_code);
      return -1;
    }

    error_ = L"";
    return bytes_written;
  }

  NativeOutputStream* GetStdoutStream() { return &stdout_; }

  NativeOutputStream* GetStderrStream() { return &stderr_; }

  // Terminates this process (and subprocesses, if job objects are available).
  bool Terminate() { return proc_.Terminate(&error_); }

  // Return the last error as a human-readable string and clear it.
  jstring GetLastErrorAsString(JNIEnv* env) {
    jstring result = env->NewString(
        reinterpret_cast<const jchar*>(error_.c_str()), error_.size());
    error_ = L"";
    return result;
  }

 private:
  bazel::windows::AutoHandle stdin_;
  NativeOutputStream stdout_;
  NativeOutputStream stderr_;
  std::wstring error_;
  bazel::windows::WaitableProcess proc_;
};

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
  // TODO(philwo) The `Create` method returns false in case of an error. But
  // there seems to be no good way to signal an error at this point to Bazel.
  // The way the code currently works is that the Java code explicitly calls
  // nativeProcessGetLastError(), so it's OK, but it would be nice if we
  // could just throw an exception here.
  result->Create(env, java_argv0, java_argv_rest, java_env, java_cwd,
                 java_stdout_redirect, java_stderr_redirect,
                 redirectErrorStream);
  return PtrAsJlong(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeWriteStdin(
    JNIEnv* env, jclass clazz, jlong process_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->WriteStdin(env, java_bytes, offset, length);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetStdout(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return PtrAsJlong(process->GetStdoutStream());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetStderr(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return PtrAsJlong(process->GetStderrStream());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeReadStream(
    JNIEnv* env, jclass clazz, jlong stream_long, jbyteArray java_bytes,
    jint offset, jint length) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  return stream->ReadStream(env, java_bytes, offset, length);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetExitCode(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return static_cast<jint>(process->GetExitCode());
}

// return values:
// 0: Wait completed successfully
// 1: Timeout
// 2: Wait returned with an error
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeWaitFor(
    JNIEnv* env, jclass clazz, jlong process_long, jlong java_timeout) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  int res = process->WaitFor(static_cast<int64_t>(java_timeout));
  process->CloseStdin();
  return static_cast<jint>(res);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeGetProcessPid(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return static_cast<jint>(process->GetPid());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeTerminate(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->Terminate() ? JNI_TRUE : JNI_FALSE;
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
  stream->Close();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeProcessGetLastError(
    JNIEnv* env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  return process->GetLastErrorAsString(env);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsProcesses_nativeStreamGetLastError(
    JNIEnv* env, jclass clazz, jlong stream_long) {
  NativeOutputStream* stream =
      reinterpret_cast<NativeOutputStream*>(stream_long);
  return stream->GetLastErrorAsString(env);
}
