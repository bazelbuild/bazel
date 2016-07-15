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

#include <jni.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include <string>

std::string GetLastErrorString(const std::string& cause) {
  DWORD last_error = GetLastError();
  if (last_error == 0) {
    return "";
  }

  LPSTR message;
  DWORD size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER
          | FORMAT_MESSAGE_FROM_SYSTEM
          | FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      last_error,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR) &message,
      0,
      NULL);

  if (size == 0) {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "%s: Error %d (cannot format message due to error %d)",
        cause.c_str(), last_error, GetLastError());
    buf[sizeof(buf) - 1] = 0;
  }

  std::string result = std::string(message);
  LocalFree(message);
  return cause + ": " + result;
}
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetpid(
    JNIEnv* env, jclass clazz) {
  return GetCurrentProcessId();
}

struct NativeOutputStream {
  HANDLE handle_;
  std::string error_;

  NativeOutputStream() : handle_(INVALID_HANDLE_VALUE), error_("") {}

  void close() {
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
  }
};

struct NativeProcess {
  HANDLE stdin_;
  NativeOutputStream stdout_;
  NativeOutputStream stderr_;
  HANDLE process_;
  HANDLE job_;
  std::string error_;

  NativeProcess()
      : stdin_(INVALID_HANDLE_VALUE),
        stdout_(),
        stderr_(),
        process_(INVALID_HANDLE_VALUE),
        job_(INVALID_HANDLE_VALUE),
        error_("") {}
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeCreateProcess(
  JNIEnv *env, jclass clazz, jstring java_commandline, jbyteArray java_env,
  jstring java_cwd, jstring java_stdout_redirect,
  jstring java_stderr_redirect) {
  const char* commandline = env->GetStringUTFChars(java_commandline, NULL);
  const char* stdout_redirect = NULL;
  const char* stderr_redirect = NULL;
  const char* cwd = NULL;

  if (java_stdout_redirect != NULL) {
    stdout_redirect = env->GetStringUTFChars(java_stdout_redirect, NULL);
  }

  if (java_stderr_redirect != NULL) {
    stderr_redirect = env->GetStringUTFChars(java_stderr_redirect, NULL);
  }

  if (java_cwd != NULL) {
    cwd = env->GetStringUTFChars(java_cwd, NULL);
  }

  jsize env_size = -1;
  jbyte* env_bytes = NULL;


  char* mutable_commandline = new char[strlen(commandline) + 1];
  strncpy(mutable_commandline, commandline, strlen(commandline) + 1);

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
  STARTUPINFO startup_info = {0};

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
    result->error_ = GetLastErrorString("CreatePipe(stdin)");
    goto cleanup;
  }

  if (stdout_redirect != NULL) {
    stdout_process = CreateFile(
        stdout_redirect,
        FILE_APPEND_DATA,
        0,
        &sa,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (stdout_process == INVALID_HANDLE_VALUE) {
      result->error_ = GetLastErrorString("CreateFile(stdout)");
      goto cleanup;
    }
  } else {
    if (!CreatePipe(&result->stdout_.handle_, &stdout_process, &sa, 0)) {
      result->error_ = GetLastErrorString("CreatePipe(stdout)");
      goto cleanup;
    }
  }

  if (stderr_redirect != NULL) {
    if (!strcmp(stdout_redirect, stderr_redirect)) {
      stderr_process = stdout_process;
    } else {
      stderr_process = CreateFile(
          stderr_redirect,
          FILE_APPEND_DATA,
          0,
          &sa,
          OPEN_ALWAYS,
          FILE_ATTRIBUTE_NORMAL,
          NULL);

      if (stderr_process == INVALID_HANDLE_VALUE) {
        result->error_ = GetLastErrorString("CreateFile(stderr)");
        goto cleanup;
      }
    }
  } else {
    if (!CreatePipe(&result->stderr_.handle_, &stderr_process, &sa, 0)) {
      result->error_ = GetLastErrorString("CreatePipe(stderr)");
      goto cleanup;
    }
  }


  // MDSN says that the default for job objects is that breakaway is not
  // allowed. Thus, we don't need to do any more setup here.
  HANDLE job = CreateJobObject(NULL, NULL);
  if (job == NULL) {
    result->error_ = GetLastErrorString("CreateJobObject()");
    goto cleanup;
  }

  result->job_ = job;

  JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_info = { 0 };
  job_info.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(
      job,
      JobObjectExtendedLimitInformation,
      &job_info,
      sizeof(job_info))) {
      result->error_ = GetLastErrorString("SetInformationJobObject()");
      goto cleanup;
  }

  startup_info.hStdInput = stdin_process;
  startup_info.hStdOutput = stdout_process;
  startup_info.hStdError = stderr_process;
  startup_info.dwFlags |= STARTF_USESTDHANDLES;

  BOOL ok = CreateProcess(
      NULL,
      mutable_commandline,
      NULL,
      NULL,
      TRUE,
      CREATE_NO_WINDOW  // Don't create a console window
          | CREATE_NEW_PROCESS_GROUP   // So that Ctrl-Break is not propagated
          | CREATE_SUSPENDED,  // So that it doesn't start a new job itself
      env_bytes,
      cwd,
      &startup_info,
      &process_info);

  if (!ok) {
    result->error_ = GetLastErrorString("CreateProcess()");
    goto cleanup;
  }

  result->process_ = process_info.hProcess;
  thread = process_info.hThread;

  if (!AssignProcessToJobObject(result->job_, result->process_)) {
    // todo(lberki): Fix job control (GitHub issue #1527).
    // result->error_ = GetLastErrorString("AssignProcessToJobObject()");
    // goto cleanup;
  }

  // Now that we put the process in a new job object, we can start executing it
  if (ResumeThread(thread) == -1) {
    result->error_ = GetLastErrorString("ResumeThread()");
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

  delete[] mutable_commandline;
  if (env_bytes != NULL) {
    env->ReleaseByteArrayElements(java_env, env_bytes, 0);
  }
  env->ReleaseStringUTFChars(java_commandline, commandline);

  if (stdout_redirect != NULL) {
    env->ReleaseStringUTFChars(java_stdout_redirect, stdout_redirect);
  }

  if (stderr_redirect != NULL) {
    env->ReleaseStringUTFChars(java_stderr_redirect, stderr_redirect);
  }

  if (cwd != NULL) {
    env->ReleaseStringUTFChars(java_cwd, cwd);
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

  if (!WriteFile(process->stdin_, bytes + offset, length, &bytes_written,
      NULL)) {
    process->error_ = GetLastErrorString("WriteFile()");
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

  if (stream->handle_ == INVALID_HANDLE_VALUE) {
    stream->error_ = "File handle closed";
    return -1;
  }

  jsize array_size = env->GetArrayLength(java_bytes);
  if (offset < 0 || length <= 0 || offset > array_size - length) {
    stream->error_ = "Array index out of bounds";
    return -1;
  }

  jbyte* bytes = env->GetByteArrayElements(java_bytes, NULL);
  DWORD bytes_read;
  if (!ReadFile(stream->handle_, bytes + offset, length, &bytes_read, NULL)) {
    if (GetLastError() == ERROR_BROKEN_PIPE) {
      // End of file.
      stream->error_ = "";
      bytes_read = 0;
    } else {
      stream->error_ = GetLastErrorString("ReadFile()");
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
    process->error_ = GetLastErrorString("GetExitCodeProcess()");
    return -1;
  }

  return exit_code;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeWaitFor(
    JNIEnv *env, jclass clazz, jlong process_long) {
  NativeProcess* process = reinterpret_cast<NativeProcess*>(process_long);
  HANDLE handles[1] = { process->process_ };
  switch (WaitForMultipleObjects(1, handles, FALSE, INFINITE)) {
    case 0:
      return true;

    case WAIT_FAILED:
      return false;

    default:
      process->error_ = "WaitForMultipleObjects() returned unknown result";
      return false;
  }
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

  // In theory, CloseHandle() on process->job_ would work, too, since we set
  // KILL_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, but this is a little more explicit.
  if (!TerminateJobObject(process->job_, 0)) {
    process->error_ = GetLastErrorString("TerminateJobObject()");
    return JNI_FALSE;
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
