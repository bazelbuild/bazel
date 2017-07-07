// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include <memory>
#include <string>

#include "src/main/native/windows/file.h"
#include "src/main/native/windows/jni-util.h"
#include "src/main/native/windows/util.h"

static void MaybeReportLastError(const std::string& reason, JNIEnv* env,
                                 jobjectArray error_msg_holder) {
  if (error_msg_holder != nullptr &&
      env->GetArrayLength(error_msg_holder) > 0) {
    std::string error_str = bazel::windows::GetLastErrorString(reason);
    jstring error_msg = env->NewStringUTF(error_str.c_str());
    env->SetObjectArrayElement(error_msg_holder, 0, error_msg);
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsFileOperations_nativeIsJunction(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray error_msg_holder) {
  int result = bazel::windows::IsJunctionOrDirectorySymlink(
      bazel::windows::GetJavaWstring(env, path).c_str());
  if (result == bazel::windows::IS_JUNCTION_ERROR) {
    MaybeReportLastError(std::string("GetFileAttributes(") +
                             bazel::windows::GetJavaUTFString(env, path) + ")",
                         env, error_msg_holder);
  }
  return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsFileOperations_nativeGetLongPath(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray result_holder,
    jobjectArray error_msg_holder) {
  std::unique_ptr<WCHAR[]> result;
  bool success = bazel::windows::GetLongPath(
      bazel::windows::GetJavaWstring(env, path).c_str(), &result);
  if (!success) {
    MaybeReportLastError(std::string("GetLongPathName(") +
                             bazel::windows::GetJavaUTFString(env, path) + ")",
                         env, error_msg_holder);
    return JNI_FALSE;
  }
  env->SetObjectArrayElement(
      result_holder, 0,
      env->NewString(reinterpret_cast<const jchar*>(result.get()),
                     wcslen(result.get())));
  return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_jni_WindowsFileOperations_nativeCreateJunction(
    JNIEnv* env, jclass clazz, jstring name, jstring target,
    jobjectArray error_msg_holder) {
  std::string error = bazel::windows::CreateJunction(
      bazel::windows::GetJavaWstring(env, name),
      bazel::windows::GetJavaWstring(env, target));
  if (!error.empty()) {
    MaybeReportLastError(error, env, error_msg_holder);
    return JNI_FALSE;
  }
  return JNI_TRUE;
}
