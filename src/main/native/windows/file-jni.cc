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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <memory>
#include <sstream>
#include <string>

#include "src/main/native/jni.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/jni-util.h"
#include "src/main/native/windows/util.h"

static bool CanReportError(JNIEnv* env, jobjectArray error_msg_holder) {
  return error_msg_holder != nullptr &&
         env->GetArrayLength(error_msg_holder) > 0;
}

static void ReportLastError(const std::wstring& error_str, JNIEnv* env,
                            jobjectArray error_msg_holder) {
  jstring error_msg = env->NewString(
      reinterpret_cast<const jchar*>(error_str.c_str()), error_str.size());
  env->SetObjectArrayElement(error_msg_holder, 0, error_msg);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeIsSymlinkOrJunction(
    JNIEnv* env, jclass clazz, jstring path, jbooleanArray result_holder,
    jobjectArray error_msg_holder) {
  std::wstring wpath(bazel::windows::GetJavaWstring(env, path));
  std::wstring error;
  bool is_sym = false;
  int result =
      bazel::windows::IsSymlinkOrJunction(wpath.c_str(), &is_sym, &error);
  if (result == bazel::windows::IsSymlinkOrJunctionResult::kSuccess) {
    jboolean is_sym_jbool = is_sym ? JNI_TRUE : JNI_FALSE;
    env->SetBooleanArrayRegion(result_holder, 0, 1, &is_sym_jbool);
  } else {
    if (!error.empty() && CanReportError(env, error_msg_holder)) {
      ReportLastError(
          bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                           L"nativeIsJunction", wpath, error),
          env, error_msg_holder);
    }
  }
  return static_cast<jint>(result);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeGetLongPath(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray result_holder,
    jobjectArray error_msg_holder) {
  std::unique_ptr<WCHAR[]> result;
  std::wstring wpath(bazel::windows::GetJavaWstring(env, path));
  std::wstring error(bazel::windows::GetLongPath(wpath.c_str(), &result));
  if (!error.empty()) {
    if (CanReportError(env, error_msg_holder)) {
      ReportLastError(
          bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                           L"nativeGetLongPath", wpath, error),
          env, error_msg_holder);
    }
    return JNI_FALSE;
  }
  env->SetObjectArrayElement(
      result_holder, 0,
      env->NewString(reinterpret_cast<const jchar*>(result.get()),
                     wcslen(result.get())));
  return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeCreateJunction(
    JNIEnv* env, jclass clazz, jstring name, jstring target,
    jobjectArray error_msg_holder) {
  std::wstring wname(bazel::windows::GetJavaWstring(env, name));
  std::wstring wtarget(bazel::windows::GetJavaWstring(env, target));
  std::wstring error;
  int result = bazel::windows::CreateJunction(wname, wtarget, &error);
  if (result != bazel::windows::CreateJunctionResult::kSuccess &&
      !error.empty() && CanReportError(env, error_msg_holder)) {
    ReportLastError(bazel::windows::MakeErrorMessage(
                        WSTR(__FILE__), __LINE__, L"nativeCreateJunction",
                        wname + L", " + wtarget, error),
                    env, error_msg_holder);
  }
  return static_cast<jint>(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeCreateSymlink(
    JNIEnv* env, jclass clazz, jstring name, jstring target,
    jobjectArray error_msg_holder) {
  std::wstring wname(bazel::windows::GetJavaWstring(env, name));
  std::wstring wtarget(bazel::windows::GetJavaWstring(env, target));
  std::wstring error;
  int result = bazel::windows::CreateSymlink(wname, wtarget, &error);
  if (result != bazel::windows::CreateSymlinkResult::kSuccess &&
      !error.empty() && CanReportError(env, error_msg_holder)) {
    ReportLastError(bazel::windows::MakeErrorMessage(
                        WSTR(__FILE__), __LINE__, L"nativeCreateSymlink",
                        wname + L", " + wtarget, error),
                    env, error_msg_holder);
  }
  return static_cast<jint>(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeReadSymlinkOrJunction(
    JNIEnv* env, jclass clazz, jstring name, jobjectArray target_holder,
    jobjectArray error_msg_holder) {
  std::wstring wname(bazel::windows::GetJavaWstring(env, name));
  std::wstring target, error;
  int result = bazel::windows::ReadSymlinkOrJunction(wname, &target, &error);
  if (result == bazel::windows::ReadSymlinkOrJunctionResult::kSuccess) {
    env->SetObjectArrayElement(
        target_holder, 0,
        env->NewString(reinterpret_cast<const jchar*>(target.c_str()),
                       target.size()));
  } else {
    if (!error.empty() && CanReportError(env, error_msg_holder)) {
      ReportLastError(bazel::windows::MakeErrorMessage(
                          WSTR(__FILE__), __LINE__,
                          L"nativeReadSymlinkOrJunction", wname, error),
                      env, error_msg_holder);
    }
  }
  return static_cast<jint>(result);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeDeletePath(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray error_msg_holder) {
  std::wstring wpath(bazel::windows::GetJavaWstring(env, path));
  std::wstring error;
  int result = bazel::windows::DeletePath(wpath, &error);
  if (result != bazel::windows::DeletePathResult::kSuccess && !error.empty() &&
      CanReportError(env, error_msg_holder)) {
    ReportLastError(
        bazel::windows::MakeErrorMessage(WSTR(__FILE__), __LINE__,
                                         L"nativeDeletePath", wpath, error),
        env, error_msg_holder);
  }
  return result;
}
