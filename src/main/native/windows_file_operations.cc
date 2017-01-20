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
#include <windows.h>

#include <memory>
#include <string>
#include <type_traits>  // static_assert

#include "src/main/native/windows_file_operations.h"
#include "src/main/native/windows_util.h"

namespace windows_util {

using std::unique_ptr;

// Ensure we can safely cast (const) jchar* to LP(C)WSTR.
// This is true with MSVC but usually not with GCC.
static_assert(sizeof(jchar) == sizeof(WCHAR),
              "jchar and WCHAR should be the same size");

int IsJunctionOrDirectorySymlink(const WCHAR* path) {
  DWORD attrs = ::GetFileAttributesW(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return IS_JUNCTION_ERROR;
  } else {
    if ((attrs & FILE_ATTRIBUTE_DIRECTORY) &&
        (attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
      return IS_JUNCTION_YES;
    } else {
      return IS_JUNCTION_NO;
    }
  }
}

bool GetLongPath(const WCHAR* path, unique_ptr<WCHAR[]>* result) {
  DWORD size = ::GetLongPathNameW(path, NULL, 0);
  if (size == 0) {
    return false;
  }
  result->reset(new WCHAR[size]);
  ::GetLongPathNameW(path, result->get(), size);
  return true;
}

static void MaybeReportLastError(string reason, JNIEnv* env,
                                 jobjectArray error_msg_holder) {
  if (error_msg_holder != nullptr &&
      env->GetArrayLength(error_msg_holder) > 0) {
    std::string error_str = windows_util::GetLastErrorString(reason);
    jstring error_msg = env->NewStringUTF(error_str.c_str());
    env->SetObjectArrayElement(error_msg_holder, 0, error_msg);
  }
}

}  // namespace windows_util

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeIsJunction(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray error_msg_holder) {
  bool report_error =
      error_msg_holder != nullptr && env->GetArrayLength(error_msg_holder) > 0;
  const jchar* wpath = env->GetStringChars(path, nullptr);
  int result = windows_util::IsJunctionOrDirectorySymlink((LPCWSTR)wpath);
  env->ReleaseStringChars(path, wpath);
  if (result == windows_util::IS_JUNCTION_ERROR && report_error) {
    // Getting the string's characters again in UTF8 encoding is
    // easier than converting `wpath` using `wcstombs(3)`.
    const char* path_cstr = env->GetStringUTFChars(path, nullptr);
    windows_util::MaybeReportLastError(
        std::string("GetFileAttributes(") + path_cstr + ")", env,
        error_msg_holder);
    env->ReleaseStringUTFChars(path, path_cstr);
  }
  return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeGetLongPath(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray result_holder,
    jobjectArray error_msg_holder) {
  const jchar* cpath = nullptr;
  cpath = env->GetStringChars(path, nullptr);
  std::unique_ptr<WCHAR[]> result;
  bool success =
      windows_util::GetLongPath(reinterpret_cast<const WCHAR*>(cpath), &result);
  env->ReleaseStringChars(path, cpath);
  if (!success) {
    const char* path_cstr = env->GetStringUTFChars(path, nullptr);
    windows_util::MaybeReportLastError(
        std::string("GetLongPathName(") + path_cstr + ")", env,
        error_msg_holder);
    env->ReleaseStringUTFChars(path, path_cstr);
    return JNI_FALSE;
  }
  std::wstring wresult(result.get());
  env->SetObjectArrayElement(
      result_holder, 0,
      env->NewString(reinterpret_cast<const jchar*>(wresult.c_str()),
                     wresult.size()));
  return JNI_TRUE;
}
