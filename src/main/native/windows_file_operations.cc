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

#include "src/main/native/windows_util.h"

namespace windows_util {

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
enum {
  IS_JUNCTION_YES = 0,
  IS_JUNCTION_NO = 1,
  IS_JUNCTION_ERROR = 2,
};

// Determines whether `path` is a junction point or directory symlink.
//
// Uses the `GetFileAttributesW` WinAPI function.
// `path` should be a valid Windows-style or UNC path.
//
// To read about differences between junction points and directory symlinks,
// see http://superuser.com/a/343079.
//
// Returns:
// - IS_JUNCTION_YES, if `path` exists and is either a directory junction or a
//   directory symlink
// - IS_JUNCTION_NO, if `path` exists but is neither a directory junction nor a
//   directory symlink; also when `path` is a symlink to a directory but it was
//   created using "mklink" instead of "mklink /d", as such symlinks don't
//   behave the same way as directories (e.g. they can't be listed)
// - IS_JUNCTION_ERROR, if `path` doesn't exist or some error occurred
static int IsJunctionOrDirectorySymlink(const wchar_t* path) {
  DWORD attrs = GetFileAttributesW(path);
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

}  // namespace windows_util

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeIsJunction(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray error_msg_holder) {
  bool report_error =
      error_msg_holder != NULL && env->GetArrayLength(error_msg_holder) > 0;
  std::unique_ptr<wchar_t[]> long_path(
      windows_util::JstringToWstring(env, path));
  int result = windows_util::IsJunctionOrDirectorySymlink(long_path.get());
  if (result == windows_util::IS_JUNCTION_ERROR && report_error) {
    // Getting the string's characters again in UTF8 encoding is probably
    // easier than converting `long_path` using `wcstombs(3)`.
    const char* path_cstr = env->GetStringUTFChars(path, NULL);
    std::string error_str = windows_util::GetLastErrorString(
        std::string("GetFileAttributes(") + std::string(path_cstr) +
        std::string(")"));
    env->ReleaseStringUTFChars(path, path_cstr);
    jstring error_msg = env->NewStringUTF(error_str.c_str());
    env->SetObjectArrayElement(error_msg_holder, 0, error_msg);
  }
  return result;
}
