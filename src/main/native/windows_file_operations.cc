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

#include <string>

#include "src/main/native/windows_error_handling.h"

// Keep in sync with j.c.g.devtools.build.lib.windows.WindowsFileOperations
enum {
  IS_JUNCTION_YES = 0,
  IS_JUNCTION_NO = 1,
  IS_JUNCTION_ERROR = 2,
};

// Determines whether `path` is a junction point or directory symlink.
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
int IsJunctionOrDirectorySymlink(const char* path) {
  DWORD attrs = GetFileAttributesA(path);
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

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsFileOperations_nativeIsJunction(
    JNIEnv* env, jclass clazz, jstring path, jobjectArray error_msg_holder) {
  const char* path_cstr = env->GetStringUTFChars(path, NULL);
  bool report_error =
      error_msg_holder != NULL && env->GetArrayLength(error_msg_holder) > 0;
  int result = IsJunctionOrDirectorySymlink(path_cstr);
  if (result == IS_JUNCTION_ERROR && report_error) {
    std::string error_str =
        GetLastErrorString(std::string("GetFileAttributesA(") +
                           std::string(path_cstr) + std::string(")"));
    jstring error_msg = env->NewStringUTF(error_str.c_str());
    env->SetObjectArrayElement(error_msg_holder, 0, error_msg);
  }

  env->ReleaseStringUTFChars(path, path_cstr);
  return result;
}
