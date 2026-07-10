// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include <string>

#include "src/main/native/jni.h"
#include "src/main/native/windows/jni-util.h"

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_windows_WindowsSemaphore_createSemaphore0(
    JNIEnv* env, jclass clazz, jstring name, jint initial_count, jint max_count) {
  std::wstring wname(bazel::windows::GetJavaWstring(env, name));
  HANDLE handle = CreateSemaphoreW(nullptr, initial_count, max_count, wname.c_str());
  return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsSemaphore_release0(
    JNIEnv* env, jclass clazz, jlong handle, jint delta) {
  return ReleaseSemaphore(reinterpret_cast<HANDLE>(handle), delta, nullptr)
             ? JNI_TRUE
             : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_windows_WindowsSemaphore_tryAcquire(
    JNIEnv* env, jclass clazz, jlong handle) {
  DWORD result = WaitForSingleObject(reinterpret_cast<HANDLE>(handle), 0);
  return result == WAIT_OBJECT_0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_windows_WindowsSemaphore_close(JNIEnv* env, jclass clazz,
                                                                  jlong handle) {
  CloseHandle(reinterpret_cast<HANDLE>(handle));
}
