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

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_helloWorld(
    JNIEnv* env, jclass clazz, jint arg, jstring fruit) {
  char buf[512];
  const char* utf_fruit = env->GetStringUTFChars(fruit, NULL);
  snprintf(buf, sizeof(buf), "I have %d delicious %s fruits", arg, utf_fruit);
  jstring result = env->NewStringUTF(buf);
  env->ReleaseStringUTFChars(fruit, utf_fruit);
  return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_windows_WindowsProcesses_nativeGetpid(
    JNIEnv* env, jclass clazz) {
  return GetCurrentProcessId();
}
