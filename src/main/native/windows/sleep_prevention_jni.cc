// Copyright 2019 The Bazel Authors. All rights reserved.
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

#include <assert.h>

#include <atomic>

#include "src/main/native/jni.h"

// Keep track of our pushes and pops of sleep state.
static std::atomic_int g_sleep_stack{0};

/*
 * Class:     com_google_devtools_build_lib_platform_SleepPreventionModule_SleepPrevention
 * Method:    pushDisableSleep
 * Signature: ()I
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_platform_SleepPreventionModule_00024SleepPrevention_pushDisableSleep(
    JNIEnv *, jclass) {
  if (g_sleep_stack++ == 0) {
      SetThreadExecutionState(
          ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);
  }
  return 0;
}

/*
 * Class:     com.google.devtools.build.lib.platform.SleepPreventionModule.SleepPrevention
 * Method:    popDisableSleep
 * Signature: ()I
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_platform_SleepPreventionModule_00024SleepPrevention_popDisableSleep(
    JNIEnv *, jclass) {
  int stack_value = --g_sleep_stack;
  assert(stack_value >= 0);
  if (stack_value == 0) {
    SetThreadExecutionState(ES_CONTINUOUS);
  }
  return 0;
}
