// Copyright 2020 The Bazel Authors. All rights reserved.
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

// Starlark CPU profiler stubs for unsupported platforms.

#include <jni.h>
#include <stdlib.h>

namespace cpu_profiler {

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_syntax_CpuProfiler_supported(JNIEnv *env,
                                                                jclass clazz) {
  return false;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_devtools_build_lib_syntax_CpuProfiler_gettid(JNIEnv *env,
                                                             jclass clazz) {
  abort();
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_google_devtools_build_lib_syntax_CpuProfiler_createPipe(JNIEnv *env,
                                                                 jclass clazz) {
  abort();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_syntax_CpuProfiler_startTimer(
    JNIEnv *env, jclass clazz, jlong period_micros) {
  abort();
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_syntax_CpuProfiler_stopTimer(JNIEnv *env,
                                                                jclass clazz) {
  abort();
}

}  // namespace cpu_profiler
