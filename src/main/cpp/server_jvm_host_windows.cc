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

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/main/cpp/server_jvm_host.h"

namespace {

// The jli.dll functions that java.exe calls:
// https://github.com/openjdk/jdk27u/blob/815ff4dc327fe17f2433c7d115a5a503af10f3c4/src/java.base/share/native/libjli/java.h
// https://github.com/openjdk/jdk27u/blob/815ff4dc327fe17f2433c7d115a5a503af10f3c4/src/java.base/share/native/libjli/jli_util.h
struct StdArg {
  char* arg;
  jboolean has_wildcard;
};
typedef void(JNICALL* InitArgProcessingFn)(jboolean has_java_args,
                                           jboolean disable_argfile);
typedef void(JNICALL* CmdToArgsFn)(char* cmdline);
typedef int(JNICALL* GetStdArgcFn)();
typedef StdArg*(JNICALL* GetStdArgsFn)();
typedef int(JNICALL* LaunchFn)(int argc, char** argv, int jargc,
                               const char** jargv, int appclassc,
                               const char** appclassv,
                               const char* fullversion, const char* dotversion,
                               const char* pname, const char* lname,
                               jboolean javaargs, jboolean cpwildcard,
                               jboolean javaw, jint ergo);

template <typename T>
T GetFunction(HMODULE jli, const char* jli_path, const char* name) {
  T function = reinterpret_cast<T>(
      reinterpret_cast<void*>(GetProcAddress(jli, name)));
  if (function == nullptr) {
    fprintf(stderr, "Error: %s not found in %s (error %lu)\n", name, jli_path,
            GetLastError());
    exit(1);
  }
  return function;
}

}  // namespace

// Runs the server JVM like java.exe, see server_jvm_host.h.
int main(int argc, char** argv) {
  const char* javabase = getenv(blaze::kServerJvmHostJavabaseEnvVar);
  if (javabase == nullptr) {
    fprintf(stderr, "Error: %s is not set\n",
            blaze::kServerJvmHostJavabaseEnvVar);
    return 1;
  }
  // jli.dll depends on the C++ runtime DLLs in the same directory. java.exe
  // finds them because it is located in that directory.
  size_t jli_path_size = strlen(javabase) + sizeof("\\bin\\jli.dll");
  char* jli_path = static_cast<char*>(malloc(jli_path_size));
  snprintf(jli_path, jli_path_size, "%s\\bin\\jli.dll", javabase);
  // Neither the JVM nor the processes it starts should see the variable.
  _putenv_s(blaze::kServerJvmHostJavabaseEnvVar, "");

  HMODULE jli =
      LoadLibraryExA(jli_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (jli == nullptr) {
    fprintf(stderr, "Error: could not load %s (error %lu)\n", jli_path,
            GetLastError());
    return 1;
  }
  auto init_arg_processing =
      GetFunction<InitArgProcessingFn>(jli, jli_path, "JLI_InitArgProcessing");
  auto cmd_to_args = GetFunction<CmdToArgsFn>(jli, jli_path, "JLI_CmdToArgs");
  auto get_std_argc =
      GetFunction<GetStdArgcFn>(jli, jli_path, "JLI_GetStdArgc");
  auto get_std_args =
      GetFunction<GetStdArgsFn>(jli, jli_path, "JLI_GetStdArgs");
  auto launch = GetFunction<LaunchFn>(jli, jli_path, "JLI_Launch");

  // The rest is the Windows code path of java.exe's main function:
  // https://github.com/openjdk/jdk27u/blob/815ff4dc327fe17f2433c7d115a5a503af10f3c4/src/java.base/share/native/launcher/main.c
  init_arg_processing(/*has_java_args=*/JNI_FALSE,
                      /*disable_argfile=*/JNI_FALSE);

  if (getenv("_JAVA_LAUNCHER_DEBUG") != nullptr) {
    printf("Windows original main args:\n");
    for (int i = 0; i < argc; i++) {
      printf("wwwd_args[%d] = %s\n", i, argv[i]);
    }
  }

  // Obtain the command line in UTF-16, then convert it to the ANSI code page
  // without the "best-fit" option.
  LPWSTR wc_cmdline = GetCommandLineW();
  constexpr DWORD kFlags =
      WC_NO_BEST_FIT_CHARS | WC_COMPOSITECHECK | WC_DEFAULTCHAR;
  int mb_size = WideCharToMultiByte(CP_ACP, kFlags, wc_cmdline, -1, nullptr, 0,
                                    nullptr, nullptr);
  char* mb_cmdline = static_cast<char*>(malloc(mb_size));
  if (mb_cmdline == nullptr ||
      WideCharToMultiByte(CP_ACP, kFlags, wc_cmdline, -1, mb_cmdline, mb_size,
                          nullptr, nullptr) == 0) {
    perror("command line encoding conversion failure");
    exit(1);
  }
  cmd_to_args(mb_cmdline);
  free(mb_cmdline);

  int margc = get_std_argc();
  StdArg* stdargs = get_std_args();
  char** margv = static_cast<char**>(malloc((margc + 1) * sizeof(char*)));
  for (int i = 0; i < margc; i++) {
    margv[i] = stdargs[i].arg;
  }
  margv[margc] = nullptr;

  return launch(margc, margv, /*jargc=*/0, /*jargv=*/nullptr,
                /*appclassc=*/0, /*appclassv=*/nullptr, /*fullversion=*/"",
                /*dotversion=*/"0.0", /*pname=*/"java", /*lname=*/"openjdk",
                /*javaargs=*/JNI_FALSE, /*cpwildcard=*/JNI_TRUE,
                /*javaw=*/JNI_FALSE, /*ergo=*/0);
}
