// Copyright 2018 The Bazel Authors. All rights reserved.
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

// Mock C++ binary, only used in tests.

#ifdef IS_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif  // IS_WINDOWS

#include "tools/cpp/runfiles/runfiles.h"

#ifdef IS_WINDOWS
#include <windows.h>
#else  // not IS_WINDOWS
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif  // IS_WINDOWS

#include <string.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

using bazel::tools::cpp::runfiles::Runfiles;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unique_ptr;

string child_binary_name(const char* lang) {
#ifdef IS_WINDOWS
  return string("foo_ws/bar/bar-") + lang + ".exe";
#else
  return string("foo_ws/bar/bar-") + lang;
#endif  // IS_WINDOWS
}

bool is_file(const string& path) {
  if (path.empty()) {
    return false;
  }
  return ifstream(path).is_open();
}

#ifdef IS_WINDOWS
unique_ptr<char[]> create_env_block(const Runfiles& runfiles) {
  char systemroot[MAX_PATH];
  DWORD systemroot_size =
      GetEnvironmentVariable("SYSTEMROOT", systemroot, MAX_PATH);
  if (!systemroot_size || systemroot_size == MAX_PATH) {
    cerr << "ERROR[" << __FILE__ << "]: %SYSTEMROOT% is too long" << endl;
    return std::move(unique_ptr<char[]>());
  }

  size_t total_envblock_size =
      10 /* the string "SYSTEMROOT" */ + 1 /* equals-sign */ + systemroot_size +
      1 /* null-terminator */ +
      2 /* two null-terminator at the end of the environment block */;

  // count total size of the environment block
  const auto envvars = runfiles.EnvVars();
  for (const auto i : envvars) {
    total_envblock_size += i.first.size() + 1 /* equals-sign */ +
                           i.second.size() + 1 /* null-terminator */;
  }

  // copy environment variables from `envvars`
  unique_ptr<char[]> result(new char[total_envblock_size]);
  char* p = result.get();
  for (const auto i : envvars) {
    strncpy(p, i.first.c_str(), i.first.size());
    p += i.first.size();
    *p++ = '=';
    strncpy(p, i.second.c_str(), i.second.size());
    p += i.second.size();
    *p++ = '\0';
  }

  // SYSTEMROOT environment variable
  strncpy(p, "SYSTEMROOT=", 11);
  p += 11;
  strncpy(p, systemroot, systemroot_size);
  p += systemroot_size;
  *p++ = '\0';

  // final two null-terminators
  p[0] = '\0';
  p[1] = '\0';

  return std::move(result);
}
#endif

int _main(int argc, char** argv) {
  cout << "Hello C++ Foo!" << endl;
  string error;
  unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
  if (runfiles == nullptr) {
    cerr << "ERROR[" << __FILE__ << "]: " << error << endl;
    return 1;
  }
  string path = runfiles->Rlocation("foo_ws/foo/datadep/hello.txt");
  if (!is_file(path)) {
    return 1;
  }
  cout << "rloc=" << path << endl;

#ifdef IS_WINDOWS
  auto envvars = create_env_block(*runfiles);
#else
  const auto envvars = runfiles->EnvVars();
#endif

  // Run a subprocess, propagate the runfiles envvar to it. The subprocess will
  // use this process's runfiles manifest or runfiles directory.
  for (const char* lang : {"py", "java", "sh", "cc"}) {
    const string bar = runfiles->Rlocation(child_binary_name(lang));

    unique_ptr<char[]> argv0(new char[bar.size() + 1]);
    strncpy(argv0.get(), bar.c_str(), bar.size());
    argv0.get()[bar.size()] = 0;

#ifdef IS_WINDOWS
    PROCESS_INFORMATION processInfo;
    STARTUPINFOA startupInfo = {0};
    BOOL ok =
        CreateProcessA(nullptr, argv0.get(), nullptr, nullptr, FALSE, 0,
                       envvars.get(), nullptr, &startupInfo, &processInfo);
    if (!ok) {
      DWORD err = GetLastError();
      fprintf(stderr, "ERROR: CreateProcessA error: %d\n", err);
      return 1;
    }
    WaitForSingleObject(processInfo.hProcess, INFINITE);
    CloseHandle(processInfo.hProcess);
    CloseHandle(processInfo.hThread);
#else
    char* args[2] = {argv0.get(), nullptr};
    pid_t child = fork();
    if (child) {
      int status;
      waitpid(child, &status, 0);
    } else {
      for (const auto i : envvars) {
        setenv(i.first.c_str(), i.second.c_str(), 1);
      }
      execv(args[0], args);
    }
#endif
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return _main(argc, argv); }
