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
#ifndef BAZEL_SRC_MAIN_NATIVE_WINDOWS_PROCESS_H_
#define BAZEL_SRC_MAIN_NATIVE_WINDOWS_PROCESS_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <string>

#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

class WaitableProcess {
 public:
  WaitableProcess();
  bool Create(const std::wstring& argv0, const std::wstring& argv_rest,
              const std::wstring& cwd, void* env, HANDLE child_stdin,
              HANDLE child_stdout, HANDLE child_stderr, std::wstring* error);
  int Wait(long timeout_msec, std::wstring* error);
  bool Terminate(std::wstring* error);
  int GetExitCode(std::wstring* error);
  DWORD GetPid() const { return pid_; }

 private:
  WaitableProcess(const WaitableProcess&) = delete;
  WaitableProcess& operator=(const WaitableProcess&) = delete;

  AutoHandle process_, job_, ioport_;
  DWORD pid_, exit_code_;
};

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_PROCESS_H_
