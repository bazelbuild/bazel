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

#include <stdint.h>

#include <string>

#include "src/main/native/windows/util.h"

namespace bazel {
namespace windows {

class WaitableProcess {
 public:
  static bool Create(
      const std::wstring& wpath, const std::wstring& argv_rest, void* env,
      const std::wstring& wcwd, HANDLE stdin_process, HANDLE stdout_process,
      HANDLE stderr_process, bazel::windows::AutoHandle* out_job,
      bazel::windows::AutoHandle* out_ioport,
      bazel::windows::AutoHandle* out_process, DWORD* out_pid,
      std::wstring* error);

  static int WaitFor(
      int64_t timeout_msec, DWORD pid, bazel::windows::AutoHandle* in_out_job,
      bazel::windows::AutoHandle* in_out_ioport,
      bazel::windows::AutoHandle* in_out_process, DWORD* out_exit_code,
      std::wstring* error);

  static int GetExitCode(
      const bazel::windows::AutoHandle& process, DWORD pid,
      DWORD* out_exit_code, std::wstring* error);

  static bool Terminate(
      const bazel::windows::AutoHandle& job,
      const bazel::windows::AutoHandle& process, DWORD pid,
      DWORD* out_exit_code, std::wstring* error);
};

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_PROCESS_H_
