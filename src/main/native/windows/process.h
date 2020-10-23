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
  // These are the possible return values from the WaitFor() method.
  enum {
    kWaitSuccess = 0,
    kWaitTimeout = 1,
    kWaitError = 2,
  };

  WaitableProcess() : pid_(0), exit_code_(STILL_ACTIVE) {}

  bool Create(const std::wstring& argv0, const std::wstring& argv_rest,
              void* env, const std::wstring& wcwd, std::wstring* error);

  bool Create(const std::wstring& argv0, const std::wstring& argv_rest,
              void* env, const std::wstring& wcwd, HANDLE stdin_process,
              HANDLE stdout_process, HANDLE stderr_process,
              LARGE_INTEGER* opt_out_start_time, std::wstring* error);

  int WaitFor(int64_t timeout_msec, LARGE_INTEGER* opt_out_end_time,
              std::wstring* error);

  int GetExitCode(std::wstring* error);

  bool Terminate(std::wstring* error);

  DWORD GetPid() const { return pid_; }

 private:
  bool Create(const std::wstring& argv0, const std::wstring& argv_rest,
              void* env, const std::wstring& wcwd, HANDLE stdin_process,
              HANDLE stdout_process, HANDLE stderr_process,
              LARGE_INTEGER* opt_out_start_time, bool create_window,
              bool handle_signals, std::wstring* error);

  AutoHandle process_, job_, ioport_;
  DWORD pid_, exit_code_;
};

// Escape a command line argument using Windows escaping syntax.
//
// This escaping lets us safely pass arguments to subprocesses created with
// CreateProcessW. (The escaping rules are a bit complex, look at the function
// implementation.)
std::wstring WindowsEscapeArg(const std::wstring& arg);

}  // namespace windows
}  // namespace bazel

#endif  // BAZEL_SRC_MAIN_NATIVE_WINDOWS_PROCESS_H_
