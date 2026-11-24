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

#ifndef BAZEL_SRC_MAIN_TOOLS_SECCOMP_EXEC_H_
#define BAZEL_SRC_MAIN_TOOLS_SECCOMP_EXEC_H_

// Installs a Linux seccomp-bpf filter that forbids child process creation
// (fork/vfork/execve/execveat/clone() without CLONE_THREAD) calls while
// still allowing thread creation via clone(CLONE_THREAD, ...), then executes
// "exe" with "argv".
//
// On success this function does not return. On error, it terminates the process
// via err(). On non-Linux platforms, it terminates the process via exit().
void ExecWithSeccompProcessBlocker(const char* exe, char* const* argv)
    __attribute__((noreturn));

#endif  // BAZEL_SRC_MAIN_TOOLS_SECCOMP_EXEC_H_
