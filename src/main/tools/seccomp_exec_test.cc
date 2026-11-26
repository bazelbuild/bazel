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

#include "src/main/tools/seccomp_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#include "googletest/include/gtest/gtest.h"

// Bazel's hermetic Linux toolchain compiles against Linux 3.9.11 kernel headers
// which predate execveat(2) (3.19).
#ifndef __NR_execveat
#if defined(__x86_64__)
#define __NR_execveat 322
#elif defined(__aarch64__)
#define __NR_execveat 281
#endif
#endif

namespace {

constexpr char kChildModeEnv[] = "SECCOMP_EXEC_TEST_CHILD_MODE";

void* ThreadFunc(void* arg) {
  auto* ran = static_cast<bool*>(arg);
  *ran = true;
  return nullptr;
}

int RunChildMode(const char* mode) {
  if (strcmp(mode, "thread") == 0) {
    bool ran = false;
    pthread_t thread;
    if (pthread_create(&thread, nullptr, ThreadFunc, &ran) != 0) {
      return 10;
    }
    if (pthread_join(thread, nullptr) != 0 || !ran) {
      return 11;
    }
    return 0;
  }
  if (strcmp(mode, "fork") == 0) {
    pid_t child = fork();
    if (child == 0) {
      _exit(20);
    }
    return (child == -1 && errno == EPERM) ? 0 : 20;
  }
  if (strcmp(mode, "posix_spawn") == 0) {
    pid_t child = -1;
    char* argv[] = {const_cast<char*>("/bin/true"), nullptr};
    int rc = posix_spawn(&child, "/bin/true", nullptr, nullptr, argv, nullptr);
    return rc == EPERM ? 0 : 21;
  }
  if (strcmp(mode, "second_execve") == 0) {
    char* argv[] = {const_cast<char*>("/bin/true"), nullptr};
    char* envp[] = {nullptr};
    execve("/bin/true", argv, envp);
    return errno == EPERM ? 0 : 22;
  }
  if (strcmp(mode, "second_execveat") == 0) {
    char* argv[] = {const_cast<char*>("/bin/true"), nullptr};
    char* envp[] = {nullptr};
    syscall(__NR_execveat, AT_FDCWD, "/bin/true", argv, envp, 0);
    return errno == EPERM ? 0 : 23;
  }
  return 99;
}

int SpawnSelfUnderSeccomp(const char* mode) {
  char self_path[4096];
  ssize_t len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
  if (len <= 0) {
    return -1;
  }
  self_path[len] = '\0';

  pid_t pid = fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    setenv(kChildModeEnv, mode, 1);
    char* argv[] = {self_path, nullptr};
    ExecWithSeccompProcessBlocker(self_path, argv);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      return -1;
    }
  }
  return status;
}

TEST(SeccompExecTest, AllowsInitialExecAndThreads) {
#if !defined(__linux__) || !(defined(__x86_64__) || defined(__aarch64__))
  GTEST_SKIP() << "Seccomp process blocker requires Linux x86_64/aarch64.";
#endif
  int status = SpawnSelfUnderSeccomp("thread");
  ASSERT_TRUE(WIFEXITED(status)) << "Killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SeccompExecTest, ReturnsEpermOnForkAfterExec) {
#if !defined(__linux__) || !(defined(__x86_64__) || defined(__aarch64__))
  GTEST_SKIP() << "Seccomp process blocker requires Linux x86_64/aarch64.";
#endif
  int status = SpawnSelfUnderSeccomp("fork");
  ASSERT_TRUE(WIFEXITED(status)) << "Killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SeccompExecTest, ReturnsEpermOnPosixSpawnAfterExec) {
#if !defined(__linux__) || !(defined(__x86_64__) || defined(__aarch64__))
  GTEST_SKIP() << "Seccomp process blocker requires Linux x86_64/aarch64.";
#endif
  int status = SpawnSelfUnderSeccomp("posix_spawn");
  ASSERT_TRUE(WIFEXITED(status)) << "Killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SeccompExecTest, ReturnsEpermOnExecveAfterExec) {
#if !defined(__linux__) || !(defined(__x86_64__) || defined(__aarch64__))
  GTEST_SKIP() << "Seccomp process blocker requires Linux x86_64/aarch64.";
#endif
  int status = SpawnSelfUnderSeccomp("second_execve");
  ASSERT_TRUE(WIFEXITED(status)) << "Killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(SeccompExecTest, ReturnsEpermOnExecveatAfterExec) {
#if !defined(__linux__) || !(defined(__x86_64__) || defined(__aarch64__))
  GTEST_SKIP() << "Seccomp process blocker requires Linux x86_64/aarch64.";
#endif
  int status = SpawnSelfUnderSeccomp("second_execveat");
  ASSERT_TRUE(WIFEXITED(status)) << "Killed by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

}  // namespace

int main(int argc, char** argv) {
  if (const char* mode = getenv(kChildModeEnv); mode != nullptr) {
    _exit(RunChildMode(mode));
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
