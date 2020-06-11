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

#include <signal.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <memory>

#include "src/main/tools/process-tools.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace {

// Looks for the given process in the process table. Returns the entry if
// found and nullptr otherwise. Aborts on error.
std::unique_ptr<kinfo_proc> FindProcess(pid_t pid) {
  int name[] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
  std::unique_ptr<kinfo_proc> proc(new kinfo_proc);
  size_t len = sizeof(kinfo_proc);
  if (sysctl(name, 4, proc.get(), &len, nullptr, 0) == -1) {
    abort();
  }
  if (len == 0 || proc->kp_proc.p_pid == 0) {
    return nullptr;
  } else {
    if (proc->kp_proc.p_pid != pid) {
      // Did not expect to get a process with a PID we did not ask for.
      abort();
    }
    return proc;
  }
}

class WaitForProcessToTerminateTest : public testing::Test {};

TEST_F(WaitForProcessToTerminateTest, TestExit) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    _exit(42);
  }

  ASSERT_NE(WaitForProcessToTerminate(pid), -1);
  // The WaitForProcessToTerminate call guarantees that the process is done,
  // so we should not be able to affect its exit status any longer.
  kill(pid, SIGKILL);

  int status;
  ASSERT_NE(waitpid(pid, &status, 0), -1);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 42);
}

TEST_F(WaitForProcessToTerminateTest, TestSignal) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    sleep(30);
    _exit(0);
  }
  kill(pid, SIGTERM);

  ASSERT_NE(WaitForProcessToTerminate(pid), -1);
  // The WaitForProcessToTerminate call guarantees that the process is done,
  // so we should not be able to affect its exit status any longer.
  kill(pid, SIGKILL);

  int status;
  ASSERT_NE(waitpid(pid, &status, 0), -1);
  ASSERT_TRUE(WIFSIGNALED(status));
  ASSERT_EQ(WTERMSIG(status), SIGTERM);
}

class WaitForProcessGroupToTerminateTest : public testing::Test {};

TEST_F(WaitForProcessGroupToTerminateTest, TestOnlyLeader) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    setpgid(0, getpid());
    sleep(30);
    _exit(0);
  }
  setpgid(pid, pid);

  ASSERT_NE(WaitForProcessGroupToTerminate(pid), -1);
  kill(pid, SIGKILL);  // Abort sleep to finish test quickly.
  ASSERT_NE(waitpid(pid, nullptr, 0), -1);
}

TEST_F(WaitForProcessGroupToTerminateTest, TestManyProcesses) {
  int fds[2];
  ASSERT_NE(pipe(fds), -1);

  const size_t nprocs = 3;

  pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    setpgid(0, getpid());

    close(fds[0]);

    // Spawn a bunch of subprocesses in the same process group as the leader
    // and report their PIDs to the test before exiting.
    for (size_t i = 0; i < nprocs; i++) {
      const pid_t subpid = fork();
      if (subpid == -1) {
        abort();
      } else if (subpid == 0) {
        close(fds[1]);
        // Sleep for a very long amount of time to ensure we actually wait for
        // and terminate processes in the process group.
        sleep(10000);
        _exit(0);
      }
      if (write(fds[1], &subpid, sizeof(subpid)) != sizeof(subpid)) {
        abort();
      }
    }
    close(fds[1]);

    _exit(0);
  }
  setpgid(pid, pid);

  // Collect the PIDs of all subprocesses (except for the leader).
  close(fds[1]);
  pid_t pids[nprocs];
  for (size_t i = 0; i < nprocs; i++) {
    ASSERT_EQ(read(fds[0], &pids[i], sizeof(pids[i])), sizeof(pids[i]));
  }
  close(fds[0]);

  ASSERT_NE(WaitForProcessGroupToTerminate(pid), -1);
  // The process leader must still exist (as a zombie or not, we don't know)
  // but all other processes in the group must be gone by now.
  ASSERT_NE(FindProcess(pid), nullptr);
  for (size_t i = 0; i < nprocs; i++) {
    // This check is racy: some other process might have reclaimed the PID of
    // the process we already terminated. But it's very unlikely because the
    // kernel tries very hard to not reassign PIDs too quickly.
    ASSERT_EQ(FindProcess(pids[i]), nullptr);
  }

  ASSERT_NE(waitpid(pid, nullptr, 0), -1);
}

}  // namespace
