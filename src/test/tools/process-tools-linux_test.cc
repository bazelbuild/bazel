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
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/main/tools/process-tools.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace {

class TerminateAndWaitForAllTest : public testing::Test {
  void SetUp(void) override {
    // TerminateAndWaitForAll requires the caller to have enabled the child
    // subreaper feature before spawning any processes.
    ASSERT_NE(prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0), -1);
  }
};

TEST_F(TerminateAndWaitForAllTest, TestOnlyLeader) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    setpgid(0, getpid());
    sleep(30);
    _exit(0);
  }
  setpgid(pid, pid);

  kill(pid, SIGKILL);  // Abort sleep to finish test quickly.
  ASSERT_NE(TerminateAndWaitForAll(pid), -1);
  ASSERT_EQ(waitpid(pid, nullptr, 0), -1);
}

TEST_F(TerminateAndWaitForAllTest, TestOutsideOfGroup) {
  int fds[2];
  ASSERT_NE(pipe(fds), -1);

  const size_t nprocs = 32;

  pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    setpgid(0, getpid());

    close(fds[0]);

    // Spawn a bunch of subprocesses and report their PIDs to the test before
    // exiting.
    for (size_t i = 0; i < nprocs; i++) {
      const pid_t subpid = fork();
      if (subpid == -1) {
        abort();
      } else if (subpid == 0) {
        close(fds[1]);

        // Keep some subprocesses in the process group and make others escape.
        if (i % 2 == 0) {
          setsid();
        }

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

  ASSERT_NE(waitpid(pid, nullptr, 0), -1);

  ASSERT_NE(TerminateAndWaitForAll(pid), -1);
  for (size_t i = 0; i < nprocs; i++) {
    // This check is racy: some other process might have reclaimed the PID of
    // the process we already terminated. But it's very unlikely because the
    // kernel tries very hard to not reassign PIDs too quickly.
    ASSERT_EQ(kill(pids[i], 0), -1);
  }
}

}  // namespace
