// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <string.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/path.h"
#include "googletest/include/gtest/gtest.h"

namespace blaze {

// Test fixture for the UnlimitResources function.
//
// The test cases in this fixture are special because the setup forks a
// subprocess and the actual testing is supposed to happen in such subprocess.
// This is because resource limits are process-wide so we must ensure that our
// testing does not interfere with other tests in this fixture or with other
// tests in the whole test program.
//
// What this means is that each test case function must check if IsChild() is
// false first.  If it is, the function must return immediately.  If it is not,
// then the function can proceed to execute the test but care must be taken: the
// test function cannot use any of the gunit functions, nor it can use std::exit
// to terminate.  Instead, the function must use Die() to exit on a failure.
class UnlimitResourcesTest : public testing::Test {
 protected:
  UnlimitResourcesTest() {
    pid_ = fork();
    EXPECT_NE(-1, pid_);
  }

  virtual ~UnlimitResourcesTest() {
    if (IsChild()) {
      _exit(EXIT_SUCCESS);
    } else {
      int status;
      EXPECT_NE(-1, waitpid(pid_, &status, 0));
      EXPECT_TRUE(WIFEXITED(status));
      EXPECT_EQ(EXIT_SUCCESS, WEXITSTATUS(status));
    }
  }

  // Returns true if the test function is running in the child subprocess.
  bool IsChild() {
    return pid_ == 0;
  }

  // Description of the resource limits to test for.
  static struct limits_spec {
    const char* name;
    const int resource;
  } limits_[];

  // Aborts execution with the given message and fails the test case.
  // This can only be called when IsChild() is true.
  static void Die(const char* fmt, ...) ATTRIBUTE_NORETURN {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    _exit(EXIT_FAILURE);
  }

  // Version of getrlimit(3) that fails the test on error.
  // This can only be called when IsChild() is true.
  static struct rlimit GetrlimitOrDie(const int resource) {
    struct rlimit rl;
    if (getrlimit(resource, &rl) == -1) {
      Die("getrlimit(%d) failed: %s\n", resource, strerror(errno));
    }
    return rl;
  }

  // Version of setrlimit(3) that fails the test on error.
  // This can only be called when IsChild() is true.
  static void SetrlimitOrDie(const int resource, struct rlimit rl) {
    if (setrlimit(resource, &rl) == -1) {
      Die("setrlimit(%d) failed with cur=%" PRIdMAX ", max=%" PRIdMAX ": %s\n",
          resource, static_cast<intmax_t>(rl.rlim_cur),
          static_cast<intmax_t>(rl.rlim_max), strerror(errno));
    }
  }

 private:
  // PID of the test subprocess, or 0 if we are the subprocess.
  pid_t pid_;
};

struct UnlimitResourcesTest::limits_spec UnlimitResourcesTest::limits_[] = {
  { "RLIMIT_NOFILE", RLIMIT_NOFILE },
  { "RLIMIT_NPROC", RLIMIT_NPROC },
  { nullptr, 0 },
};

TEST_F(UnlimitResourcesTest, SuccessWithExplicitLimits) {
  if (!IsChild()) return;
  // The rest of this test runs in a subprocess.  See the fixture's docstring
  // for details on what this implies.

  // Lower the limits to very low values that should always work.
  for (struct limits_spec* limit = limits_; limit->name != nullptr; limit++) {
    struct rlimit rl = GetrlimitOrDie(limit->resource);
    rl.rlim_cur = 1;
    rl.rlim_max = 8;
    SetrlimitOrDie(limit->resource, rl);
  }

  if (!blaze::UnlimitResources()) {
    Die("UnlimitResources returned error; see output for diagnostics\n");
  }

  // Check that the soft limits were raised to the explicit hard limits we set.
  for (struct limits_spec* limit = limits_; limit->name != nullptr; limit++) {
    const struct rlimit rl = GetrlimitOrDie(limit->resource);
    if (rl.rlim_cur != rl.rlim_max) {
      Die("UnlimitResources did not increase the soft %s to its hard limit\n",
          limit->name);
    }
  }
}

TEST_F(UnlimitResourcesTest, SuccessWithPossiblyInfiniteLimits) {
  if (!IsChild()) return;
  // The rest of this test runs in a subprocess.  See the fixture's docstring
  // for details on what this implies.

  if (GetExplicitSystemLimit(-1) == -1) {
    fprintf(stderr, "GetExplicitSystemLimit not implemented for this platform; "
            "cannot verify the behavior of UnlimitResources\n");
    return;
  }

  // Lower only the soft limits to very low values and assume that the hard
  // limits are set to infinity; otherwise, there is nothing we can do because
  // we may not have permissions to increase them.
  for (struct limits_spec* limit = limits_; limit->name != nullptr; limit++) {
    struct rlimit rl = GetrlimitOrDie(limit->resource);
    if (rl.rlim_max != RLIM_INFINITY) {
      fprintf(stderr, "Hard resource limit for %s is not infinity; will not "
              "be able to meaningfully test anything\n", limit->name);
    }
    rl.rlim_cur = 1;
    SetrlimitOrDie(limit->resource, rl);
  }

  if (!blaze::UnlimitResources()) {
    Die("UnlimitResources returned error; see output for diagnostics\n");
  }

  // Check that the soft limits were increased to a higher explicit number.
  for (struct limits_spec* limit = limits_; limit->name != nullptr; limit++) {
    const struct rlimit rl = GetrlimitOrDie(limit->resource);
    if (rl.rlim_cur == 1 || rl.rlim_cur == RLIM_INFINITY) {
      Die("UnlimitResources did not increase the soft %s to the system limit\n",
          limit->name);
    }
  }
}

TEST_F(UnlimitResourcesTest, Coredumps) {
  if (!IsChild()) return;
  // The rest of this test runs in a subprocess.  See the fixture's docstring
  // for details on what this implies.

  // Lower only the soft limit to a very low value and assume that the hard
  // limit is non-zero.
  struct rlimit rl = GetrlimitOrDie(RLIMIT_CORE);
  if (rl.rlim_max <= 1) {
    fprintf(stderr,
            "Hard resource limit for RLIMIT_CORE is %" PRIuMAX
            "; cannot test anything meaningful\n",
            static_cast<uintmax_t>(rl.rlim_max));
    return;
  }
  rl.rlim_cur = 1;
  SetrlimitOrDie(RLIMIT_CORE, rl);

  if (!blaze::UnlimitCoredumps()) {
    Die("UnlimitCoredumps returned error; see output for diagnostics\n");
  }

  // Check that the soft limits were increased to a higher explicit number.
  rl = GetrlimitOrDie(RLIMIT_CORE);
  if (rl.rlim_cur == 1) {
    Die("UnlimitCoredumps did not increase the soft RLIMIT_CORE to the system "
        "limit\n");
  }
}

namespace {

void RunExecuteRunRequestAndCapture(const std::string& script_contents,
                                    const std::vector<std::string>& extra_args,
                                    bool clear_path,
                                    int* exit_code,
                                    std::string* captured_stdout) {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string tmpl =
      std::string(test_tmpdir != nullptr ? test_tmpdir : "/tmp") +
      "/exec_run_request_XXXXXX";
  std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
  tmpl_buf.push_back('\0');
  char* dir = mkdtemp(tmpl_buf.data());
  ASSERT_NE(dir, nullptr);

  std::string script_path = std::string(dir) + "/script.sh";
  {
    std::ofstream out(script_path);
    ASSERT_TRUE(out.good());
    out << script_contents;
  }
  ASSERT_EQ(chmod(script_path.c_str(), 0755), 0);

  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    close(pipefd[0]);
    if (dup2(pipefd[1], STDOUT_FILENO) == -1) {
      _exit(125);
    }
    close(pipefd[1]);
    if (clear_path) {
      setenv("PATH", "/non-existent", 1);
    }
    std::vector<std::string> args;
    args.push_back(script_path);
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    ExecuteRunRequest(blaze_util::Path(script_path), args);
  }

  close(pipefd[1]);
  captured_stdout->clear();
  char buf[256];
  ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
    captured_stdout->append(buf, n);
  }
  close(pipefd[0]);

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  *exit_code = WEXITSTATUS(status);

  unlink(script_path.c_str());
  rmdir(dir);
}

}  // namespace

TEST(ExecuteRunRequestTest, ExecutesShebangScriptDirectly) {
  int exit_code = -1;
  std::string output;
  RunExecuteRunRequestAndCapture(
      "#!/bin/sh\nprintf 'shebang:%s:%s\n' \"$1\" \"$2\"\n",
      {"arg1", "arg with spaces"},
      /* clear_path= */ true,
      &exit_code,
      &output);
  EXPECT_EQ(exit_code, 0);
  EXPECT_EQ(output, "shebang:arg1:arg with spaces\n");
}

TEST(ExecuteRunRequestTest, FallsBackOnEnoexecForShebanglessScript) {
  int exit_code = -1;
  std::string output;
  RunExecuteRunRequestAndCapture(
      "printf 'enoexec:%s:%s\n' \"$1\" \"$2\"\n",
      {"arg1", "arg with spaces"},
      /* clear_path= */ false,
      &exit_code,
      &output);
  EXPECT_EQ(exit_code, 0);
  EXPECT_EQ(output, "enoexec:arg1:arg with spaces\n");
}

TEST(ExecuteRunRequestTest, FallsBackToBinShOnEnoexecWhenPathHasNoBash) {
  int exit_code = -1;
  std::string output;
  RunExecuteRunRequestAndCapture(
      "printf 'binsh:%s:%s\n' \"$1\" \"$2\"\n",
      {"arg1", "arg with spaces"},
      /* clear_path= */ true,
      &exit_code,
      &output);
  EXPECT_EQ(exit_code, 0);
  EXPECT_EQ(output, "binsh:arg1:arg with spaces\n");
}

}  // namespace blaze
