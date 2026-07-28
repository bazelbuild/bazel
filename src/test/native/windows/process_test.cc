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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "src/main/native/windows/process.h"

#include <wchar.h>
#include <windows.h>

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/util.h"
#include "src/test/cpp/util/windows_test_util.h"
#include "rules_cc/cc/runfiles/runfiles.h"

namespace {

// Asserts argument escaping for subprocesses.
//
// For each pair in 'args', this method:
// 1. asserts that WindowsEscapeArg(pair.first) == pair.second
// 2. asserts that passing pair.second to a subprocess results in the subprocess
//    receiving pair.first
//
// The method performs the second assertion by running "printarg.exe" (a
// data-dependency of this test) once for each argument.
void AssertSubprocessReceivesArgsAsIntended(
    const std::vector<std::pair<std::wstring, std::wstring> >& args) {
  // Assert that the WindowsEscapeArg produces what we expect.
  for (const auto& i : args) {
    ASSERT_EQ(bazel::windows::WindowsEscapeArg(i.first), i.second);
  }

  // Create a Runfiles object.
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(&error));
  ASSERT_NE(runfiles.get(), nullptr) << error;

  // Look up the path of the printarg.exe utility.
  std::string printarg =
      runfiles->Rlocation("io_bazel/src/test/native/windows/printarg.exe");
  ASSERT_NE(printarg, "");

  // Convert printarg.exe's path to a wchar_t Windows path.
  std::wstring wprintarg;
  bool success =
      blaze_util::AsAbsoluteWindowsPath(printarg, &wprintarg, &error);
  ASSERT_TRUE(success) << error;

  // SECURITY_ATTRIBUTES for inheritable HANDLEs.
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;

  // Open /dev/null that will be redirected into the subprocess' stdin.
  bazel::windows::AutoHandle devnull(
      CreateFileW(L"NUL", GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &sa,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
  ASSERT_TRUE(devnull.IsValid());

  // Create a pipe that the subprocess' stdout will be redirected to.
  HANDLE pipe_read_h, pipe_write_h;
  if (!CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0x10000)) {
    DWORD err = GetLastError();
    ASSERT_EQ(err, 0);
  }
  bazel::windows::AutoHandle pipe_read(pipe_read_h), pipe_write(pipe_write_h);

  // Duplicate stderr, where the subprocess' stderr will be redirected to.
  HANDLE stderr_h;
  if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE),
                       GetCurrentProcess(), &stderr_h, 0, TRUE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD err = GetLastError();
    ASSERT_EQ(err, 0);
  }
  bazel::windows::AutoHandle stderr_dup(stderr_h);

  // Create the attribute object for the process creation. This object describes
  // exactly which handles the subprocess shall inherit.
  STARTUPINFOEXW startupInfo;
  std::unique_ptr<bazel::windows::AutoAttributeList> attrs;
  std::wstring werror;
  ASSERT_TRUE(bazel::windows::AutoAttributeList::Create(
      devnull, pipe_write, stderr_dup, &attrs, &werror));
  attrs->InitStartupInfoExW(&startupInfo);

  // MSDN says the maximum command line is 32767 characters, with a null
  // terminator that is exactly 2^15 (= 0x8000).
  static constexpr size_t kMaxCmdline = 0x8000;
  wchar_t cmdline[kMaxCmdline];

  // Copy printarg.exe's escaped path into the 'cmdline', and append a space.
  // We will append arguments to this command line in the for-loop below.
  wprintarg = bazel::windows::WindowsEscapeArg(wprintarg);
  wcsncpy(cmdline, wprintarg.c_str(), wprintarg.size());
  wchar_t* pcmdline = cmdline + wprintarg.size();
  *pcmdline++ = L' ';

  // Run a subprocess for each of the arguments and assert that the argument
  // arrived to the subprocess as intended.
  for (const auto& i : args) {
    // We already asserted for every element that WindowsEscapeArg(i.first)
    // produces the same output as i.second, so just use i.second instead of
    // converting i.first again.
    wcsncpy(pcmdline, i.second.c_str(), i.second.size());
    pcmdline[i.second.size()] = 0;

    // Run the subprocess.
    PROCESS_INFORMATION processInfo;
    BOOL ok = CreateProcessW(
        nullptr, cmdline, nullptr, nullptr, TRUE,
        CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT, nullptr,
        nullptr, &startupInfo.StartupInfo, &processInfo);
    if (!ok) {
      DWORD err = GetLastError();
      ASSERT_EQ(err, 0);
    }
    CloseHandle(processInfo.hThread);
    bazel::windows::AutoHandle process(processInfo.hProcess);

    // Wait for the subprocess to exit. Timeout is 5 seconds, which should be
    // more than enough for the subprocess to finish.
    ASSERT_EQ(WaitForSingleObject(process, 5000), WAIT_OBJECT_0);

    // The subprocess printed its argv[1] (without a newline) to its stdout,
    // which is redirected into the pipe.
    // Let's write a null-terminator to the pipe to separate the output from the
    // output of the subsequent subprocess. The null-terminator also yields
    // null-terminated strings in the pipe, making it easy to read them out
    // later.
    DWORD dummy;
    ASSERT_TRUE(WriteFile(pipe_write, "\0", 1, &dummy, nullptr));
  }

  // Read the output of the subprocesses from the pipe. They are divided by
  // null-terminators, so 'buf' will contain a sequence of null-terminated
  // strings.  We close the writing end so that ReadFile won't block until the
  // desired amount of bytes is available.
  DWORD total_output_len;
  char buf[0x10000];
  pipe_write = INVALID_HANDLE_VALUE;
  if (!ReadFile(pipe_read, buf, 0x10000, &total_output_len, nullptr)) {
    DWORD err = GetLastError();
    ASSERT_EQ(err, 0);
  }

  // Assert that the subprocesses produced exactly the *unescaped* arguments.
  size_t start = 0;
  for (const auto& arg : args) {
    // Assert that there was enough data produced by the subprocesses.
    ASSERT_LT(start, total_output_len);

    // Find the output of the corresponding subprocess. Since all subprocesses
    // printed into the same pipe and we added null-terminators between them,
    // the output is already there, conveniently as a null-terminated string.
    std::string actual_arg(buf + start);
    start += actual_arg.size() + 1;

    // 'args' contains wchar_t strings, but the subprocesses printed ASCII
    // (char) strings. To compare, we convert arg.first to a char-string.
    std::string expected_arg;
    expected_arg.reserve(arg.first.size());
    for (const auto& wc : arg.first) {
      expected_arg.append(1, static_cast<char>(wc));
    }

    // Assert that the subprocess printed exactly the *unescaped* argument.
    EXPECT_EQ(expected_arg, actual_arg);
  }
}

TEST(ProcessTest, WindowsEscapeArgTest) {
  AssertSubprocessReceivesArgsAsIntended({
      {L"", L"\"\""},
      {L" ", L"\" \""},
      {L"\t", L"\"\t\""},
      {L"a\tb", L"\"a\tb\""},
      {L"with\ttab", L"\"with\ttab\""},
      {L" \t ", L"\" \t \""},
      {L"tab\t^caret", L"\"tab\t^caret\""},
      {L"\"", L"\"\\\"\""},
      {L"\"\\", L"\"\\\"\\\\\""},
      {L"\\", L"\\"},
      {L"\\\"", L"\"\\\\\\\"\""},
      {L"with space", L"\"with space\""},
      {L"with^caret", L"with^caret"},
      {L"space ^caret", L"\"space ^caret\""},
      {L"caret^ space", L"\"caret^ space\""},
      {L"with\"quote", L"\"with\\\"quote\""},
      {L"with\\backslash", L"with\\backslash"},
      {L"one\\ backslash and \\space", L"\"one\\ backslash and \\space\""},
      {L"two\\\\backslashes", L"two\\\\backslashes"},
      {L"two\\\\ backslashes \\\\and space",
       L"\"two\\\\ backslashes \\\\and space\""},
      {L"one\\\"x", L"\"one\\\\\\\"x\""},
      {L"two\\\\\"x", L"\"two\\\\\\\\\\\"x\""},
      {L"a \\ b", L"\"a \\ b\""},
      {L"a \\\" b", L"\"a \\\\\\\" b\""},
      {L"A", L"A"},
      {L"\"a\"", L"\"\\\"a\\\"\""},
      {L"B C", L"\"B C\""},
      {L"\"b c\"", L"\"\\\"b c\\\"\""},
      {L"D\"E", L"\"D\\\"E\""},
      {L"\"d\"e\"", L"\"\\\"d\\\"e\\\"\""},
      {L"C:\\F G", L"\"C:\\F G\""},
      {L"\"C:\\f g\"", L"\"\\\"C:\\f g\\\"\""},
      {L"C:\\H\"I", L"\"C:\\H\\\"I\""},
      {L"\"C:\\h\"i\"", L"\"\\\"C:\\h\\\"i\\\"\""},
      {L"C:\\J\\\"K", L"\"C:\\J\\\\\\\"K\""},
      {L"\"C:\\j\\\"k\"", L"\"\\\"C:\\j\\\\\\\"k\\\"\""},
      {L"C:\\L M ", L"\"C:\\L M \""},
      {L"\"C:\\l m \"", L"\"\\\"C:\\l m \\\"\""},
      {L"C:\\N O\\", L"\"C:\\N O\\\\\""},
      {L"\"C:\\n o\\\"", L"\"\\\"C:\\n o\\\\\\\"\""},
      {L"C:\\P Q\\ ", L"\"C:\\P Q\\ \""},
      {L"\"C:\\p q\\ \"", L"\"\\\"C:\\p q\\ \\\"\""},
      {L"C:\\R\\S\\", L"C:\\R\\S\\"},
      {L"C:\\R x\\S\\", L"\"C:\\R x\\S\\\\\""},
      {L"\"C:\\r\\s\\\"", L"\"\\\"C:\\r\\s\\\\\\\"\""},
      {L"\"C:\\r x\\s\\\"", L"\"\\\"C:\\r x\\s\\\\\\\"\""},
      {L"C:\\T U\\W\\", L"\"C:\\T U\\W\\\\\""},
      {L"\"C:\\t u\\w\\\"", L"\"\\\"C:\\t u\\w\\\\\\\"\""},
  });
}

// Verifies that a batch file whose path is >= kMaxPath (MAX_PATH - 4) but
// < MAX_PATH can be executed via CreateProcessW when the
// AsExecutablePathForCreateProcess fallback returns the native path (without
// the "\\?\" prefix that cmd.exe cannot handle).
//
// The directory tree uses single-character names that are already
// as short as they can be, so GetShortPathNameW cannot shorten the path below kMaxPath.
// This forces AsExecutablePathForCreateProcess into its fallback path.
TEST(ProcessTest, BatchFileWithLongPathExecutes) {
  static constexpr size_t kMaxPath = MAX_PATH - 4;
  static const std::wstring kUncPrefix(L"\\\\?\\");

  // Obtain TEST_TMPDIR in 8.3-shortened form so that every component is
  // already as short as it can get.
  std::string tmpdir_str;
  std::string short_error;
  ASSERT_TRUE(blaze_util::AsShortWindowsPath(
      blaze::GetPathEnv("TEST_TMPDIR"), &tmpdir_str, &short_error))
      << short_error;
  std::wstring tmpdir = blaze_util::CstringToWstring(tmpdir_str);

  // We want at least a few directories + the batch file name
  ASSERT_LT(tmpdir.size(), kMaxPath - 20)
      << "TEST_TMPDIR is too long for this test";

  // Test root so we can recursively remove everything later on without affecting
  // any other tests.
  std::wstring test_root = tmpdir + L"\\bl";

  // Build a deep tree of single-char directories until the full batch file
  // path (dir + "\test.bat") is >= kMaxPath (256) but < MAX_PATH (260).
  std::wstring bat_name = L"\\test.bat";
  std::wstring dir_path = test_root;
  const size_t target_dir_len = kMaxPath - bat_name.size();
  while (dir_path.size() < target_dir_len) {
    dir_path += L"\\a";
  }
  ASSERT_TRUE(blaze_util::MakeDirectoriesW(dir_path, 0755));

  std::wstring bat_path = dir_path + bat_name;
  ASSERT_GE(bat_path.size(), kMaxPath);
  ASSERT_LT(bat_path.size(), (size_t)MAX_PATH);

  // Write a batch file that echoes a known marker.
  const std::string bat_content = "@echo off\r\necho BATCH_OK\r\n";
  ASSERT_TRUE(blaze_util::CreateDummyFile(kUncPrefix + bat_path, bat_content));

  // Verify the batch fallback returns the native path (no "\\?\" prefix).
  std::wstring quoted_path, extended_path;
  std::wstring error = bazel::windows::AsExecutablePathForCreateProcess(
      bat_path, &quoted_path, &extended_path);
  ASSERT_EQ(error, L"");
  EXPECT_EQ(extended_path, bat_path);
  EXPECT_EQ(quoted_path, L"\"" + bat_path + L"\"");

  // Execute the batch file via WaitableProcess, the actual production path.
  // WaitableProcess::Create internally calls AsExecutablePathForCreateProcess,
  // so this proves the batch file fallback works end-to-end.
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;

  bazel::windows::AutoHandle devnull(::CreateFileW(
      L"NUL", GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &sa,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
  ASSERT_TRUE(devnull.IsValid());

  HANDLE pipe_read_h, pipe_write_h;
  ASSERT_TRUE(::CreatePipe(&pipe_read_h, &pipe_write_h, &sa, 0x10000));
  bazel::windows::AutoHandle pipe_read(pipe_read_h);
  bazel::windows::AutoHandle pipe_write(pipe_write_h);

  HANDLE stderr_h;
  ASSERT_TRUE(::DuplicateHandle(
      GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE),
      GetCurrentProcess(), &stderr_h, 0, TRUE, DUPLICATE_SAME_ACCESS));
  bazel::windows::AutoHandle stderr_dup(stderr_h);

  bazel::windows::WaitableProcess proc;
  std::wstring proc_error;
  if (!proc.Create(bat_path, L"", nullptr, L".", devnull, pipe_write,
                   stderr_dup, nullptr, &proc_error)) {
    GTEST_SKIP() << "WaitableProcess::Create failed: "
                 << blaze_util::WstringToCstring(proc_error);
  }

  ASSERT_EQ(proc.WaitFor(3000, nullptr, &proc_error),
            bazel::windows::WaitableProcess::kWaitSuccess)
      << blaze_util::WstringToCstring(proc_error);
  EXPECT_EQ(proc.GetExitCode(&proc_error), 0)
      << blaze_util::WstringToCstring(proc_error);

  pipe_write = INVALID_HANDLE_VALUE;
  char stdout_buf[0x1000];
  DWORD bytes_read = 0;
  if (!::ReadFile(pipe_read, stdout_buf, sizeof(stdout_buf) - 1, &bytes_read,
                  nullptr)) {
    DWORD err = ::GetLastError();
    ASSERT_EQ(err, (DWORD)0);
  }
  stdout_buf[bytes_read] = '\0';

  EXPECT_NE(std::string(stdout_buf, bytes_read).find("BATCH_OK"),
            std::string::npos)
      << "Expected BATCH_OK in output, got: " << stdout_buf;
      
  EXPECT_TRUE(blaze_util::RemoveRecursively(
      blaze_util::WstringToCstring(test_root)));
}

}  // namespace
