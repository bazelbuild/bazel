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

#include <windows.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/util.h"
#include "src/tools/launcher/util/launcher_util.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace bazel {
namespace launcher {

using bazel::tools::cpp::runfiles::Runfiles;
using std::getenv;
using std::ios;
using std::ofstream;
using std::string;
using std::wstring;

class LaunchUtilTest : public ::testing::Test {
 protected:
  LaunchUtilTest() {}

  virtual ~LaunchUtilTest() {}

  void SetUp() override {
    char* tmpdir = getenv("TEST_TMPDIR");
    if (tmpdir != NULL) {
      test_tmpdir = blaze_util::CstringToWstring(string(tmpdir));
    } else {
      tmpdir = getenv("TEMP");
      ASSERT_FALSE(tmpdir == NULL);
      test_tmpdir = blaze_util::CstringToWstring(string(tmpdir));
    }
  }

  void TearDown() override {}

  wstring GetTmpDir() { return this->test_tmpdir; }

  // Create an empty file at path
  static void CreateEmptyFile(const wstring& path) {
    ofstream file_stream(path.c_str(), ios::out | ios::binary);
    file_stream.put('\0');
  }

 private:
  wstring test_tmpdir;
};

TEST_F(LaunchUtilTest, GetBinaryPathWithoutExtensionTest) {
  ASSERT_EQ(L"foo", GetBinaryPathWithoutExtension(L"foo.exe"));
  ASSERT_EQ(L"foo.sh", GetBinaryPathWithoutExtension(L"foo.sh.exe"));
  ASSERT_EQ(L"foo.sh", GetBinaryPathWithoutExtension(L"foo.sh"));
}

TEST_F(LaunchUtilTest, GetBinaryPathWithExtensionTest) {
  ASSERT_EQ(L"foo.exe", GetBinaryPathWithExtension(L"foo"));
  ASSERT_EQ(L"foo.sh.exe", GetBinaryPathWithExtension(L"foo.sh.exe"));
  ASSERT_EQ(L"foo.sh.exe", GetBinaryPathWithExtension(L"foo.sh"));
}

TEST_F(LaunchUtilTest, BashEscapeArgTest) {
  ASSERT_EQ(L"\"\"", BashEscapeArg(L""));
  ASSERT_EQ(L"foo", BashEscapeArg(L"foo"));
  ASSERT_EQ(L"\"foo bar\"", BashEscapeArg(L"foo bar"));
  ASSERT_EQ(L"\"\\\"foo bar\\\"\"", BashEscapeArg(L"\"foo bar\""));
  ASSERT_EQ(L"foo\\\\bar", BashEscapeArg(L"foo\\bar"));
  ASSERT_EQ(L"foo\\\"bar", BashEscapeArg(L"foo\"bar"));
  ASSERT_EQ(L"C:\\\\foo\\\\bar\\\\", BashEscapeArg(L"C:\\foo\\bar\\"));
  ASSERT_EQ(L"\"C:\\\\foo foo\\\\bar\\\\\"",
            BashEscapeArg(L"C:\\foo foo\\bar\\"));
}

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
    std::wstring (*escape_func)(const std::wstring& s),
    const std::vector<std::pair<wstring, wstring> >& args) {
  // Assert that the WindowsEscapeArg produces what we expect.
  for (const auto& i : args) {
    ASSERT_EQ(escape_func(i.first), i.second);
  }

  // Create a Runfiles object.
  string error;
  std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
      bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error));
  ASSERT_NE(runfiles.get(), nullptr) << error;

  // Look up the path of the printarg.exe utility.
  string printarg =
      runfiles->Rlocation("io_bazel/src/tools/launcher/util/printarg.exe");
  ASSERT_NE(printarg, "");

  // Convert printarg.exe's path to a wchar_t Windows path.
  wstring wprintarg;
  bool success =
      blaze_util::AsAbsoluteWindowsPath(printarg, &wprintarg, &error);
  ASSERT_TRUE(success) << error;

  // SECURITY_ATTRIBUTES for inheritable HANDLEs.
  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = NULL;
  sa.bInheritHandle = TRUE;

  // Open /dev/null that will be redirected into the subprocess' stdin.
  bazel::windows::AutoHandle devnull(
      CreateFileW(L"NUL", GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &sa,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
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
  wstring werror;
  ASSERT_TRUE(bazel::windows::AutoAttributeList::Create(
      devnull, pipe_write, stderr_dup, &attrs, &werror));
  attrs->InitStartupInfoExW(&startupInfo);

  // MSDN says the maximum command line is 32767 characters, with a null
  // terminator that is exactly 2^15 (= 0x8000).
  static constexpr size_t kMaxCmdline = 0x8000;
  wchar_t cmdline[kMaxCmdline];

  // Copy printarg.exe's escaped path into the 'cmdline', and append a space.
  // We will append arguments to this command line in the for-loop below.
  wprintarg = escape_func(wprintarg);
  wcsncpy(cmdline, wprintarg.c_str(), wprintarg.size());
  wchar_t* pcmdline = cmdline + wprintarg.size();
  *pcmdline++ = L' ';

  // Run a subprocess for each of the arguments and assert that the argument
  // arrived to the subprocess as intended.
  for (const auto& i : args) {
    // We already asserted for every element that escape_func(i.first)
    // produces the same output as i.second, so just use i.second instead of
    // converting i.first again.
    wcsncpy(pcmdline, i.second.c_str(), i.second.size());
    pcmdline[i.second.size()] = 0;

    // Run the subprocess.
    PROCESS_INFORMATION processInfo;
    BOOL ok = CreateProcessW(
        NULL, cmdline, NULL, NULL, TRUE,
        CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT, NULL, NULL,
        &startupInfo.StartupInfo, &processInfo);
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
    ASSERT_TRUE(WriteFile(pipe_write, "\0", 1, &dummy, NULL));
  }

  // Read the output of the subprocesses from the pipe. They are divided by
  // null-terminators, so 'buf' will contain a sequence of null-terminated
  // strings.  We close the writing end so that ReadFile won't block until the
  // desired amount of bytes is available.
  DWORD total_output_len;
  char buf[0x10000];
  pipe_write = INVALID_HANDLE_VALUE;
  if (!ReadFile(pipe_read, buf, 0x10000, &total_output_len, NULL)) {
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
    string actual_arg(buf + start);
    start += actual_arg.size() + 1;

    // 'args' contains wchar_t strings, but the subprocesses printed ASCII
    // (char) strings. To compare, we convert arg.first to a char-string.
    string expected_arg;
    expected_arg.reserve(arg.first.size());
    for (const auto& wc : arg.first) {
      expected_arg.append(1, static_cast<char>(wc));
    }

    // Assert that the subprocess printed exactly the *unescaped* argument.
    EXPECT_EQ(expected_arg, actual_arg);
  }
}

TEST_F(LaunchUtilTest, WindowsEscapeArgTest) {
  // List of arguments with their expected WindowsEscapeArg-encoded version.
  AssertSubprocessReceivesArgsAsIntended(
      WindowsEscapeArg,
      {
          // Each pair is:
          // - first: argument to pass (and expected output from subprocess)
          // - second: expected WindowsEscapeArg-encoded string
          {L"foo", L"foo"},
          {L"", L"\"\""},
          {L" ", L"\" \""},
          {L"foo\\bar", L"foo\\bar"},
          {L"C:\\foo\\bar\\", L"C:\\foo\\bar\\"},
          // TODO(laszlocsomor): fix WindowsEscapeArg to use correct escaping
          // semantics (not Bash semantics) and add more tests. The example
          // below is
          // escaped incorrectly.
          // {L"C:\\foo bar\\", L"\"C:\\foo bar\\\""},
      });
}

TEST_F(LaunchUtilTest, WindowsEscapeArg2Test) {
  AssertSubprocessReceivesArgsAsIntended(
      WindowsEscapeArg2,
      {
          {L"", L"\"\""},
          {L" ", L"\" \""},
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

TEST_F(LaunchUtilTest, DoesFilePathExistTest) {
  wstring file1 = GetTmpDir() + L"/foo";
  wstring file2 = GetTmpDir() + L"/bar";
  CreateEmptyFile(file1);
  ASSERT_TRUE(DoesFilePathExist(file1.c_str()));
  ASSERT_FALSE(DoesFilePathExist(file2.c_str()));
}

TEST_F(LaunchUtilTest, DoesDirectoryPathExistTest) {
  wstring dir1 = GetTmpDir() + L"/dir1";
  wstring dir2 = GetTmpDir() + L"/dir2";
  CreateDirectoryW(dir1.c_str(), NULL);
  ASSERT_TRUE(DoesDirectoryPathExist(dir1.c_str()));
  ASSERT_FALSE(DoesDirectoryPathExist(dir2.c_str()));
}

TEST_F(LaunchUtilTest, SetAndGetEnvTest) {
  ASSERT_TRUE(SetEnv(L"foo", L"bar"));
  wstring value;
  ASSERT_TRUE(GetEnv(L"foo", &value));
  ASSERT_EQ(value, L"bar");
  SetEnv(L"FOO", L"");
  ASSERT_FALSE(GetEnv(L"FOO", &value));
}

TEST_F(LaunchUtilTest, NormalizePathTest) {
  wstring value;
  ASSERT_TRUE(NormalizePath(L"C:\\foo\\bar\\", &value));
  ASSERT_EQ(L"c:\\foo\\bar", value);
  ASSERT_TRUE(NormalizePath(L"c:/foo/bar/", &value));
  ASSERT_EQ(L"c:\\foo\\bar", value);
  ASSERT_TRUE(NormalizePath(L"FoO\\\\bAr\\", &value));
  ASSERT_EQ(L"foo\\bar", value);
  ASSERT_TRUE(NormalizePath(L"X\\Y/Z\\", &value));
  ASSERT_EQ(L"x\\y\\z", value);
  ASSERT_TRUE(NormalizePath(L"c://foo//bar", &value));
  ASSERT_EQ(L"c:\\foo\\bar", value);
  ASSERT_FALSE(NormalizePath(L"c:foo\\bar", &value));
}

TEST_F(LaunchUtilTest, RelativeToTest) {
  wstring value;
  ASSERT_TRUE(RelativeTo(L"c:\\foo\\bar1", L"c:\\foo\\bar2", &value));
  ASSERT_EQ(L"..\\bar1", value);
  ASSERT_TRUE(RelativeTo(L"c:\\foo\\bar", L"c:\\", &value));
  ASSERT_EQ(L"foo\\bar", value);
  ASSERT_TRUE(RelativeTo(L"c:\\foo\\bar", L"c:\\foo\\bar", &value));
  ASSERT_EQ(L"", value);
  ASSERT_TRUE(RelativeTo(L"c:\\foo\\bar", L"c:\\foo", &value));
  ASSERT_EQ(L"bar", value);
  ASSERT_TRUE(RelativeTo(L"c:\\foo\\bar", L"c:\\foo\\ba", &value));
  ASSERT_EQ(L"..\\bar", value);
  ASSERT_TRUE(RelativeTo(L"c:\\", L"c:\\foo", &value));
  ASSERT_EQ(L"..\\", value);
  ASSERT_TRUE(RelativeTo(L"c:\\", L"c:\\a\\b\\c", &value));
  ASSERT_EQ(L"..\\..\\..\\", value);
  ASSERT_TRUE(RelativeTo(L"c:\\aa\\bb\\cc", L"c:\\a\\b", &value));
  ASSERT_EQ(L"..\\..\\aa\\bb\\cc", value);

  ASSERT_TRUE(RelativeTo(L"foo\\bar", L"foo\\bar", &value));
  ASSERT_EQ(L"", value);
  ASSERT_TRUE(RelativeTo(L"foo\\bar1", L"foo\\bar2", &value));
  ASSERT_EQ(L"..\\bar1", value);
  ASSERT_TRUE(RelativeTo(L"foo\\bar1", L"foo\\bar", &value));
  ASSERT_EQ(L"..\\bar1", value);
  ASSERT_TRUE(RelativeTo(L"foo\\bar1", L"foo", &value));
  ASSERT_EQ(L"bar1", value);
  ASSERT_TRUE(RelativeTo(L"foo\\bar1", L"fo", &value));
  ASSERT_EQ(L"..\\foo\\bar1", value);
  ASSERT_TRUE(RelativeTo(L"foo\\ba", L"foo\\bar", &value));
  ASSERT_EQ(L"..\\ba", value);
  ASSERT_TRUE(RelativeTo(L"foo", L"foo\\bar", &value));
  ASSERT_EQ(L"..\\", value);
  ASSERT_TRUE(RelativeTo(L"fo", L"foo\\bar", &value));
  ASSERT_EQ(L"..\\..\\fo", value);
  ASSERT_TRUE(RelativeTo(L"", L"foo\\bar", &value));
  ASSERT_EQ(L"..\\..\\", value);
  ASSERT_TRUE(RelativeTo(L"foo\\bar", L"", &value));
  ASSERT_EQ(L"foo\\bar", value);
  ASSERT_TRUE(RelativeTo(L"a\\b\\c", L"x\\y", &value));
  ASSERT_EQ(L"..\\..\\a\\b\\c", value);

  ASSERT_FALSE(RelativeTo(L"c:\\foo\\bar1", L"foo\\bar2", &value));
  ASSERT_FALSE(RelativeTo(L"c:foo\\bar1", L"c:\\foo\\bar2", &value));
  ASSERT_FALSE(RelativeTo(L"c:\\foo\\bar1", L"d:\\foo\\bar2", &value));
}

}  // namespace launcher
}  // namespace bazel
