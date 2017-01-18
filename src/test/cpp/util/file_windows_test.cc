// Copyright 2016 The Bazel Authors. All rights reserved.
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
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

#if !defined(COMPILER_MSVC) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(COMPILER_MSVC) && !defined(__CYGWIN__)

namespace blaze_util {

using std::string;
using std::wstring;

// Methods defined in file_windows.cc that are only visible for testing.
bool AsWindowsPath(const string& path, wstring* result);
void ResetMsysRootForTesting();
string NormalizeWindowsPath(string path);

// This is a macro so the assertions will have the correct line number.
#define GET_TEST_TMPDIR(/* string& */ result)                            \
  {                                                                      \
    char buf[MAX_PATH] = {0};                                            \
    DWORD len = ::GetEnvironmentVariableA("TEST_TMPDIR", buf, MAX_PATH); \
    result = buf;                                                        \
    ASSERT_GT(result.size(), 0);                                         \
  }

TEST(FileTest, TestNormalizeWindowsPath) {
  ASSERT_EQ(string(""), NormalizeWindowsPath(""));
  ASSERT_EQ(string(""), NormalizeWindowsPath("."));
  ASSERT_EQ(string("foo"), NormalizeWindowsPath("foo"));
  ASSERT_EQ(string("foo"), NormalizeWindowsPath("foo/"));
  ASSERT_EQ(string("foo\\bar"), NormalizeWindowsPath("foo//bar"));
  ASSERT_EQ(string("foo\\bar"), NormalizeWindowsPath("../..//foo/./bar"));
  ASSERT_EQ(string("foo\\bar"), NormalizeWindowsPath("../foo/baz/../bar"));
  ASSERT_EQ(string("c:\\"), NormalizeWindowsPath("c:"));
  ASSERT_EQ(string("c:\\"), NormalizeWindowsPath("c:/"));
  ASSERT_EQ(string("c:\\"), NormalizeWindowsPath("c:\\"));
  ASSERT_EQ(string("c:\\foo\\bar"), NormalizeWindowsPath("c:\\..//foo/./bar/"));
}

TEST(FileTest, TestDirname) {
  ASSERT_EQ("", Dirname(""));
  ASSERT_EQ("/", Dirname("/"));
  ASSERT_EQ("", Dirname("foo"));
  ASSERT_EQ("/", Dirname("/foo"));
  ASSERT_EQ("/foo", Dirname("/foo/"));
  ASSERT_EQ("foo", Dirname("foo/bar"));
  ASSERT_EQ("foo/bar", Dirname("foo/bar/baz"));
  ASSERT_EQ("\\", Dirname("\\foo"));
  ASSERT_EQ("\\foo", Dirname("\\foo\\"));
  ASSERT_EQ("foo", Dirname("foo\\bar"));
  ASSERT_EQ("foo\\bar", Dirname("foo\\bar\\baz"));
  ASSERT_EQ("foo\\bar/baz", Dirname("foo\\bar/baz\\qux"));
  ASSERT_EQ("c:/", Dirname("c:/"));
  ASSERT_EQ("c:\\", Dirname("c:\\"));
  ASSERT_EQ("c:/", Dirname("c:/foo"));
  ASSERT_EQ("c:\\", Dirname("c:\\foo"));
  ASSERT_EQ("\\\\?\\c:\\", Dirname("\\\\?\\c:\\"));
  ASSERT_EQ("\\\\?\\c:\\", Dirname("\\\\?\\c:\\foo"));
}

TEST(FileTest, TestBasename) {
  ASSERT_EQ("", Basename(""));
  ASSERT_EQ("", Basename("/"));
  ASSERT_EQ("foo", Basename("foo"));
  ASSERT_EQ("foo", Basename("/foo"));
  ASSERT_EQ("", Basename("/foo/"));
  ASSERT_EQ("bar", Basename("foo/bar"));
  ASSERT_EQ("baz", Basename("foo/bar/baz"));
  ASSERT_EQ("foo", Basename("\\foo"));
  ASSERT_EQ("", Basename("\\foo\\"));
  ASSERT_EQ("bar", Basename("foo\\bar"));
  ASSERT_EQ("baz", Basename("foo\\bar\\baz"));
  ASSERT_EQ("qux", Basename("foo\\bar/baz\\qux"));
  ASSERT_EQ("", Basename("c:/"));
  ASSERT_EQ("", Basename("c:\\"));
  ASSERT_EQ("foo", Basename("c:/foo"));
  ASSERT_EQ("foo", Basename("c:\\foo"));
  ASSERT_EQ("", Basename("\\\\?\\c:\\"));
  ASSERT_EQ("foo", Basename("\\\\?\\c:\\foo"));
}

TEST(FileTest, IsAbsolute) {
  ASSERT_FALSE(IsAbsolute(""));
  ASSERT_TRUE(IsAbsolute("/"));
  ASSERT_TRUE(IsAbsolute("/foo"));
  ASSERT_TRUE(IsAbsolute("\\"));
  ASSERT_TRUE(IsAbsolute("\\foo"));
  ASSERT_FALSE(IsAbsolute("c:"));
  ASSERT_TRUE(IsAbsolute("c:/"));
  ASSERT_TRUE(IsAbsolute("c:\\"));
  ASSERT_TRUE(IsAbsolute("c:\\foo"));
  ASSERT_TRUE(IsAbsolute("\\\\?\\c:\\"));
  ASSERT_TRUE(IsAbsolute("\\\\?\\c:\\foo"));
}

TEST(FileTest, IsRootDirectory) {
  ASSERT_FALSE(IsRootDirectory(""));
  ASSERT_TRUE(IsRootDirectory("/"));
  ASSERT_FALSE(IsRootDirectory("/foo"));
  ASSERT_TRUE(IsRootDirectory("\\"));
  ASSERT_FALSE(IsRootDirectory("\\foo"));
  ASSERT_FALSE(IsRootDirectory("c:"));
  ASSERT_TRUE(IsRootDirectory("c:/"));
  ASSERT_TRUE(IsRootDirectory("c:\\"));
  ASSERT_FALSE(IsRootDirectory("c:\\foo"));
  ASSERT_TRUE(IsRootDirectory("\\\\?\\c:\\"));
  ASSERT_FALSE(IsRootDirectory("\\\\?\\c:\\foo"));
}

TEST(FileTest, TestAsWindowsPath) {
  SetEnvironmentVariableA("BAZEL_SH", "c:\\msys\\some\\long\\path\\bash.exe");
  ResetMsysRootForTesting();
  wstring actual;

  ASSERT_TRUE(AsWindowsPath("", &actual));
  ASSERT_EQ(wstring(L""), actual);

  ASSERT_TRUE(AsWindowsPath("", &actual));
  ASSERT_EQ(wstring(L""), actual);

  ASSERT_TRUE(AsWindowsPath("foo/bar", &actual));
  ASSERT_EQ(wstring(L"foo\\bar"), actual);

  ASSERT_TRUE(AsWindowsPath("c:", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("c:/", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("c:\\", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("\\\\?\\c:\\", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("\\\\?\\c://../foo", &actual));
  ASSERT_EQ(wstring(L"c:\\foo"), actual);

  ASSERT_TRUE(AsWindowsPath("/dev/null", &actual));
  ASSERT_EQ(wstring(L"NUL"), actual);

  ASSERT_TRUE(AsWindowsPath("Nul", &actual));
  ASSERT_EQ(wstring(L"NUL"), actual);

  ASSERT_TRUE(AsWindowsPath("/c", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("/c/", &actual));
  ASSERT_EQ(wstring(L"c:\\"), actual);

  ASSERT_TRUE(AsWindowsPath("/c/blah", &actual));
  ASSERT_EQ(wstring(L"c:\\blah"), actual);

  ASSERT_TRUE(AsWindowsPath("/d/progra~1/micros~1", &actual));
  ASSERT_EQ(wstring(L"d:\\progra~1\\micros~1"), actual);

  ASSERT_TRUE(AsWindowsPath("/foo", &actual));
  ASSERT_EQ(wstring(L"c:\\msys\\foo"), actual);

  wstring wlongpath(L"\\dummy_long_path");
  string longpath("dummy_long_path/");
  while (longpath.size() <= MAX_PATH) {
    wlongpath += wlongpath;
    longpath += longpath;
  }
  wlongpath = wstring(L"c:") + wlongpath;
  longpath = string("/c/") + longpath;
  ASSERT_TRUE(AsWindowsPath(longpath, &actual));
  ASSERT_EQ(wlongpath, actual);
}

TEST(FileTest, TestAsShortWindowsPath) {
  string actual;
  ASSERT_TRUE(AsShortWindowsPath("/dev/null", &actual));
  ASSERT_EQ(string("NUL"), actual);

  ASSERT_TRUE(AsShortWindowsPath("nul", &actual));
  ASSERT_EQ(string("NUL"), actual);

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  string short_tmpdir;
  ASSERT_TRUE(AsShortWindowsPath(tmpdir, &short_tmpdir));
  ASSERT_LT(0, short_tmpdir.size());
  ASSERT_TRUE(PathExists(short_tmpdir));

  string dirname(JoinPath(short_tmpdir, "LONGpathNAME"));
  ASSERT_EQ(0, mkdir(dirname.c_str()));
  ASSERT_TRUE(PathExists(dirname));

  ASSERT_TRUE(AsShortWindowsPath(dirname, &actual));
  ASSERT_EQ(short_tmpdir + "\\longpa~1", actual);
  ASSERT_EQ(0, rmdir(dirname.c_str()));
}

TEST(FileTest, TestMsysRootRetrieval) {
  wstring actual;

  SetEnvironmentVariableA("BAZEL_SH", "c:/foo/msys/bar/qux.exe");
  ResetMsysRootForTesting();
  ASSERT_TRUE(AsWindowsPath("/blah", &actual));
  ASSERT_EQ(wstring(L"c:\\foo\\msys\\blah"), actual);

  SetEnvironmentVariableA("BAZEL_SH", "c:/foo/MSYS64/bar/qux.exe");
  ResetMsysRootForTesting();
  ASSERT_TRUE(AsWindowsPath("/blah", &actual));
  ASSERT_EQ(wstring(L"c:\\foo\\msys64\\blah"), actual);

  SetEnvironmentVariableA("BAZEL_SH", "c:/qux.exe");
  ResetMsysRootForTesting();
  ASSERT_FALSE(AsWindowsPath("/blah", &actual));

  SetEnvironmentVariableA("BAZEL_SH", nullptr);
  ResetMsysRootForTesting();
}

static void RunCommand(const string& cmdline) {
  STARTUPINFOA startupInfo = {sizeof(STARTUPINFO)};
  PROCESS_INFORMATION processInfo;
  // command line maximum size is 32K
  // Source (on 2017-01-04):
  // https://msdn.microsoft.com/en-us/library/windows/desktop/ms682425(v=vs.85).aspx
  char mutable_cmdline[0x8000];
  strncpy(mutable_cmdline, cmdline.c_str(), 0x8000);
  BOOL ok = CreateProcessA(
      /* lpApplicationName */ NULL,
      /* lpCommandLine */ mutable_cmdline,
      /* lpProcessAttributes */ NULL,
      /* lpThreadAttributes */ NULL,
      /* bInheritHandles */ TRUE,
      /* dwCreationFlags */ 0,
      /* lpEnvironment */ NULL,
      /* lpCurrentDirectory */ NULL,
      /* lpStartupInfo */ &startupInfo,
      /* lpProcessInformation */ &processInfo);
  ASSERT_TRUE(ok);

  // Wait 1 second for the process to finish.
  ASSERT_EQ(WAIT_OBJECT_0, WaitForSingleObject(processInfo.hProcess, 1000));

  DWORD exit_code = 1;
  ASSERT_TRUE(GetExitCodeProcess(processInfo.hProcess, &exit_code));
  ASSERT_EQ(0, exit_code);
}

TEST(FileTest, TestPathExistsWindows) {
  ASSERT_FALSE(PathExists(""));
  ASSERT_TRUE(PathExists("."));
  ASSERT_FALSE(PathExists("non.existent"));
  ASSERT_TRUE(PathExists("/dev/null"));
  ASSERT_TRUE(PathExists("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_TRUE(PathExists(tmpdir));

  // Create a fake msys root. We'll also use it as a junction target.
  string fake_msys_root(tmpdir + "/fake_msys");
  ASSERT_EQ(0, mkdir(fake_msys_root.c_str()));
  ASSERT_TRUE(PathExists(fake_msys_root));

  // Set the BAZEL_SH root so we can resolve MSYS paths.
  SetEnvironmentVariableA("BAZEL_SH",
                          (fake_msys_root + "/fake_bash.exe").c_str());
  ResetMsysRootForTesting();

  // Assert existence check for MSYS paths.
  ASSERT_FALSE(PathExists("/this/should/not/exist/mkay"));
  ASSERT_TRUE(PathExists("/"));

  // Create a junction pointing to an existing directory.
  RunCommand(string("cmd.exe /C mklink /J \"") + tmpdir + "/junc1\" \"" +
             fake_msys_root + "\" >NUL 2>NUL");
  ASSERT_TRUE(PathExists(fake_msys_root));
  ASSERT_TRUE(PathExists(JoinPath(tmpdir, "junc1")));

  // Create a junction pointing to a non-existent directory.
  RunCommand(string("cmd.exe /C mklink /J \"") + tmpdir + "/junc2\" \"" +
             fake_msys_root + "/i.dont.exist\" >NUL 2>NUL");
  ASSERT_FALSE(PathExists(JoinPath(fake_msys_root, "i.dont.exist")));
  ASSERT_FALSE(PathExists(JoinPath(tmpdir, "junc2")));

  // Clean up.
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "junc1").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "junc2").c_str()));
  ASSERT_EQ(0, rmdir(fake_msys_root.c_str()));
  ASSERT_FALSE(PathExists(JoinPath(tmpdir, "junc1")));
  ASSERT_FALSE(PathExists(JoinPath(tmpdir, "junc2")));
}

TEST(FileTest, TestIsDirectory) {
  ASSERT_FALSE(IsDirectory(""));
  ASSERT_FALSE(IsDirectory("/dev/null"));
  ASSERT_FALSE(IsDirectory("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_TRUE(IsDirectory(tmpdir));
  ASSERT_TRUE(IsDirectory("C:\\"));
  ASSERT_TRUE(IsDirectory("C:/"));
  ASSERT_TRUE(IsDirectory("/c"));

  ASSERT_FALSE(IsDirectory("non.existent"));
  // Create a directory under `tempdir`, verify that IsDirectory reports true.
  // Call it msys_dir1 so we can also use it as a mock msys root.
  string dir1(JoinPath(tmpdir, "msys_dir1"));
  ASSERT_EQ(0, mkdir(dir1.c_str()));
  ASSERT_TRUE(IsDirectory(dir1));

  // Use dir1 as the mock msys root, verify that IsDirectory works for a MSYS
  // path.
  SetEnvironmentVariableA("BAZEL_SH", JoinPath(dir1, "bash.exe").c_str());
  ResetMsysRootForTesting();
  ASSERT_TRUE(IsDirectory("/"));

  // Verify that IsDirectory works for a junction.
  string junc1(JoinPath(tmpdir, "junc1"));
  RunCommand(string("cmd.exe /C mklink /J \"") + junc1 + "\" \"" + dir1 +
             "\" >NUL 2>NUL");
  ASSERT_TRUE(IsDirectory(junc1));

  ASSERT_EQ(0, rmdir(dir1.c_str()));
  ASSERT_FALSE(IsDirectory(dir1));
  ASSERT_FALSE(IsDirectory(junc1));

  ASSERT_EQ(0, rmdir(junc1.c_str()));
}

TEST(FileTest, TestUnlinkPath) {
  ASSERT_FALSE(UnlinkPath("/dev/null"));
  ASSERT_FALSE(UnlinkPath("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);

  // Create a directory under `tempdir`, a file inside it, and a junction
  // pointing to it.
  string dir1(JoinPath(tmpdir, "dir1"));
  ASSERT_EQ(0, mkdir(dir1.c_str()));
  FILE* fh = fopen(JoinPath(dir1, "foo.txt").c_str(), "wt");
  ASSERT_NE(nullptr, fh);
  ASSERT_LT(0, fprintf(fh, "hello\n"));
  fclose(fh);
  string junc1(JoinPath(tmpdir, "junc1"));
  RunCommand(string("cmd.exe /C mklink /J \"") + junc1 + "\" \"" + dir1 +
             "\" >NUL 2>NUL");
  ASSERT_TRUE(PathExists(junc1));
  ASSERT_TRUE(PathExists(JoinPath(junc1, "foo.txt")));

  // Non-existent files cannot be unlinked.
  ASSERT_FALSE(UnlinkPath("does.not.exist"));
  // Directories cannot be unlinked.
  ASSERT_FALSE(UnlinkPath(dir1));
  // Junctions can be unlinked, even if the pointed directory is not empty.
  ASSERT_TRUE(UnlinkPath(JoinPath(junc1, "foo.txt")));
  // Files can be unlinked.
  ASSERT_TRUE(UnlinkPath(junc1));
  // Clean up the now empty directory.
  ASSERT_EQ(0, rmdir(dir1.c_str()));
}

TEST(FileTest, TestMakeDirectories) {
  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_LT(0, tmpdir.size());

  SetEnvironmentVariableA(
      "BAZEL_SH", (JoinPath(tmpdir, "fake_msys/fake_bash.exe")).c_str());
  ResetMsysRootForTesting();
  ASSERT_EQ(0, mkdir(JoinPath(tmpdir, "fake_msys").c_str()));
  ASSERT_TRUE(IsDirectory(JoinPath(tmpdir, "fake_msys")));

  // Test that we can create come directories, can't create others.
  ASSERT_FALSE(MakeDirectories("", 0777));
  ASSERT_FALSE(MakeDirectories("/dev/null", 0777));
  ASSERT_FALSE(MakeDirectories("c:/", 0777));
  ASSERT_FALSE(MakeDirectories("c:\\", 0777));
  ASSERT_FALSE(MakeDirectories("/", 0777));
  ASSERT_TRUE(MakeDirectories("/foo", 0777));
  ASSERT_TRUE(MakeDirectories(".", 0777));
  ASSERT_TRUE(MakeDirectories(tmpdir, 0777));
  ASSERT_TRUE(MakeDirectories(JoinPath(tmpdir, "dir1/dir2/dir3"), 0777));
  ASSERT_TRUE(MakeDirectories(string("\\\\?\\") + tmpdir + "/dir4/dir5", 0777));

  // Clean up.
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "fake_msys/foo").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "fake_msys").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "dir1/dir2/dir3").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "dir1/dir2").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "dir1").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "dir4/dir5").c_str()));
  ASSERT_EQ(0, rmdir(JoinPath(tmpdir, "dir4").c_str()));
}

TEST(FileTest, CanAccess) {
  ASSERT_FALSE(CanReadFile("C:/windows/this/should/not/exist/mkay"));
  ASSERT_FALSE(CanExecuteFile("C:/this/should/not/exist/mkay"));
  ASSERT_FALSE(CanAccessDirectory("C:/this/should/not/exist/mkay"));

  ASSERT_FALSE(CanReadFile("non.existent"));
  ASSERT_FALSE(CanExecuteFile("non.existent"));
  ASSERT_FALSE(CanAccessDirectory("non.existent"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  string dir(JoinPath(tmpdir, "canaccesstest"));
  ASSERT_EQ(0, mkdir(dir.c_str()));

  ASSERT_FALSE(CanReadFile(dir));
  ASSERT_FALSE(CanExecuteFile(dir));
  ASSERT_TRUE(CanAccessDirectory(dir));

  string junc(JoinPath(tmpdir, "junc1"));
  RunCommand(string("cmd.exe /C mklink /J \"") + junc + "\" \"" + dir +
             "\" >NUL 2>NUL");
  ASSERT_FALSE(CanReadFile(junc));
  ASSERT_FALSE(CanExecuteFile(junc));
  ASSERT_TRUE(CanAccessDirectory(junc));

  string file(JoinPath(dir, "foo.txt"));
  FILE* fh = fopen(file.c_str(), "wt");
  ASSERT_NE(nullptr, fh);
  ASSERT_LT(0, fprintf(fh, "hello"));
  fclose(fh);

  ASSERT_TRUE(CanReadFile(file));
  ASSERT_FALSE(CanExecuteFile(file));
  ASSERT_FALSE(CanAccessDirectory(file));

  ASSERT_TRUE(CanReadFile(JoinPath(junc, "foo.txt")));
  ASSERT_FALSE(CanExecuteFile(JoinPath(junc, "foo.txt")));
  ASSERT_FALSE(CanAccessDirectory(JoinPath(junc, "foo.txt")));

  string file2(JoinPath(dir, "foo.exe"));
  ASSERT_EQ(0, rename(file.c_str(), file2.c_str()));
  ASSERT_TRUE(CanReadFile(file2));
  ASSERT_TRUE(CanExecuteFile(file2));
  ASSERT_FALSE(CanAccessDirectory(file2));

  ASSERT_EQ(0, unlink(file2.c_str()));
  ASSERT_EQ(0, rmdir(junc.c_str()));
  ASSERT_EQ(0, rmdir(dir.c_str()));
}

}  // namespace blaze_util
