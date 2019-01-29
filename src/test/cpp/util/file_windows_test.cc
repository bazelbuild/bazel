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

#include <algorithm>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"
#include "src/main/native/windows/util.h"
#include "src/test/cpp/util/test_util.h"
#include "src/test/cpp/util/windows_test_util.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

#define TOSTRING(x) #x

namespace blaze_util {

using bazel::windows::CreateJunction;
using bazel::windows::CreateJunctionResult;
using std::string;
using std::unique_ptr;
using std::wstring;

class FileWindowsTest : public ::testing::Test {
 public:
  void TearDown() override { DeleteAllUnder(GetTestTmpDirW()); }
};

// This is a macro so the assertions will have the correct line number.
#define GET_TEST_TMPDIR(/* string& */ result)                            \
  {                                                                      \
    char buf[MAX_PATH] = {0};                                            \
    DWORD len = ::GetEnvironmentVariableA("TEST_TMPDIR", buf, MAX_PATH); \
    result = buf;                                                        \
    ASSERT_GT(result.size(), 0);                                         \
  }

#define CREATE_JUNCTION(/* const string& */ name, /* const string& */ target) \
  {                                                                           \
    wstring wname;                                                            \
    wstring wtarget;                                                          \
    EXPECT_TRUE(AsWindowsPath(name, &wname, nullptr));                        \
    EXPECT_TRUE(AsWindowsPath(target, &wtarget, nullptr));                    \
    EXPECT_EQ(CreateJunction(wname, wtarget, nullptr),                        \
              CreateJunctionResult::kSuccess);                                \
  }

// Asserts that dir1 can be created with some content, and dir2 doesn't exist.
static void AssertTearDown(const WCHAR* dir1, const WCHAR* dir2) {
  wstring wtmpdir(GetTestTmpDirW());
  wstring dir1str(wtmpdir + L"\\" + dir1);
  wstring subdir(dir1str + L"\\subdir");
  wstring wfile(subdir + L"\\hello.txt");
  EXPECT_TRUE(::CreateDirectoryW(dir1str.c_str(), NULL));
  EXPECT_TRUE(::CreateDirectoryW(subdir.c_str(), NULL));
  EXPECT_TRUE(CreateDummyFile(wfile));
  EXPECT_NE(::GetFileAttributesW(wfile.c_str()), INVALID_FILE_ATTRIBUTES);
  ASSERT_EQ(::GetFileAttributesW((wtmpdir + L"\\" + dir2).c_str()),
            INVALID_FILE_ATTRIBUTES);
}

// One half of the teardown test: assert that test.teardown.b was cleaned up.
TEST_F(FileWindowsTest, TestTearDownA) {
  AssertTearDown(L"test.teardown.a", L"test.teardown.b");
}

// Other half of the teardown test: assert that test.teardown.a was cleaned up.
TEST_F(FileWindowsTest, TestTearDownB) {
  AssertTearDown(L"test.teardown.b", L"test.teardown.a");
}

TEST_F(FileWindowsTest, TestMsysRootRetrieval) {
  wstring actual;

  // We just need "bin/<something>" or "usr/bin/<something>".
  // Forward slashes are converted to backslashes.
  SetEnvironmentVariableA("BAZEL_SH", "c:/foo\\bin/some_bash.exe");

  string error;
  ASSERT_FALSE(AsWindowsPath("/blah", &actual, &error));
  EXPECT_TRUE(error.find("Unix-style") != string::npos);

  SetEnvironmentVariableA("BAZEL_SH", "c:/tools/msys64/usr/bin/bash.exe");
  ASSERT_FALSE(AsWindowsPath("/blah", &actual, &error));
  EXPECT_TRUE(error.find("Unix-style") != string::npos);
}

TEST_F(FileWindowsTest, TestPathExistsWindows) {
  ASSERT_FALSE(PathExists(""));
  ASSERT_TRUE(PathExists("."));
  ASSERT_FALSE(PathExists("non.existent"));
  ASSERT_TRUE(PathExists("/dev/null"));
  ASSERT_TRUE(PathExists("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_TRUE(PathExists(tmpdir));

  // Create junction target that'll double as the fake msys root.
  string fake_msys_root(tmpdir + "/blah");
  ASSERT_EQ(0, mkdir(fake_msys_root.c_str()));
  ASSERT_TRUE(PathExists(fake_msys_root));

  // Set the BAZEL_SH root so we can resolve MSYS paths.
  SetEnvironmentVariableA("BAZEL_SH",
                          (fake_msys_root + "/bin/fake_bash.exe").c_str());

  // Create a junction pointing to an existing directory.
  CREATE_JUNCTION(tmpdir + "/junc1", fake_msys_root);
  ASSERT_TRUE(PathExists(fake_msys_root));
  ASSERT_TRUE(PathExists(JoinPath(tmpdir, "junc1")));

  // Create a junction pointing to a non-existent directory.
  CREATE_JUNCTION(tmpdir + "/junc2", fake_msys_root + "/i.dont.exist");
  ASSERT_FALSE(PathExists(JoinPath(fake_msys_root, "i.dont.exist")));
  ASSERT_FALSE(PathExists(JoinPath(tmpdir, "junc2")));
}

TEST_F(FileWindowsTest, TestIsDirectory) {
  ASSERT_FALSE(IsDirectory(""));
  ASSERT_FALSE(IsDirectory("/dev/null"));
  ASSERT_FALSE(IsDirectory("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_TRUE(IsDirectory(tmpdir));
  ASSERT_TRUE(IsDirectory("C:\\"));
  ASSERT_TRUE(IsDirectory("C:/"));

  ASSERT_FALSE(IsDirectory("non.existent"));
  // Create a directory under `tempdir`, verify that IsDirectory reports true.
  string dir1(JoinPath(tmpdir, "dir1"));
  ASSERT_EQ(0, mkdir(dir1.c_str()));
  ASSERT_TRUE(IsDirectory(dir1));

  wstring wtmpdir(GetTestTmpDirW());
  EXPECT_TRUE(CreateDummyFile(wtmpdir + L"\\dummy.txt"));
  ASSERT_FALSE(IsDirectory(tmpdir + "\\dummy.txt"));

  // Verify that IsDirectory works for a junction.
  string junc1(JoinPath(tmpdir, "junc1"));
  CREATE_JUNCTION(junc1, dir1);
  ASSERT_TRUE(IsDirectory(junc1));

  ASSERT_EQ(0, rmdir(dir1.c_str()));
  ASSERT_FALSE(IsDirectory(dir1));
  ASSERT_FALSE(IsDirectory(junc1));
}

TEST_F(FileWindowsTest, TestUnlinkPath) {
  ASSERT_FALSE(UnlinkPath("/dev/null"));
  ASSERT_FALSE(UnlinkPath("Nul"));

  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);

  // Create a directory under `tempdir`, a file inside it, and a junction
  // pointing to it.
  string dir1(JoinPath(tmpdir, "dir1"));
  ASSERT_EQ(0, mkdir(dir1.c_str()));
  AutoFileStream fh(fopen(JoinPath(dir1, "foo.txt").c_str(), "wt"));
  EXPECT_TRUE(fh.IsOpen());
  ASSERT_LT(0, fprintf(fh, "hello\n"));
  fh.Close();
  string junc1(JoinPath(tmpdir, "junc1"));
  CREATE_JUNCTION(junc1, dir1);
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
}

TEST_F(FileWindowsTest, TestMakeDirectories) {
  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  ASSERT_LT(0, tmpdir.size());

  // Test that we can create come directories, can't create others.
  ASSERT_FALSE(MakeDirectories("", 0777));
  ASSERT_FALSE(MakeDirectories("/dev/null", 0777));
  ASSERT_TRUE(MakeDirectories("c:/", 0777));
  ASSERT_TRUE(MakeDirectories("c:\\", 0777));
  ASSERT_TRUE(MakeDirectories(".", 0777));
  ASSERT_TRUE(MakeDirectories(tmpdir, 0777));
  ASSERT_TRUE(MakeDirectories(JoinPath(tmpdir, "dir1/dir2/dir3"), 0777));

  string winpath = tmpdir + "\\dir4\\dir5";
  std::replace(winpath.begin(), winpath.end(), '/', '\\');
  ASSERT_TRUE(MakeDirectories(string("\\\\?\\") + winpath, 0777));
}

TEST_F(FileWindowsTest, TestCanAccess) {
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
  CREATE_JUNCTION(junc, dir);
  ASSERT_FALSE(CanReadFile(junc));
  ASSERT_FALSE(CanExecuteFile(junc));
  ASSERT_TRUE(CanAccessDirectory(junc));

  string file(JoinPath(dir, "foo.txt"));
  AutoFileStream fh(fopen(file.c_str(), "wt"));
  EXPECT_TRUE(fh.IsOpen());
  ASSERT_LT(0, fprintf(fh, "hello"));
  fh.Close();

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
}

TEST_F(FileWindowsTest, TestMakeCanonical) {
  string tmpdir;
  GET_TEST_TMPDIR(tmpdir);
  // Create some scratch directories: $TEST_TMPDIR/directory/subdirectory
  string dir1(JoinPath(tmpdir, "directory"));
  string dir2(JoinPath(dir1, "subdirectory"));
  EXPECT_TRUE(MakeDirectories(dir2, 0700));
  // Create a dummy file: $TEST_TMPDIR/directory/subdirectory/foo.txt
  string foo(JoinPath(dir2, "foo.txt"));
  wstring wfoo;
  EXPECT_TRUE(AsAbsoluteWindowsPath(foo, &wfoo, nullptr));
  EXPECT_TRUE(CreateDummyFile(wfoo));
  EXPECT_TRUE(CanReadFile(foo));
  // Create junctions next to directory and subdirectory, pointing to them.
  // Use short paths and mixed casing to test that the canonicalization can
  // resolve these.
  //   $TEST_TMPDIR/junc12345 -> $TEST_TMPDIR/DIRECT~1
  //   $TEST_TMPDIR/junc12~1/junc67890 -> $TEST_TMPDIR/JUNC12~1/SubDir~1
  string sym1(JoinPath(tmpdir, "junc12345"));
  string sym2(JoinPath(JoinPath(tmpdir, "junc12~1"), "junc67890"));
  string sym1value(JoinPath(tmpdir, "DIRECT~1"));
  string sym2value(JoinPath(JoinPath(tmpdir, "JUNC12~1"), "SubDir~1"));
  CREATE_JUNCTION(sym1, sym1value);
  CREATE_JUNCTION(sym2, sym2value);
  // Expect that $TEST_TMPDIR/sym1/sym2/foo.txt is readable.
  string symfoo(JoinPath(sym2, "foo.txt"));
  EXPECT_TRUE(CanReadFile(symfoo));
  // Assert the canonical path of foo.txt via the real path and via sym2.
  // The latter contains at least two junction components, shortened paths, and
  // mixed casing.
  string dircanon(MakeCanonical(foo.c_str()));
  string symcanon(MakeCanonical(symfoo.c_str()));
  string expected("directory\\subdirectory\\foo.txt");
  ASSERT_NE(symcanon, "");
  ASSERT_EQ(symcanon.find(expected), symcanon.size() - expected.size());
  ASSERT_EQ(dircanon, symcanon);
  // Assert the canonical path of "subdirectory" via the real path and via sym2.
  // The latter contains at least two junction components, shortened paths, and
  // mixed casing.
  dircanon = MakeCanonical(dir2.c_str());
  symcanon = MakeCanonical(sym2.c_str());
  expected = "directory\\subdirectory";
  ASSERT_NE(symcanon, "");
  ASSERT_EQ(symcanon.find(expected), symcanon.size() - expected.size());
  ASSERT_EQ(dircanon, symcanon);
}

TEST_F(FileWindowsTest, TestMtimeHandling) {
  const char* tempdir_cstr = getenv("TEST_TMPDIR");
  ASSERT_NE(tempdir_cstr, nullptr);
  ASSERT_NE(tempdir_cstr[0], 0);
  string tempdir(tempdir_cstr);

  string target(JoinPath(tempdir, "target" TOSTRING(__LINE__)));
  wstring wtarget;
  EXPECT_TRUE(AsWindowsPath(target, &wtarget, nullptr));
  EXPECT_TRUE(CreateDirectoryW(wtarget.c_str(), NULL));

  std::unique_ptr<IFileMtime> mtime(CreateFileMtime());
  // Assert that a directory is always a good embedded binary. (We do not care
  // about directories' mtimes.)
  ASSERT_TRUE(mtime.get()->IsUntampered(target));
  // Assert that junctions whose target exists are "good" embedded binaries.
  string sym(JoinPath(tempdir, "junc" TOSTRING(__LINE__)));
  CREATE_JUNCTION(sym, target);
  ASSERT_TRUE(mtime.get()->IsUntampered(sym));
  // Assert that checking fails for non-existent directories and dangling
  // junctions.
  EXPECT_TRUE(RemoveDirectoryW(wtarget.c_str()));
  ASSERT_FALSE(mtime.get()->IsUntampered(target));
  ASSERT_FALSE(mtime.get()->IsUntampered(sym));
}

}  // namespace blaze_util
