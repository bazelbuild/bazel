// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: laszlocsomor@google.com (Laszlo Csomor)
//
// Unit tests for long-path-aware open/mkdir/access on Windows.
//
// This file is only used on Windows, it's empty on other platforms.

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <errno.h>
#include <stdlib.h>
#include <windows.h>

#include <google/protobuf/stubs/io_win32.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace google {
namespace protobuf {
namespace stubs {
namespace {

using std::string;
using std::unique_ptr;
using std::wstring;

class IoWin32Test : public ::testing::Test {
 public:
  void SetUp() override;
  void TearDown() override;

 protected:
  static bool DeleteAllUnder(wstring path);

  string test_tmpdir;
  wstring wtest_tmpdir;
};

#define ASSERT_INITIALIZED              \
  {                                     \
    EXPECT_FALSE(test_tmpdir.empty());  \
    EXPECT_FALSE(wtest_tmpdir.empty()); \
  }

void IoWin32Test::SetUp() {
  test_tmpdir.clear();
  wtest_tmpdir.clear();

  const char* test_tmpdir_env = getenv("TEST_TMPDIR");
  if (test_tmpdir_env == nullptr || *test_tmpdir_env == 0) {
    // Using assertions in SetUp/TearDown seems to confuse the test framework,
    // so just leave the member variables empty in case of failure.
    return;
  }

  test_tmpdir = string(test_tmpdir_env);
  while (test_tmpdir.back() == '/' || test_tmpdir.back() == '\\') {
    test_tmpdir.pop_back();
  }

  // CreateDirectoryA's limit is 248 chars, see MSDN.
  // https://msdn.microsoft.com/en-us/library/windows/ \
  //   desktop/aa363855(v=vs.85).aspx
  wtest_tmpdir = testonly_path_to_winpath(test_tmpdir, 248);
}

void IoWin32Test::TearDown() {
  if (!wtest_tmpdir.empty()) {
    DeleteAllUnder(wtest_tmpdir);
  }
}

bool IoWin32Test::DeleteAllUnder(wstring path) {
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");

  // Prepend UNC prefix if the path doesn't have it already. Don't bother
  // checking if the path is shorter than MAX_PATH, let's just do it
  // unconditionally.
  if (path.find(L"\\\\?\\") != 0) {
    path = wstring(L"\\\\?\\") + path;
  }
  // Append "\" if necessary.
  if (path.back() != '\\') {
    path.push_back('\\');
  }

  WIN32_FIND_DATAW metadata;
  HANDLE handle = ::FindFirstFileW((path + L"*").c_str(), &metadata);
  if (handle == INVALID_HANDLE_VALUE) {
    return true;  // directory doesn't exist
  }

  bool result = true;
  do {
    wstring childname = metadata.cFileName;
    if (kDot != childname && kDotDot != childname) {
      wstring childpath = path + childname;
      if ((metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        // If this is not a junction, delete its contents recursively.
        // Finally delete this directory/junction too.
        if (((metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0 &&
             !DeleteAllUnder(childpath)) ||
            !::RemoveDirectoryW(childpath.c_str())) {
          result = false;
          break;
        }
      } else {
        if (!::DeleteFileW(childpath.c_str())) {
          result = false;
          break;
        }
      }
    }
  } while (::FindNextFileW(handle, &metadata));
  ::FindClose(handle);
  return result;
}

TEST_F(IoWin32Test, AccessTest) {
  ASSERT_INITIALIZED;

  string path = test_tmpdir;
  while (path.size() < MAX_PATH - 30) {
    path += "\\accesstest";
    EXPECT_EQ(mkdir(path.c_str(), 0644), 0);
  }
  string file = path + "\\file.txt";
  int fd = open(file.c_str(), O_CREAT | O_WRONLY, 0644);
  EXPECT_GT(fd, 0);
  EXPECT_EQ(close(fd), 0);

  EXPECT_EQ(access(test_tmpdir.c_str(), F_OK), 0);
  EXPECT_EQ(access(path.c_str(), F_OK), 0);
  EXPECT_EQ(access(path.c_str(), W_OK), 0);
  EXPECT_EQ(access(file.c_str(), F_OK | W_OK), 0);
  EXPECT_NE(access((file + ".blah").c_str(), F_OK), 0);
  EXPECT_NE(access((file + ".blah").c_str(), W_OK), 0);

  // chdir into the test_tmpdir, because the current working directory must
  // always be shorter than MAX_PATH, even with "\\?\" prefix (except on
  // Windows 10 version 1607 and beyond, after opting in to long paths by
  // default [1]).
  //
  // [1] https://msdn.microsoft.com/en-us/library/windows/ \
  //   desktop/aa365247(v=vs.85).aspx#maxpath
  EXPECT_EQ(_chdir(test_tmpdir.c_str()), 0);
  EXPECT_EQ(access(".", F_OK), 0);
  EXPECT_EQ(access(".", W_OK), 0);
  EXPECT_EQ(access("accesstest", F_OK | W_OK), 0);
  ASSERT_EQ(access("./normalize_me/../.././accesstest", F_OK | W_OK), 0);
  EXPECT_NE(access("blah", F_OK), 0);
  EXPECT_NE(access("blah", W_OK), 0);

  ASSERT_EQ(access("c:bad", F_OK), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(access("/tmp/bad", F_OK), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(access("\\bad", F_OK), -1);
  ASSERT_EQ(errno, ENOENT);
}

TEST_F(IoWin32Test, OpenTest) {
  ASSERT_INITIALIZED;

  string path = test_tmpdir;
  while (path.size() < MAX_PATH) {
    path += "\\opentest";
    EXPECT_EQ(mkdir(path.c_str(), 0644), 0);
  }
  string file = path + "\\file.txt";
  int fd = open(file.c_str(), O_CREAT | O_WRONLY, 0644);
  ASSERT_GT(fd, 0);
  EXPECT_EQ(write(fd, "hello", 5), 5);
  EXPECT_EQ(close(fd), 0);

  // chdir into the test_tmpdir, because the current working directory must
  // always be shorter than MAX_PATH, even with "\\?\" prefix (except on
  // Windows 10 version 1607 and beyond, after opting in to long paths by
  // default [1]).
  //
  // [1] https://msdn.microsoft.com/en-us/library/windows/ \
  //   desktop/aa365247(v=vs.85).aspx#maxpath
  EXPECT_EQ(_chdir(test_tmpdir.c_str()), 0);
  fd = open("file-relative.txt", O_CREAT | O_WRONLY, 0644);
  ASSERT_GT(fd, 0);
  EXPECT_EQ(write(fd, "hello", 5), 5);
  EXPECT_EQ(close(fd), 0);

  fd = open("./normalize_me/../.././file-relative.txt", O_RDONLY);
  ASSERT_GT(fd, 0);
  EXPECT_EQ(close(fd), 0);

  ASSERT_EQ(open("c:bad.txt", O_CREAT | O_WRONLY, 0644), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(open("/tmp/bad.txt", O_CREAT | O_WRONLY, 0644), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(open("\\bad.txt", O_CREAT | O_WRONLY, 0644), -1);
  ASSERT_EQ(errno, ENOENT);
}

TEST_F(IoWin32Test, MkdirTest) {
  ASSERT_INITIALIZED;

  string path = test_tmpdir;
  do {
    path += "\\mkdirtest";
    ASSERT_EQ(mkdir(path.c_str(), 0644), 0);
  } while (path.size() <= MAX_PATH);

  // chdir into the test_tmpdir, because the current working directory must
  // always be shorter than MAX_PATH, even with "\\?\" prefix (except on
  // Windows 10 version 1607 and beyond, after opting in to long paths by
  // default [1]).
  //
  // [1] https://msdn.microsoft.com/en-us/library/windows/ \
  //   desktop/aa365247(v=vs.85).aspx#maxpath
  EXPECT_EQ(_chdir(test_tmpdir.c_str()), 0);
  ASSERT_EQ(mkdir("relative_mkdirtest", 0644), 0);
  ASSERT_EQ(mkdir("./normalize_me/../.././blah", 0644), 0);

  ASSERT_EQ(mkdir("c:bad", 0644), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(mkdir("/tmp/bad", 0644), -1);
  ASSERT_EQ(errno, ENOENT);
  ASSERT_EQ(mkdir("\\bad", 0644), -1);
  ASSERT_EQ(errno, ENOENT);
}

}  // namespace
}  // namespace stubs
}  // namespace protobuf
}  // namespace google

#endif  // defined(_WIN32)

