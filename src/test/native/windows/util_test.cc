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
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <algorithm>   // replace
#include <functional>  // function
#include <memory>      // unique_ptr
#include <string>

#include "gtest/gtest.h"
#include "src/main/native/windows/util.h"

#if !defined(COMPILER_MSVC) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(COMPILER_MSVC) && !defined(__CYGWIN__)

namespace bazel {
namespace windows {

using std::function;
using std::string;
using std::unique_ptr;
using std::wstring;

static const wstring kUncPrefix = wstring(L"\\\\?\\");

// Retrieves TEST_TMPDIR as a shortened path. Result won't have a "\\?\" prefix.
static void GetShortTempDir(wstring* result) {
  unique_ptr<WCHAR[]> buf;
  DWORD size = ::GetEnvironmentVariableW(L"TEST_TMPDIR", NULL, 0);
  ASSERT_GT(size, (DWORD)0);
  // `size` accounts for the null-terminator
  buf.reset(new WCHAR[size]);
  ::GetEnvironmentVariableW(L"TEST_TMPDIR", buf.get(), size);

  // Add "\\?\" prefix.
  wstring tmpdir = kUncPrefix + wstring(buf.get());

  // Remove trailing '/' or '\' and replace all '/' with '\'.
  if (tmpdir.back() == '/' || tmpdir.back() == '\\') {
    tmpdir.pop_back();
  }
  std::replace(tmpdir.begin(), tmpdir.end(), '/', '\\');

  // Convert to 8dot3 style short path.
  size = ::GetShortPathNameW(tmpdir.c_str(), NULL, 0);
  ASSERT_GT(size, (DWORD)0);
  // `size` accounts for the null-terminator
  buf.reset(new WCHAR[size]);
  ::GetShortPathNameW(tmpdir.c_str(), buf.get(), size);

  // Set the result, omit the "\\?\" prefix.
  // Ensure that the result is shorter than MAX_PATH and also has room for a
  // backslash (1 char) and a single-letter executable name with .bat
  // extension (5 chars).
  *result = wstring(buf.get() + 4);
  ASSERT_LT(result->size(), MAX_PATH - 6);
}

// If `success` is true, returns an empty string, otherwise an error message.
// The error message will have the format "Failed: operation(arg)" using the
// specified `operation` and `arg` strings.
static wstring ReturnEmptyOrError(bool success, const wstring& operation,
                                  const wstring& arg) {
  return success ? L"" : (wstring(L"Failed: ") + operation + L"(" + arg + L")");
}

// Creates a dummy file under `path`. `path` should NOT have a "\\?\" prefix.
static wstring CreateDummyFile(wstring path) {
  path = kUncPrefix + path;
  HANDLE handle = ::CreateFileW(
      /* lpFileName */ path.c_str(),
      /* dwDesiredAccess */ GENERIC_WRITE,
      /* dwShareMode */ FILE_SHARE_READ,
      /* lpSecurityAttributes */ NULL,
      /* dwCreationDisposition */ CREATE_ALWAYS,
      /* dwFlagsAndAttributes */ FILE_ATTRIBUTE_NORMAL,
      /* hTemplateFile */ NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return ReturnEmptyOrError(false, L"CreateFileW", path);
  }
  DWORD actually_written = 0;
  WriteFile(handle, "hello", 5, &actually_written, NULL);
  if (actually_written == 0) {
    return ReturnEmptyOrError(false, L"WriteFile", path);
  }
  CloseHandle(handle);
  return L"";
}

// Asserts that a dummy file under `path` can be created.
// This is a macro so the assertions will have the correct line number.
#define CREATE_FILE(/* const wstring& */ path) \
  { ASSERT_EQ(CreateDummyFile(path), L""); }

// Deletes a file under `path`. `path` should NOT have a "\\?\" prefix.
static wstring DeleteDummyFile(wstring path) {
  path = kUncPrefix + path;
  return ReturnEmptyOrError(::DeleteFileW(path.c_str()), L"DeleteFileW", path);
}

// Asserts that a file under `path` can be deleted.
// This is a macro so the assertions will have the correct line number.
#define DELETE_FILE(/* const wstring& */ path) \
  { ASSERT_EQ(DeleteDummyFile(path), L""); }

// Creates a directory under `path`. `path` should NOT have a "\\?\" prefix.
static wstring CreateDir(wstring path) {
  path = kUncPrefix + path;
  return ReturnEmptyOrError(::CreateDirectoryW(path.c_str(), NULL),
                            L"CreateDirectoryW", path);
}

// Asserts that a directory under `path` can be created.
// This is a macro so the assertions will have the correct line number.
#define CREATE_DIR(/* const wstring& */ path) \
  { ASSERT_EQ(CreateDir(path), L""); }

// Deletes an empty directory under `path`.
// `path` should NOT have a "\\?\" prefix.
static wstring DeleteDir(wstring path) {
  path = kUncPrefix + path;
  return ReturnEmptyOrError(::RemoveDirectoryW(path.c_str()),
                            L"RemoveDirectoryW", path);
}

// Asserts that the empty directory under `path` can be deleted.
// This is a macro so the assertions will have the correct line number.
#define DELETE_DIR(/* const wstring& */ path) \
  { ASSERT_EQ(DeleteDir(path), L""); }

// Appends a file name segment with ".bat" extension to `result`.
// `length` specifies how long the segment may be, and it includes the "\" at
// the beginning. `length` must be in [6..13], so the shortest segment is
// "\a.bat", the longest is "\abcdefgh.bat".
// For example APPEND_FILE_SEGMENT(8, result) will append "\abc.bat" to
// `result`.
// This is a macro so the assertions will have the correct line number.
#define APPEND_FILE_SEGMENT(/* size_t */ length, /* wstring* */ result) \
  {                                                                     \
    ASSERT_GE(length, 6);                                               \
    ASSERT_LE(length, 13);                                              \
    *result += wstring(L"\\abcdefgh", length - 4) + L".bat";            \
  }

// Creates subdirectories under `basedir` and sets `result_path` to the deepest.
//
// `basedir` should be a shortened path, without "\\?\" prefix.
// `result_path` will be also a short path under `basedir`.
//
// Every directory in `result_path` will be created. The entire length of this
// path will be exactly MAX_PATH - 7 (not including null-terminator).
// Just by appending a file name segment between 6 and 8 characters long (i.e.
// "\a.bat", "\ab.bat", or "\abc.bat") the caller can obtain a path that is
// MAX_PATH - 1 long, or MAX_PATH long, or MAX_PATH + 1 long, respectively,
// and cannot be shortened further.
static void CreateShortDirsUnder(wstring basedir, wstring* result_path) {
  ASSERT_LT(basedir.size(), MAX_PATH);
  size_t remaining_len = MAX_PATH - 1 - basedir.size();
  ASSERT_GE(remaining_len, 6);  // has room for suffix "\a.bat"

  // If `remaining_len` is odd, make it even.
  if (remaining_len % 2) {
    remaining_len -= 3;
    basedir += wstring(L"\\ab");
    CREATE_DIR(basedir);
  }

  // Keep adding 2 chars long segments until we only have 6 chars left.
  while (remaining_len >= 8) {
    remaining_len -= 2;
    basedir += wstring(L"\\a");
    CREATE_DIR(basedir);
  }
  ASSERT_EQ(basedir.size(), MAX_PATH - 1 - 6);
  *result_path = basedir;
}

// Deletes `deepest_subdir` and all of its parents below `basedir`.
// `basedir` must be a prefix (ancestor) of `deepest_subdir`. Neither of them
// should have a "\\?\" prefix.
// Every subdirectory connecting `basedir` and `deepest_subdir` must be empty
// except for the single directory child connecting these two nodes. In other
// words it should be possible to remove all directories starting at
// `deepest_subdir` and walking up the tree until `basedir` is reached.
// `basedir` is NOT deleted and it doesn't need to be empty either.
static void DeleteDirsUnder(const wstring& basedir,
                            const wstring& deepest_subdir) {
  // Assert that `deepest_subdir` starts with `basedir`.
  ASSERT_EQ(deepest_subdir.find(basedir), 0);

  // Make a mutable copy of `deepest_subdir`.
  unique_ptr<WCHAR[]> mutable_path(new WCHAR[deepest_subdir.size() + 1]);
  memcpy(mutable_path.get(), deepest_subdir.c_str(),
         deepest_subdir.size() * sizeof(wchar_t));
  mutable_path.get()[deepest_subdir.size()] = 0;

  // Mark the end of the path. We'll keep setting the last directory separator
  // to the null-terminator, thus walking up the directory tree.
  wchar_t* p_end = mutable_path.get() + deepest_subdir.size();

  while (p_end > mutable_path.get() + basedir.size()) {
    DELETE_DIR(mutable_path.get());
    // Walk up one level in the path.
    while (*p_end != '\\') {
      --p_end;
    }
    *p_end = '\0';
  }
}

// Converts a wstring to a string using `wcstombs`.
static string AsString(const wstring& s) {
  size_t size = wcstombs(nullptr, s.c_str(), 0) + 1;
  unique_ptr<char[]> result(new char[size]);
  wcstombs(result.get(), s.c_str(), size);
  return string(result.get());
}

// Converts a string to a wstring using `mbstowcs`.
static wstring AsWstring(const char* s) {
  size_t size = mbstowcs(nullptr, s, 0) + 1;
  unique_ptr<WCHAR[]> result(new WCHAR[size]);
  mbstowcs(result.get(), s, size);
  return wstring(result.get());
}

static function<wstring()> MakeConversionFunc(const char* input) {
  return [input]() { return AsWstring(input); };
}

// Asserts that `str` contains substring `substr`.
// This is a macro so the assertions will have the correct line number.
#define ASSERT_CONTAINS(/* const string& */ str, /* const char* */ substr) \
  {                                                                        \
    ASSERT_NE(str, "");                                                    \
    ASSERT_NE(str.find(substr), string::npos);                             \
  }

// This is a macro so the assertions will have the correct line number.
#define ASSERT_SHORTENING_FAILS(/* const char* */ input,            \
                                /* const char* */ error_msg)        \
  {                                                                 \
    string actual;                                                  \
    ASSERT_CONTAINS(AsExecutablePathForCreateProcess(               \
                        input, MakeConversionFunc(input), &actual), \
                    error_msg);                                     \
  }

// This is a macro so the assertions will have the correct line number.
#define ASSERT_SHORTENING_SUCCEEDS(/* const char* */ input,             \
                                   /* const string& */ expected_result) \
  {                                                                     \
    string actual;                                                      \
    ASSERT_EQ(AsExecutablePathForCreateProcess(                         \
                  input, MakeConversionFunc(input), &actual),           \
              "");                                                      \
    ASSERT_EQ(actual, expected_result);                                 \
  }

TEST(WindowsUtilTest, TestAsExecutablePathForCreateProcessBadInputs) {
  ASSERT_SHORTENING_FAILS("", "should not be empty");
  ASSERT_SHORTENING_FAILS("\"cmd.exe\"", "should not be quoted");
  ASSERT_SHORTENING_FAILS("/dev/null", "path='/dev/null' is absolute");
  ASSERT_SHORTENING_FAILS("/usr/bin/bash", "path='/usr/bin/bash' is absolute");
  ASSERT_SHORTENING_FAILS("foo\\bar.exe", "absolute");
  ASSERT_SHORTENING_FAILS("foo\\..\\bar.exe", "normalized");
  ASSERT_SHORTENING_FAILS("\\bar.exe", "path='\\bar.exe' is absolute");

  string dummy = "hello";
  while (dummy.size() < MAX_PATH) {
    dummy += dummy;
  }
  dummy += ".exe";
  ASSERT_SHORTENING_FAILS(dummy.c_str(), "a file name but too long");
}

TEST(WindowsUtilTest, TestAsExecutablePathForCreateProcessConversions) {
  wstring tmpdir;
  GetShortTempDir(&tmpdir);
  wstring short_root;
  CreateShortDirsUnder(tmpdir, &short_root);

  // Assert that we have enough room to append a file name that is just short
  // enough to fit into MAX_PATH - 1, or one that's just long enough to make
  // the whole path MAX_PATH long or longer.
  ASSERT_EQ(short_root.size(), MAX_PATH - 1 - 6);

  string actual;
  string error;
  for (size_t i = 0; i < 3; ++i) {
    wstring wfilename = short_root;

    APPEND_FILE_SEGMENT(6 + i, &wfilename);
    string filename = AsString(wfilename);
    ASSERT_EQ(filename.size(), MAX_PATH - 1 + i);

    // When i=0 then `filename` is MAX_PATH - 1 long, so
    // `AsExecutablePathForCreateProcess` will not attempt to shorten it, and
    // so it also won't notice that the file doesn't exist. If however we pass
    // a non-existent path to CreateProcessA, then it'll fail, so we'll find out
    // about this error in production code.
    // When i>0 then `filename` is at least MAX_PATH long, so
    // `AsExecutablePathForCreateProcess` will attempt to shorten it, but
    // because the file doesn't yet exist, the shortening attempt will fail.
    if (i > 0) {
      ASSERT_EQ(::GetFileAttributesA(filename.c_str()),
                INVALID_FILE_ATTRIBUTES);
      ASSERT_SHORTENING_FAILS(filename.c_str(), "GetShortPathName failed");
    }

    // Create the file, now we should be able to shorten it when i=0, but not
    // otherwise.
    CREATE_FILE(wfilename);
    if (i == 0) {
      // The filename was short enough.
      ASSERT_SHORTENING_SUCCEEDS(filename.c_str(),
                                 string("\"") + filename + "\"");
    } else {
      // The filename was too long to begin with, and it was impossible to
      // shorten any of the segments (since we deliberately created them that
      // way), so shortening failed.
      ASSERT_SHORTENING_FAILS(filename.c_str(), "would not shorten");
    }
    DELETE_FILE(wfilename);
  }

  // Finally construct a path that can and will be shortened. Just walk up a few
  // levels in `short_root` and create a long file name that can be shortened.
  wstring wshortenable_root = short_root;
  while (wshortenable_root.size() > MAX_PATH - 1 - 13) {
    wshortenable_root =
        wshortenable_root.substr(0, wshortenable_root.find_last_of('\\'));
  }
  wstring wshortenable = wshortenable_root + wstring(L"\\") +
                         wstring(MAX_PATH - wshortenable_root.size(), 'a') +
                         wstring(L".bat");
  ASSERT_GT(wshortenable.size(), MAX_PATH);

  // Attempt to shorten. It will fail because the file doesn't exist yet.
  ASSERT_SHORTENING_FAILS(AsString(wshortenable).c_str(),
                          "GetShortPathName failed");

  // Create the file so shortening will succeed.
  CREATE_FILE(wshortenable);
  ASSERT_SHORTENING_SUCCEEDS(
      AsString(wshortenable).c_str(),
      string("\"") + AsString(wshortenable_root) + "\\AAAAAA~1.BAT\"");
  DELETE_FILE(wshortenable);

  DeleteDirsUnder(tmpdir, short_root);
}

}  // namespace windows
}  // namespace bazel
