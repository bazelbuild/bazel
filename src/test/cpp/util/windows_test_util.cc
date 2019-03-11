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
#include <windows.h>

#include <algorithm>
#include <memory>
#include <string>

#include "src/test/cpp/util/windows_test_util.h"

#if !defined(_WIN32) && !defined(__CYGWIN__)
#error("This test should only be run on Windows")
#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

namespace blaze_util {

using std::unique_ptr;
using std::wstring;

wstring GetTestTmpDirW() {
  DWORD size = ::GetEnvironmentVariableW(L"TEST_TMPDIR", NULL, 0);
  unique_ptr<WCHAR[]> buf(new WCHAR[size]);
  ::GetEnvironmentVariableW(L"TEST_TMPDIR", buf.get(), size);
  wstring result(buf.get());
  std::replace(result.begin(), result.end(), '/', '\\');
  if (result.back() == '\\') {
    result.pop_back();
  }
  return result;
}

bool DeleteAllUnder(wstring path) {
  static const wstring kDot(L".");
  static const wstring kDotDot(L"..");

  // Prepend UNC prefix if the path doesn't have it already. Don't bother
  // checking if the path is shorter than MAX_PATH, let's just do it
  // unconditionally; this is a test after all, performance isn't paramount.
  if (path.find(L"\\\\?\\") != 0) {
    path = wstring(L"\\\\?\\") + path;
  }
  // Append "\" if necessary.
  if (path.back() != '\\') {
    path.push_back('\\');
  }

  WIN32_FIND_DATAW metadata;
  HANDLE handle = FindFirstFileW((path + L"*").c_str(), &metadata);
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
  } while (FindNextFileW(handle, &metadata));
  FindClose(handle);
  return result;
}

bool CreateDummyFile(const wstring& path, const std::string& content) {
  return CreateDummyFile(path, content.c_str(), content.size());
}

bool CreateDummyFile(const std::wstring& path, const void* content,
                     const DWORD size) {
  HANDLE handle =
      ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL,
                    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  bool result = true;
  DWORD actually_written = 0;
  if (!::WriteFile(handle, content, size, &actually_written, NULL) &&
      actually_written != size) {
    result = false;
  }
  CloseHandle(handle);
  return result;
}

}  // namespace blaze_util
