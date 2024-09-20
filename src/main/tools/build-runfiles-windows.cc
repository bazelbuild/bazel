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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <string.h>
#include <windows.h>

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/native/windows/file.h"

using std::ifstream;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::wstring;

#ifndef SYMBOLIC_LINK_FLAG_DIRECTORY
#define SYMBOLIC_LINK_FLAG_DIRECTORY 0x1
#endif

namespace {

const std::regex kEscapedBackslash(R"(\\b)");
const std::regex kEscapedSpace(R"(\\s)");

const wchar_t* manifest_filename;
const wchar_t* runfiles_base_dir;

string GetLastErrorString() {
  DWORD last_error = GetLastError();
  if (last_error == 0) {
    return string();
  }

  char* message_buffer;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, last_error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&message_buffer, 0, nullptr);

  stringstream result;
  result << "(error: " << last_error << "): " << message_buffer;
  LocalFree(message_buffer);
  return result.str();
}

void die(const wchar_t* format, ...) {
  va_list ap;
  va_start(ap, format);
  fputws(L"build-runfiles error: ", stderr);
  vfwprintf(stderr, format, ap);
  va_end(ap);
  fputwc(L'\n', stderr);
  fputws(L"manifest file name: ", stderr);
  fputws(manifest_filename, stderr);
  fputwc(L'\n', stderr);
  fputws(L"runfiles base directory: ", stderr);
  fputws(runfiles_base_dir, stderr);
  fputwc(L'\n', stderr);
  exit(1);
}

wstring AsAbsoluteWindowsPath(const wchar_t* path) {
  wstring wpath;
  string error;
  if (!blaze_util::AsAbsoluteWindowsPath(path, &wpath, &error)) {
    die(L"Couldn't convert %s to absolute Windows path: %hs", path,
        error.c_str());
  }
  return wpath;
}

bool DoesDirectoryPathExist(const wchar_t* path) {
  DWORD dwAttrib = GetFileAttributesW(AsAbsoluteWindowsPath(path).c_str());

  return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
          (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

wstring GetParentDirFromPath(const wstring& path) {
  return path.substr(0, path.find_last_of(L"\\/"));
}

bool ReadSymlink(const wstring& abs_path, wstring* target, wstring* error) {
  switch (bazel::windows::ReadSymlinkOrJunction(abs_path, target, error)) {
    case bazel::windows::ReadSymlinkOrJunctionResult::kSuccess:
      return true;
    case bazel::windows::ReadSymlinkOrJunctionResult::kAccessDenied:
      *error = L"access is denied";
      break;
    case bazel::windows::ReadSymlinkOrJunctionResult::kDoesNotExist:
      *error = L"path does not exist";
      break;
    case bazel::windows::ReadSymlinkOrJunctionResult::kNotALink:
      *error = L"path is not a link";
      break;
    case bazel::windows::ReadSymlinkOrJunctionResult::kUnknownLinkType:
      *error = L"unknown link type";
      break;
    default:
      // This is bazel::windows::ReadSymlinkOrJunctionResult::kError (1).
      // The JNI code puts a custom message in 'error'.
      break;
  }
  return false;
}

}  // namespace

class RunfilesCreator {
  typedef std::unordered_map<std::wstring, std::wstring> ManifestFileMap;

 public:
  RunfilesCreator(const wstring& manifest_path,
                  const wstring& runfiles_output_base)
      : manifest_path_(manifest_path),
        runfiles_output_base_(runfiles_output_base) {
    SetupOutputBase();
    if (!SetCurrentDirectoryW(runfiles_output_base_.c_str())) {
      die(L"SetCurrentDirectoryW failed (%s): %hs",
          runfiles_output_base_.c_str(), GetLastErrorString().c_str());
    }
  }

  void ReadManifest(bool allow_relative, bool ignore_metadata) {
    ifstream manifest_file(
        AsAbsoluteWindowsPath(manifest_path_.c_str()).c_str());

    if (!manifest_file) {
      die(L"Couldn't open MANIFEST file: %s", manifest_path_.c_str());
    }

    string line;
    int lineno = 0;
    while (getline(manifest_file, line)) {
      lineno++;
      // Skip metadata lines. They are used solely for
      // dependency checking.
      if (ignore_metadata && lineno % 2 == 0) {
        continue;
      }

      wstring link;
      wstring target;
      if (!line.empty() && line[0] == ' ') {
        // The link path contains escape sequences for spaces and backslashes.
        string::size_type idx = line.find(' ', 1);
        if (idx == string::npos) {
          die(L"Missing separator in manifest line: %hs", line.c_str());
        }
        std::string link_path = line.substr(1, idx - 1);
        link_path = std::regex_replace(link_path, kEscapedSpace, " ");
        link_path = std::regex_replace(link_path, kEscapedBackslash, "\\");
        link = blaze_util::CstringToWstring(link_path);
        target = blaze_util::CstringToWstring(line.substr(idx + 1));
      } else {
        string::size_type idx = line.find(' ');
        if (idx == string::npos) {
          die(L"Missing separator in manifest line: %hs", line.c_str());
        }
        link = blaze_util::CstringToWstring(line.substr(0, idx));
        target = blaze_util::CstringToWstring(line.substr(idx + 1));
      }

      // We sometimes need to create empty files under the runfiles tree.
      // For example, for python binary, __init__.py is needed under every
      // directory. Storing an entry with an empty target indicates we need to
      // create such a file when creating the runfiles tree.
      if (!allow_relative && !target.empty() &&
          !blaze_util::IsAbsolute(target)) {
        die(L"Target cannot be relative path: %hs", line.c_str());
      }

      link = AsAbsoluteWindowsPath(link.c_str());
      if (!target.empty()) {
        target = AsAbsoluteWindowsPath(target.c_str());
      }

      manifest_file_map.insert(make_pair(link, target));
    }
  }

  void CreateRunfiles() {
    ScanTreeAndPrune(runfiles_output_base_);
    CreateFiles();
    CopyManifestFile();
  }

 private:
  void SetupOutputBase() {
    if (!DoesDirectoryPathExist(runfiles_output_base_.c_str())) {
      MakeDirectoriesOrDie(runfiles_output_base_);
    }
  }

  void MakeDirectoriesOrDie(const wstring& path) {
    if (!blaze_util::MakeDirectoriesW(path, 0755)) {
      die(L"MakeDirectoriesW failed (%s): %hs", path.c_str(),
          GetLastErrorString().c_str());
    }
  }

  void RemoveDirectoryOrDie(const wstring& path) {
    if (!RemoveDirectoryW(path.c_str())) {
      die(L"RemoveDirectoryW failed (%s): %hs", GetLastErrorString().c_str());
    }
  }

  void DeleteFileOrDie(const wstring& path) {
    SetFileAttributesW(path.c_str(), GetFileAttributesW(path.c_str()) &
                                         ~FILE_ATTRIBUTE_READONLY);
    if (!DeleteFileW(path.c_str())) {
      die(L"DeleteFileW failed (%s): %hs", path.c_str(),
          GetLastErrorString().c_str());
    }
  }

  // This function scan the current directory, remove all
  // files/symlinks/directories that are not presented in manifest file. If a
  // symlink already exists and points to the correct target, this function
  // erases its entry from manifest_file_map, so that we won't recreate it.
  void ScanTreeAndPrune(const wstring& path) {
    static const wstring kDot(L".");
    static const wstring kDotDot(L"..");

    WIN32_FIND_DATAW metadata;
    HANDLE handle = ::FindFirstFileW((path + L"\\*").c_str(), &metadata);
    if (handle == INVALID_HANDLE_VALUE) {
      return;  // directory does not exist or is empty
    }

    do {
      if (kDot != metadata.cFileName && kDotDot != metadata.cFileName) {
        wstring subpath = path + L"\\" + metadata.cFileName;
        subpath = AsAbsoluteWindowsPath(subpath.c_str());
        bool is_dir =
            (metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        bool is_symlink =
            (metadata.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
        if (is_symlink) {
          wstring target, werror;
          if (!ReadSymlink(subpath, &target, &werror)) {
            die(L"ReadSymlinkW failed (%s): %hs", subpath.c_str(),
                werror.c_str());
          }

          target = AsAbsoluteWindowsPath(target.c_str());
          ManifestFileMap::iterator expected_target =
              manifest_file_map.find(subpath);

          if (expected_target == manifest_file_map.end() ||
              expected_target->second.empty()
              // Both paths are normalized paths in lower case, we can compare
              // them directly.
              || target != expected_target->second.c_str() ||
              blaze_util::IsDirectoryW(target) != is_dir) {
            if (is_dir) {
              RemoveDirectoryOrDie(subpath);
            } else {
              DeleteFileOrDie(subpath);
            }
          } else {
            manifest_file_map.erase(expected_target);
          }
        } else {
          if (is_dir) {
            ScanTreeAndPrune(subpath);
            // If the directory is empty, then we remove the directory.
            // Otherwise RemoveDirectory will fail with ERROR_DIR_NOT_EMPTY,
            // which we can just ignore.
            // Because if the directory is not empty, it means it contains some
            // symlinks already pointing to the correct targets (we just called
            // ScanTreeAndPrune). Then this directory shouldn't be removed in
            // the first place.
            if (!RemoveDirectoryW(subpath.c_str()) &&
                GetLastError() != ERROR_DIR_NOT_EMPTY) {
              die(L"RemoveDirectoryW failed (%s): %hs", subpath.c_str(),
                  GetLastErrorString().c_str());
            }
          } else {
            DeleteFileOrDie(subpath);
          }
        }
      }
    } while (::FindNextFileW(handle, &metadata));
    ::FindClose(handle);
  }

  void CreateFiles() {
    for (const auto& it : manifest_file_map) {
      // Ensure the parent directory exists
      wstring parent_dir = GetParentDirFromPath(it.first);
      if (!DoesDirectoryPathExist(parent_dir.c_str())) {
        MakeDirectoriesOrDie(parent_dir);
      }

      if (it.second.empty()) {
        // Create an empty file
        HANDLE h = CreateFileW(it.first.c_str(),  // name of the file
                               GENERIC_WRITE,     // open for writing
                               // Must share for reading, otherwise
                               // symlink-following file existence checks (e.g.
                               // java.nio.file.Files.exists()) fail.
                               FILE_SHARE_READ,
                               0,  // use default security descriptor
                               CREATE_ALWAYS,  // overwrite if exists
                               FILE_ATTRIBUTE_NORMAL, 0);
        if (h != INVALID_HANDLE_VALUE) {
          CloseHandle(h);
        } else {
          die(L"CreateFileW failed (%s): %hs", it.first.c_str(),
              GetLastErrorString().c_str());
        }
      } else {
        DWORD create_dir = 0;
        if (blaze_util::IsDirectoryW(it.second.c_str())) {
          create_dir = SYMBOLIC_LINK_FLAG_DIRECTORY;
        }
        if (!CreateSymbolicLinkW(
                it.first.c_str(), it.second.c_str(),
                bazel::windows::symlinkPrivilegeFlag | create_dir)) {
          if (GetLastError() == ERROR_INVALID_PARAMETER) {
            // We are on a version of Windows that does not support this flag.
            // Retry without the flag and return to error handling if necessary.
            if (CreateSymbolicLinkW(it.first.c_str(), it.second.c_str(),
                                    create_dir)) {
              return;
            }
          }
          if (GetLastError() == ERROR_PRIVILEGE_NOT_HELD) {
            die(L"CreateSymbolicLinkW failed:\n%hs\n",
                "Bazel needs to create symlinks to build the runfiles tree.\n"
                "Creating symlinks on Windows requires one of the following:\n"
                "    1. Bazel is run with administrator privileges.\n"
                "    2. The system version is Windows 10 Creators Update "
                "(1703) or "
                "later and developer mode is enabled.",
                GetLastErrorString().c_str());
          } else {
            die(L"CreateSymbolicLinkW failed (%s -> %s): %hs", it.first.c_str(),
                it.second.c_str(), GetLastErrorString().c_str());
          }
        }
      }
    }
  }

  void CopyManifestFile() {
    wstring new_manifest_file = runfiles_output_base_ + L"\\MANIFEST";
    if (!CopyFileW(manifest_path_.c_str(), new_manifest_file.c_str(),
                   /*bFailIfExists=*/FALSE)) {
      die(L"CopyFileW failed (%s -> %s): %hs", manifest_path_.c_str(),
          new_manifest_file.c_str(), GetLastErrorString().c_str());
    }
  }

 private:
  wstring manifest_path_;
  wstring runfiles_output_base_;
  ManifestFileMap manifest_file_map;
};

int wmain(int argc, wchar_t** argv) {
  argc--;
  argv++;
  bool allow_relative = false;
  bool ignore_metadata = false;

  while (argc >= 1) {
    if (wcscmp(argv[0], L"--allow_relative") == 0) {
      allow_relative = true;
      argc--;
      argv++;
    } else if (wcscmp(argv[0], L"--use_metadata") == 0) {
      // If --use_metadata is passed, it means manifest file contains metadata
      // lines, which we should ignore when reading manifest file.
      ignore_metadata = true;
      argc--;
      argv++;
    } else {
      break;
    }
  }

  if (argc != 2) {
    fprintf(stderr,
            "usage: [--allow_relative] [--use_metadata] "
            "<manifest_file> <runfiles_base_dir>\n");
    return 1;
  }

  manifest_filename = argv[0];
  runfiles_base_dir = argv[1];

  wstring manifest_absolute_path = AsAbsoluteWindowsPath(manifest_filename);
  wstring output_base_absolute_path = AsAbsoluteWindowsPath(runfiles_base_dir);

  RunfilesCreator runfiles_creator(manifest_absolute_path,
                                   output_base_absolute_path);
  runfiles_creator.ReadManifest(allow_relative, ignore_metadata);
  runfiles_creator.CreateRunfiles();

  return 0;
}
