// Copyright 2018 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_TOOLS_TEST_WINDOWS_TW_H_
#define BAZEL_TOOLS_TEST_WINDOWS_TW_H_

#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace tools {
namespace test_wrapper {

// Info about a file/directory in the results of TestOnly_GetFileListRelativeTo.
class FileInfo {
 public:
  // C'tor for a directory.
  FileInfo(const std::wstring& rel_path)
      : rel_path_(rel_path), size_(0), is_dir_(true) {}

  // C'tor for a file.
  // Marked "explicit" because `size` is just a `int`.
  explicit FileInfo(const std::wstring& rel_path, int size)
      : rel_path_(rel_path), size_(size), is_dir_(false) {}

  inline const std::wstring& RelativePath() const { return rel_path_; }

  inline int Size() const { return size_; }

  inline bool IsDirectory() const { return is_dir_; }

 private:
  // The file's path, relative to the traversal root.
  std::wstring rel_path_;

  // The file's size, in bytes.
  //
  // Unfortunately this field has to be `int`, so it can only describe files up
  // to 2 GiB in size. The reason is, devtools_ijar::Stat::total_size is
  // declared as `int`, which is what we ultimately store the file size in,
  // therefore this field is also `int`.
  int size_;

  // Whether this is a directory (true) or a regular file (false).
  bool is_dir_;
};

// Zip entry paths for devtools_ijar::ZipBuilder.
// The function signatures mirror the signatures of ZipBuilder's functions.
class ZipEntryPaths {
 public:
  // Initialize the strings in this object.
  // `root` must be an absolute mixed-style path (Windows path with "/"
  // separators).
  // `files` must be relative, Unix-style paths.
  void Create(const std::string& root, const std::vector<std::string>& files);

  // Returns the number of paths in `AbsPathPtrs` and `EntryPathPtrs`.
  size_t Size() const { return size_; }

  // Returns a mutable array of const pointers to const char data.
  // Each pointer points to an absolute path: the file to archive.
  // The pointers are owned by this object and become invalid when the object is
  // destroyed.
  // Each entry corresponds to the entry at the same index in `EntryPathPtrs`.
  char const* const* AbsPathPtrs() const { return abs_path_ptrs_.get(); }

  // Returns a mutable array of const pointers to const char data.
  // Each pointer points to a relative path: an entry in the zip file.
  // The pointers are owned by this object and become invalid when the object is
  // destroyed.
  // Each entry corresponds to the entry at the same index in `AbsPathPtrs`.
  char const* const* EntryPathPtrs() const { return entry_path_ptrs_.get(); }

 private:
  size_t size_;
  std::unique_ptr<char[]> abs_paths_;
  std::unique_ptr<char*[]> abs_path_ptrs_;
  std::unique_ptr<char*[]> entry_path_ptrs_;
};

// The main function of the test wrapper.
int Main(int argc, wchar_t** argv);

// The "testing" namespace contains functions that should only be used by tests.
namespace testing {

// Retrieves an environment variable.
bool TestOnly_GetEnv(const wchar_t* name, std::wstring* result);

// Lists all files under `abs_root`, with paths relative to `abs_root`.
bool TestOnly_GetFileListRelativeTo(const std::wstring& abs_root,
                                    std::vector<FileInfo>* result);

// Converts a list of files to ZIP file entry paths.a
bool TestOnly_ToZipEntryPaths(
    const std::wstring& abs_root,
    const std::vector<bazel::tools::test_wrapper::FileInfo>& files,
    ZipEntryPaths* result);

// Archives `files` into a zip file at `abs_zip` (absolute path to the zip).
bool TestOnly_CreateZip(const std::wstring& abs_root,
                        const std::vector<FileInfo>& files,
                        const std::wstring& abs_zip);

bool TestOnly_AsMixedPath(const std::wstring& path, std::string* result);

}  // namespace testing

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_TEST_WINDOWS_TW_H_

