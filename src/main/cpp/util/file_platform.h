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

#ifndef BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_

#include <stdint.h>
#include <time.h>

#include <string>

namespace blaze_util {

class IPipe;

IPipe* CreatePipe();

// Split a path to dirname and basename parts.
std::pair<std::string, std::string> SplitPath(const std::string &path);

// Replaces 'content' with contents of file 'filename'.
// If `max_size` is positive, the method reads at most that many bytes;
// otherwise the method reads the whole file.
// Returns false on error. Can be called from a signal handler.
bool ReadFile(const std::string &filename, std::string *content,
              int max_size = 0);

// Writes 'content' into file 'filename', and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const std::string &content, const std::string &filename);

// Unlinks the file given by 'file_path'.
// Returns true on success. In case of failure sets errno.
bool UnlinkPath(const std::string &file_path);

// Returns true if this path exists, following symlinks.
bool PathExists(const std::string& path);

// Returns the real, absolute path corresponding to `path`.
// The method resolves all symlink components of `path`.
// Returns the empty string upon error.
//
// This is a wrapper around realpath(3).
std::string MakeCanonical(const char *path);

// Returns true if the path exists and can be accessed to read/write as desired.
//
// If `exec` is true and the path refers to a file, it means the file must be
// executable; if the path is a directory, it means the directory must be
// openable.
bool CanAccess(const std::string& path, bool read, bool write, bool exec);

// Returns true if `path` refers to a directory or a symlink/junction to one.
bool IsDirectory(const std::string& path);

// Returns true if `path` is the root directory or a Windows drive root.
bool IsRootDirectory(const std::string &path);

// Returns true if `path` is absolute.
bool IsAbsolute(const std::string &path);

// Calls fsync() on the file (or directory) specified in 'file_path'.
// pdie() if syncing fails.
void SyncFile(const std::string& path);

// Returns the last modification time of `path` in milliseconds since the Epoch.
// Returns -1 upon failure.
time_t GetMtimeMillisec(const std::string& path);

// Sets the last modification time of `path` to the given value.
// `mtime` must be milliseconds since the Epoch.
// Returns true upon success.
bool SetMtimeMillisec(const std::string& path, time_t mtime);

// mkdir -p path. All newly created directories use the given mode.
// `mode` should be an octal permission mask, e.g. 0755.
// Returns false on failure, sets errno.
bool MakeDirectories(const std::string &path, unsigned int mode);

// Returns the current working directory.
// The path is platform-specific (e.g. Windows path of Windows) and absolute.
std::string GetCwd();

// Changes the current working directory to `path`, returns true upon success.
bool ChangeDirectory(const std::string& path);

// Interface to be implemented by ForEachDirectoryEntry clients.
class DirectoryEntryConsumer {
 public:
  virtual ~DirectoryEntryConsumer() {}

  // This method is called for each entry in a directory.
  // `name` is the full path of the entry.
  // `is_directory` is true if this entry is a directory (but false if this is a
  // symlink pointing to a directory).
  virtual void Consume(const std::string &name, bool is_directory) = 0;
};

// Executes a function for each entry in a directory (except "." and "..").
//
// Returns true if the `path` referred to a directory or directory symlink,
// false otherwise.
//
// See DirectoryEntryConsumer for more details.
void ForEachDirectoryEntry(const std::string &path,
                           DirectoryEntryConsumer *consume);

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
// Converts a UTF8-encoded `path` to a widechar Windows path.
//
// Returns true if conversion succeeded and sets the contents of `result` to it.
//
// The `path` may be absolute or relative, and may be a Windows or MSYS path.
// In every case, this method replaces forward slashes with backslashes if
// necessary.
//
// Recognizes the drive letter in MSYS paths, so e.g. "/c/windows" becomes
// "c:\windows". Prepends the MSYS root (computed from the BAZEL_SH envvar) to
// absolute MSYS paths, so e.g. "/usr" becomes "c:\tools\msys64\usr".
//
// The result may be longer than MAX_PATH. It's the caller's responsibility to
// prepend the long path prefix ("\\?\") in case they need to pass it to a
// Windows API function (some require the prefix, some don't), or to quote the
// path if necessary.
bool AsWindowsPath(const std::string &path, std::wstring *result);
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_
