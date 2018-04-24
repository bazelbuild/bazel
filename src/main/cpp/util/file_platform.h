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

#include <time.h>

#include <cinttypes>
#include <string>

namespace blaze_util {

class IPipe;

IPipe* CreatePipe();

// Class to query/manipulate the last modification time (mtime) of files.
class IFileMtime {
 public:
  virtual ~IFileMtime() {}

  // Queries the mtime of `path` to see whether it's in the distant future.
  // Returns true if querying succeeded and stores the result in `result`.
  // Returns false if querying failed.
  virtual bool GetIfInDistantFuture(const std::string &path, bool *result) = 0;

  // Sets the mtime of file under `path` to the current time.
  // Returns true if the mtime was changed successfully.
  virtual bool SetToNow(const std::string &path) = 0;

  // Sets the mtime of file under `path` to the distant future.
  // "Distant future" should be on the order of some years into the future, like
  // a decade.
  // Returns true if the mtime was changed successfully.
  virtual bool SetToDistantFuture(const std::string &path) = 0;
};

// Creates a platform-specific implementation of `IFileMtime`.
IFileMtime *CreateFileMtime();

// Split a path to dirname and basename parts.
std::pair<std::string, std::string> SplitPath(const std::string &path);

#if defined(COMPILER_MSVC) || defined(__CYGWIN__)
// We cannot include <windows.h> because it #defines many symbols that conflict
// with our function names, e.g. GetUserName, SendMessage.
// Instead of typedef'ing HANDLE, let's use the actual type, void*. If that ever
// changes in the future and HANDLE would no longer be compatible with void*
// (very unlikely, given how fundamental this type is in Windows), then we'd get
// a compilation error.
typedef /* HANDLE */ void *file_handle_type;
#else   // !(defined(COMPILER_MSVC) || defined(__CYGWIN__))
typedef int file_handle_type;
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

// Result of a `ReadFromHandle` operation.
//
// This is a platform-independent abstraction of `errno`. If you need to handle
// an errno value, add an entry here and update the platform-specific
// `ReadFromHandle` implementations accordingly.
struct ReadFileResult {
  enum Errors {
    SUCCESS = 0,
    OTHER_ERROR = 1,
    INTERRUPTED = 2,
    AGAIN = 3,
  };
};

int ReadFromHandle(file_handle_type handle, void *data, size_t size,
                   int *error);

// Replaces 'content' with contents of file 'filename'.
// If `max_size` is positive, the method reads at most that many bytes;
// otherwise the method reads the whole file.
// Returns false on error. Can be called from a signal handler.
bool ReadFile(const std::string &filename, std::string *content,
              int max_size = 0);

// Reads up to `size` bytes from the file `filename` into `data`.
// There must be enough memory allocated at `data`.
// Returns true on success, false on error.
bool ReadFile(const std::string &filename, void *data, size_t size);

// Writes `size` bytes from `data` into file `filename` and chmods it to `perm`.
// Returns false on failure, sets errno.
bool WriteFile(const void *data, size_t size, const std::string &filename,
               unsigned int perm = 0644);

// Result of a `WriteToStdOutErr` operation.
//
// This is a platform-independent abstraction of `errno`. If you need to handle
// an errno value, add an entry here and update the platform-specific
// `WriteToStdOutErr` implementations accordingly.
struct WriteResult {
  enum Errors {
    SUCCESS = 0,
    OTHER_ERROR = 1,  // some uncategorized error occurred
    BROKEN_PIPE = 2,  // EPIPE (reading end of the pipe is closed)
  };
};

// Writes `size` bytes from `data` into stdout/stderr.
// Writes to stdout if `to_stdout` is true, writes to stderr otherwise.
// Returns one of `WriteResult::Errors`.
//
// This is a platform-independent abstraction of `fwrite` with `errno` checking
// and awareness of pipes (i.e. in case stderr/stdout is connected to a pipe).
int WriteToStdOutErr(const void *data, size_t size, bool to_stdout);

enum RenameDirectoryResult {
  kRenameDirectorySuccess = 0,
  kRenameDirectoryFailureNotEmpty = 1,
  kRenameDirectoryFailureOtherError = 2,
};

// Renames the directory at `old_name` to `new_name`.
// Returns one of the RenameDirectoryResult enum values.
int RenameDirectory(const std::string &old_name, const std::string &new_name);

// Reads which directory a symlink points to. Puts the target of the symlink
// in ``result`` and returns if the operation was successful. Will not work on
// symlinks that don't point to directories on Windows.
bool ReadDirectorySymlink(const std::string &symlink, std::string *result);

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

// Returns true if `path` exists, is a file or symlink to one, and is readable.
// Follows symlinks.
bool CanReadFile(const std::string &path);

// Returns true if `path` exists, is a file or symlink to one, and is writable.
// Follows symlinks.
bool CanExecuteFile(const std::string &path);

// Returns true if `path` exists, is a directory or symlink/junction to one, and
// is both readable and writable.
// Follows symlinks/junctions.
bool CanAccessDirectory(const std::string &path);

bool IsDevNull(const char *path);

// Returns true if `path` refers to a directory or a symlink/junction to one.
bool IsDirectory(const std::string& path);

// Returns true if `path` is the root directory or a Windows drive root.
bool IsRootDirectory(const std::string &path);

// Returns true if `path` is absolute.
bool IsAbsolute(const std::string &path);

// Calls fsync() on the file (or directory) specified in 'file_path'.
// pdie() if syncing fails.
void SyncFile(const std::string& path);

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
const wchar_t *RemoveUncPrefixMaybe(const wchar_t *ptr);

bool AsWindowsPath(const std::string &path, std::string *result,
                   std::string *error);

bool AsAbsoluteWindowsPath(const std::string &path, std::wstring *wpath,
                           std::string *error);

// Same as `AsWindowsPath`, but returns a lowercase 8dot3 style shortened path.
// Result will never have a UNC prefix, nor a trailing "/" or "\".
// Works also for non-existent paths; shortens as much of them as it can.
// Also works for non-existent drives.
bool AsShortWindowsPath(const std::string &path, std::string *result,
                        std::string *error);
#endif  // defined(COMPILER_MSVC) || defined(__CYGWIN__)

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_FILE_PLATFORM_H_
