// Copyright 2014 The Bazel Authors. All rights reserved.
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

#include "src/main/cpp/util/file_platform.h"

#include <dirent.h>  // DIR, dirent, opendir, closedir
#include <errno.h>
#include <fcntl.h>   // O_RDONLY
#include <limits.h>  // PATH_MAX
#include <stdlib.h>  // getenv
#include <string.h>  // strncmp
#include <sys/stat.h>
#include <unistd.h>  // access, open, close, fsync
#include <utime.h>   // utime

#include <string>
#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::string;

// Runs "stat" on `path`. Returns -1 and sets errno if stat fails or
// `path` isn't a directory. If check_perms is true, this will also
// make sure that `path` is owned by the current user and has `mode`
// permissions (observing the umask). It attempts to run chmod to
// correct the mode if necessary. If `path` is a symlink, this will
// check ownership of the link, not the underlying directory.
static bool GetDirectoryStat(const string &path, mode_t mode,
                             bool check_perms) {
  struct stat filestat = {};
  if (stat(path.c_str(), &filestat) == -1) {
    return false;
  }

  if (!S_ISDIR(filestat.st_mode)) {
    errno = ENOTDIR;
    return false;
  }

  if (check_perms) {
    // If this is a symlink, run checks on the link. (If we did lstat above
    // then it would return false for ISDIR).
    struct stat linkstat = {};
    if (lstat(path.c_str(), &linkstat) != 0) {
      return false;
    }
    if (linkstat.st_uid != geteuid()) {
      // The directory isn't owned by me.
      errno = EACCES;
      return false;
    }

    mode_t mask = umask(022);
    umask(mask);
    mode = (mode & ~mask);
    if ((filestat.st_mode & 0777) != mode && chmod(path.c_str(), mode) == -1) {
      // errno set by chmod.
      return false;
    }
  }
  return true;
}

static bool MakeDirectories(const string &path, mode_t mode, bool childmost) {
  if (path.empty() || IsRootDirectory(path)) {
    errno = EACCES;
    return false;
  }

  bool stat_succeeded = GetDirectoryStat(path, mode, childmost);
  if (stat_succeeded) {
    return true;
  }

  if (errno == ENOENT) {
    // Path does not exist, attempt to create its parents, then it.
    string parent = Dirname(path);
    if (!MakeDirectories(parent, mode, false)) {
      // errno set by stat.
      return false;
    }

    if (mkdir(path.c_str(), mode) == -1) {
      if (errno == EEXIST) {
        if (childmost) {
          // If there are multiple bazel calls at the same time then the
          // directory could be created between the MakeDirectories and mkdir
          // calls. This is okay, but we still have to check the permissions.
          return GetDirectoryStat(path, mode, childmost);
        } else {
          // If this isn't the childmost directory, we don't care what the
          // permissions were. If it's not even a directory then that error will
          // get caught when we attempt to create the next directory down the
          // chain.
          return true;
        }
      }
      // errno set by mkdir.
      return false;
    }
    return true;
  }

  return stat_succeeded;
}


string CreateTempDir(const std::string &prefix) {
  std::string parent = Dirname(prefix);
  // Need parent to exist first.
  if (!blaze_util::PathExists(parent) &&
      !blaze_util::MakeDirectories(parent, 0777)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "couldn't create '" << parent << "': "
        << blaze_util::GetLastErrorString();
  }

  std::string result(prefix + "XXXXXX");
  if (mkdtemp(&result[0]) == nullptr) {
    std::string err = GetLastErrorString();
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "could not create temporary directory under " << parent
        << " to extract install base into (" << err << ")";
  }

  // There's no better way to get the current umask than to set and reset it.
  const mode_t um = umask(0);
  umask(um);
  chmod(result.c_str(), 0777 & ~um);

  return result;
}

static bool RemoveDirRecursively(const std::string &path) {
  DIR *dir;
  if ((dir = opendir(path.c_str())) == NULL) {
    return false;
  }

  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) {
      continue;
    }

    if (!RemoveRecursively(blaze_util::JoinPath(path, ent->d_name))) {
      closedir(dir);
      return false;
    }
  }

  if (closedir(dir) != 0) {
    return false;
  }

  return rmdir(path.c_str()) == 0;
}

bool RemoveRecursively(const std::string &path) {
  struct stat stat_buf;
  if (lstat(path.c_str(), &stat_buf) == -1) {
    // Non-existent is good enough.
    return errno == ENOENT;
  }

  if (S_ISDIR(stat_buf.st_mode) && !S_ISLNK(stat_buf.st_mode)) {
    return RemoveDirRecursively(path);
  } else {
    return UnlinkPath(path);
  }
}

class PosixPipe : public IPipe {
 public:
  PosixPipe(int recv_socket, int send_socket)
      : _recv_socket(recv_socket), _send_socket(send_socket) {}

  PosixPipe() = delete;

  virtual ~PosixPipe() {
    close(_recv_socket);
    close(_send_socket);
  }

  bool Send(const void *buffer, int size) override {
    return size >= 0 && write(_send_socket, buffer, size) == size;
  }

  int Receive(void *buffer, int size, int *error) override {
    if (size < 0) {
      if (error != nullptr) {
        *error = IPipe::OTHER_ERROR;
      }
      return -1;
    }
    int result = read(_recv_socket, buffer, size);
    if (error != nullptr) {
      *error = result >= 0 ? IPipe::SUCCESS
                           : ((errno == EINTR) ? IPipe::INTERRUPTED
                                               : IPipe::OTHER_ERROR);
    }
    return result;
  }

 private:
  int _recv_socket;
  int _send_socket;
};

IPipe* CreatePipe() {
  int fd[2];
  if (pipe(fd) < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "pipe() failed: " << GetLastErrorString();
  }

  if (fcntl(fd[0], F_SETFD, FD_CLOEXEC) == -1) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "fcntl(F_SETFD, FD_CLOEXEC) failed: " << GetLastErrorString();
  }

  if (fcntl(fd[1], F_SETFD, FD_CLOEXEC) == -1) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "fcntl(F_SETFD, FD_CLOEXEC) failed: " << GetLastErrorString();
  }

  return new PosixPipe(fd[0], fd[1]);
}

int ReadFromHandle(file_handle_type fd, void *data, size_t size, int *error) {
  int result = read(fd, data, size);
  if (error != nullptr) {
    if (result >= 0) {
      *error = ReadFileResult::SUCCESS;
    } else {
      if (errno == EINTR) {
        *error = ReadFileResult::INTERRUPTED;
      } else if (errno == EAGAIN) {
        *error = ReadFileResult::AGAIN;
      } else {
        *error = ReadFileResult::OTHER_ERROR;
      }
    }
  }
  return result;
}

bool ReadFile(const string &filename, string *content, int max_size) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) return false;
  bool result = ReadFrom(fd, content, max_size);
  close(fd);
  return result;
}

bool ReadFile(const Path &path, std::string *content, int max_size) {
  return ReadFile(path.AsNativePath(), content, max_size);
}

bool ReadFile(const string &filename, void *data, size_t size) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) return false;
  bool result = ReadFrom(fd, data, size);
  close(fd);
  return result;
}

bool ReadFile(const Path &filename, void *data, size_t size) {
  return ReadFile(filename.AsNativePath(), data, size);
}

bool WriteFile(const void *data, size_t size, const string &filename,
               unsigned int perm) {
  UnlinkPath(filename);  // We don't care about the success of this.
  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, perm);
  if (fd == -1) {
    return false;
  }
  int result = write(fd, data, size);
  if (close(fd)) {
    return false;  // Can fail on NFS.
  }
  return result == static_cast<int>(size);
}

bool WriteFile(const void *data, size_t size, const Path &path,
               unsigned int perm) {
  return WriteFile(data, size, path.AsNativePath(), perm);
}

int WriteToStdOutErr(const void *data, size_t size, bool to_stdout) {
  size_t r = fwrite(data, 1, size, to_stdout ? stdout : stderr);
  return (r == size) ? WriteResult::SUCCESS
                     : ((errno == EPIPE) ? WriteResult::BROKEN_PIPE
                                         : WriteResult::OTHER_ERROR);
}

int RenameDirectory(const std::string &old_name, const std::string &new_name) {
  if (rename(old_name.c_str(), new_name.c_str()) == 0) {
    return kRenameDirectorySuccess;
  } else {
    if (errno == ENOTEMPTY || errno == EEXIST) {
      return kRenameDirectoryFailureNotEmpty;
    } else {
      return kRenameDirectoryFailureOtherError;
    }
  }
}

bool ReadDirectorySymlink(const blaze_util::Path &name, string *result) {
  char buf[PATH_MAX + 1];
  int len = readlink(name.AsNativePath().c_str(), buf, PATH_MAX);
  if (len < 0) {
    return false;
  }

  buf[len] = 0;
  *result = buf;
  return true;
}

bool UnlinkPath(const string &file_path) {
  return unlink(file_path.c_str()) == 0;
}

bool UnlinkPath(const Path &file_path) {
  return UnlinkPath(file_path.AsNativePath());
}

bool PathExists(const string& path) {
  return access(path.c_str(), F_OK) == 0;
}

bool PathExists(const Path &path) { return PathExists(path.AsNativePath()); }

string MakeCanonical(const char *path) {
  char *resolved_path = realpath(path, NULL);
  if (resolved_path == NULL) {
    return "";
  } else {
    string ret = resolved_path;
    free(resolved_path);
    return ret;
  }
}

static bool CanAccess(const string &path, bool read, bool write, bool exec) {
  int mode = 0;
  if (read) {
    mode |= R_OK;
  }
  if (write) {
    mode |= W_OK;
  }
  if (exec) {
    mode |= X_OK;
  }
  return access(path.c_str(), mode) == 0;
}

bool CanReadFile(const std::string &path) {
  return !IsDirectory(path) && CanAccess(path, true, false, false);
}

bool CanReadFile(const Path &path) {
  return CanReadFile(path.AsNativePath());
}

bool CanExecuteFile(const std::string &path) {
  return !IsDirectory(path) && CanAccess(path, false, false, true);
}

bool CanExecuteFile(const Path &path) {
  return CanExecuteFile(path.AsNativePath());
}

bool CanAccessDirectory(const std::string &path) {
  return IsDirectory(path) && CanAccess(path, true, true, true);
}

bool CanAccessDirectory(const Path &path) {
  return CanAccessDirectory(path.AsNativePath());
}

bool IsDirectory(const string& path) {
  struct stat buf;
  return stat(path.c_str(), &buf) == 0 && S_ISDIR(buf.st_mode);
}

bool IsDirectory(const Path &path) { return IsDirectory(path.AsNativePath()); }

void SyncFile(const string& path) {
  const char* file_path = path.c_str();
  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "failed to open '" << file_path
        << "' for syncing: " << GetLastErrorString();
  }
  if (fsync(fd) < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "failed to sync '" << file_path << "': " << GetLastErrorString();
  }
  close(fd);
}

void SyncFile(const Path &path) { SyncFile(path.AsNativePath()); }

class PosixFileMtime : public IFileMtime {
 public:
  PosixFileMtime()
      : near_future_(GetFuture(9)),
        distant_future_({GetFuture(10), GetFuture(10)}) {}

  bool IsUntampered(const Path &path) override;
  bool SetToNow(const Path &path) override;
  bool SetToDistantFuture(const Path &path) override;

 private:
  // 9 years in the future.
  const time_t near_future_;
  // 10 years in the future.
  const struct utimbuf distant_future_;

  static bool Set(const Path &path, const struct utimbuf &mtime);
  static time_t GetNow();
  static time_t GetFuture(unsigned int years);
};

bool PosixFileMtime::IsUntampered(const Path &path) {
  struct stat buf;
  if (stat(path.AsNativePath().c_str(), &buf)) {
    return false;
  }

  // Compare the mtime with `near_future_`, not with `GetNow()` or
  // `distant_future_`.
  // This way we don't need to call GetNow() every time we want to compare and
  // we also don't need to worry about potentially unreliable time equality
  // check (in case it uses floats or something crazy).
  return S_ISDIR(buf.st_mode) || (buf.st_mtime > near_future_);
}

bool PosixFileMtime::SetToNow(const Path &path) {
  time_t now(GetNow());
  struct utimbuf times = {now, now};
  return Set(path, times);
}

bool PosixFileMtime::SetToDistantFuture(const Path &path) {
  return Set(path, distant_future_);
}

bool PosixFileMtime::Set(const Path &path, const struct utimbuf &mtime) {
  return utime(path.AsNativePath().c_str(), &mtime) == 0;
}

time_t PosixFileMtime::GetNow() {
  time_t result = time(NULL);
  if (result == -1) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "time(NULL) failed: " << GetLastErrorString();
  }
  return result;
}

time_t PosixFileMtime::GetFuture(unsigned int years) {
  return GetNow() + 3600 * 24 * 365 * years;
}

IFileMtime *CreateFileMtime() { return new PosixFileMtime(); }

// mkdir -p path. Returns true if the path was created or already exists and
// could
// be chmod-ed to exactly the given permissions. If final part of the path is a
// symlink, this ensures that the destination of the symlink has the desired
// permissions. It also checks that the directory or symlink is owned by us.
// On failure, this returns false and sets errno.
bool MakeDirectories(const string &path, unsigned int mode) {
  return MakeDirectories(path, mode, true);
}

bool MakeDirectories(const Path &path, unsigned int mode) {
  return MakeDirectories(path.AsNativePath(), mode);
}

string GetCwd() {
  char cwdbuf[PATH_MAX];
  if (getcwd(cwdbuf, sizeof cwdbuf) == NULL) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "getcwd() failed: " << GetLastErrorString();
  }
  return string(cwdbuf);
}

bool ChangeDirectory(const string& path) {
  return chdir(path.c_str()) == 0;
}

void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(path.c_str())) == NULL) {
    // This is not a directory or it cannot be opened.
    return;
  }

  while ((ent = readdir(dir)) != NULL) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) {
      continue;
    }

    string filename(blaze_util::JoinPath(path, ent->d_name));
    bool is_directory;
// 'd_type' field isn't part of the POSIX spec.
#ifdef _DIRENT_HAVE_D_TYPE
    if (ent->d_type != DT_UNKNOWN) {
      is_directory = (ent->d_type == DT_DIR);
    } else  // NOLINT (the brace is on the next line)
#endif
      {
        struct stat buf;
        if (lstat(filename.c_str(), &buf) == -1) {
          BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
              << "stat failed for filename '" << filename
              << "': " << GetLastErrorString();
        }
        is_directory = S_ISDIR(buf.st_mode);
      }

      consume->Consume(filename, is_directory);
    }

    closedir(dir);
  }

}  // namespace blaze_util
