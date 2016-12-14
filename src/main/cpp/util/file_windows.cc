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
#include "src/main/cpp/util/file_platform.h"

#include <windows.h>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file.h"

namespace blaze_util {

using std::string;

class WindowsPipe : public IPipe {
 public:
  bool Send(void* buffer, int size) override {
    // TODO(bazel-team): implement this.
    pdie(255, "blaze_util::WindowsPipe::Send is not yet implemented");
    return false;
  }

  int Receive(void* buffer, int size) override {
    // TODO(bazel-team): implement this.
    pdie(255, "blaze_util::WindowsPipe::Receive is not yet implemented");
    return 0;
  }
};

IPipe* CreatePipe() {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::CreatePipe is not implemented on Windows");
  return nullptr;
}

bool ReadFile(const string& filename, string* content, int max_size) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ReadFile is not implemented on Windows");
  return false;
}

bool WriteFile(const void* data, size_t size, const string& filename) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::WriteFile is not implemented on Windows");
  return false;
}

bool UnlinkPath(const string& file_path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::UnlinkPath is not implemented on Windows");
  return false;
}

string Which(const string &executable) {
  pdie(255, "blaze_util::Which is not implemented on Windows");
  return "";
}

bool PathExists(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::PathExists is not implemented on Windows");
  return false;
}

string MakeCanonical(const char *path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::MakeCanonical is not implemented on Windows");
  return "";
}

bool CanAccess(const string& path, bool read, bool write, bool exec) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::CanAccess is not implemented on Windows");
  return false;
}

bool IsDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::IsDirectory is not implemented on Windows");
  return false;
}

void SyncFile(const string& path) {
  // No-op on Windows native; unsupported by Cygwin.
}

time_t GetMtimeMillisec(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetMtimeMillisec is not implemented on Windows");
  return -1;
}

bool SetMtimeMillisec(const string& path, time_t mtime) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::SetMtimeMillisec is not implemented on Windows");
  return false;
}

#ifdef COMPILER_MSVC
bool MakeDirectories(const string& path, unsigned int mode) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze::MakeDirectories is not implemented on Windows");
  return false;
}
#else   // not COMPILER_MSVC
// Runs "stat" on `path`. Returns -1 and sets errno if stat fails or
// `path` isn't a directory. If check_perms is true, this will also
// make sure that `path` is owned by the current user and has `mode`
// permissions (observing the umask). It attempts to run chmod to
// correct the mode if necessary. If `path` is a symlink, this will
// check ownership of the link, not the underlying directory.
static bool GetDirectoryStat(const string& path, mode_t mode,
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

static bool MakeDirectories(const string& path, mode_t mode, bool childmost) {
  if (path.empty() || path == "/") {
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

// mkdir -p path. Returns true if the path was created or already exists and
// could
// be chmod-ed to exactly the given permissions. If final part of the path is a
// symlink, this ensures that the destination of the symlink has the desired
// permissions. It also checks that the directory or symlink is owned by us.
// On failure, this returns false and sets errno.
bool MakeDirectories(const string& path, mode_t mode) {
  return MakeDirectories(path, mode, true);
}
#endif  // COMPILER_MSVC

string GetCwd() {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::GetCwd is not implemented on Windows");
  return "";
}

bool ChangeDirectory(const string& path) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ChangeDirectory is not implemented on Windows");
  return false;
}

void ForEachDirectoryEntry(const string &path,
                           DirectoryEntryConsumer *consume) {
  // TODO(bazel-team): implement this.
  pdie(255, "blaze_util::ForEachDirectoryEntry is not implemented on Windows");
}

}  // namespace blaze_util
