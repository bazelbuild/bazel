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
#include <fcntl.h>   // O_RDONLY
#include <limits.h>  // PATH_MAX
#include <stdlib.h>  // getenv
#include <sys/stat.h>
#include <unistd.h>  // access, open, close, fsync
#include <utime.h>   // utime

#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::pair;
using std::string;

class PosixPipe : public IPipe {
 public:
  PosixPipe(int recv_socket, int send_socket)
      : _recv_socket(recv_socket), _send_socket(send_socket) {}

  PosixPipe() = delete;

  virtual ~PosixPipe() {
    close(_recv_socket);
    close(_send_socket);
  }

  // Sends `size` bytes from `buffer` through the pipe.
  bool Send(void* buffer, size_t size) override {
    return write(_send_socket, buffer, size) == size;
  }

  // Receives at most `size` bytes into `buffer` from the pipe.
  // Returns the number of bytes received; sets `errno` upon error.
  int Receive(void* buffer, size_t size) override {
    return read(_recv_socket, buffer, size);
  }

 private:
  int _recv_socket;
  int _send_socket;
};

IPipe* CreatePipe() {
  int fd[2];
  if (pipe(fd) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "pipe()");
  }

  if (fcntl(fd[0], F_SETFD, FD_CLOEXEC) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "fcntl(F_SETFD, FD_CLOEXEC) failed");
  }

  if (fcntl(fd[1], F_SETFD, FD_CLOEXEC) == -1) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "fcntl(F_SETFD, FD_CLOEXEC) failed");
  }

  return new PosixPipe(fd[0], fd[1]);
}

string Which(const string &executable) {
  char *path_cstr = getenv("PATH");
  if (path_cstr == NULL || path_cstr[0] == '\0') {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Could not get PATH to find %s", executable.c_str());
  }

  string path(path_cstr);
  std::vector<std::string> pieces = blaze_util::Split(path, ':');
  for (auto piece : pieces) {
    if (piece.empty()) {
      piece = ".";
    }

    struct stat file_stat;
    string candidate = blaze_util::JoinPath(piece, executable);
    if (access(candidate.c_str(), X_OK) == 0 &&
        stat(candidate.c_str(), &file_stat) == 0 &&
        S_ISREG(file_stat.st_mode)) {
      return candidate;
    }
  }
  return "";
}

bool PathExists(const string& path) {
  return access(path.c_str(), F_OK) == 0;
}

bool CanAccess(const string& path, bool read, bool write, bool exec) {
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

bool IsDirectory(const string& path) {
  struct stat buf;
  return stat(path.c_str(), &buf) == 0 && S_ISDIR(buf.st_mode);
}

void SyncFile(const string& path) {
// fsync always fails on Cygwin with "Permission denied" for some reason.
#ifndef __CYGWIN__
  const char* file_path = path.c_str();
  int fd = open(file_path, O_RDONLY);
  if (fd < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
         "failed to open '%s' for syncing", file_path);
  }
  if (fsync(fd) < 0) {
    pdie(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR, "failed to sync '%s'",
         file_path);
  }
  close(fd);
#endif
}

time_t GetMtimeMillisec(const string& path) {
  struct stat buf;
  if (stat(path.c_str(), &buf)) {
    return -1;
  } else {
    return buf.st_mtime;
  }
}

bool SetMtimeMillisec(const string& path, time_t mtime) {
  struct utimbuf times = {mtime, mtime};
  return utime(path.c_str(), &times) == 0;
}

string GetCwd() {
  char cwdbuf[PATH_MAX];
  if (getcwd(cwdbuf, sizeof cwdbuf) == NULL) {
    pdie(blaze_exit_code::INTERNAL_ERROR, "getcwd() failed");
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
    if (ent->d_type == DT_UNKNOWN) {
      struct stat buf;
      if (lstat(filename.c_str(), &buf) == -1) {
        die(blaze_exit_code::INTERNAL_ERROR, "stat failed");
      }
      is_directory = S_ISDIR(buf.st_mode);
    } else {
      is_directory = (ent->d_type == DT_DIR);
    }

    consume->Consume(filename, is_directory);
  }

  closedir(dir);
}

}  // namespace blaze_util
