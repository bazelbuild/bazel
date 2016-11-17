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

#include <sys/stat.h>
#include <dirent.h>  // DIR, dirent, opendir, closedir
#include <limits.h>  // PATH_MAX
#include <stdlib.h>  // getenv
#include <unistd.h>  // access
#include <utime.h>  // utime

#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::pair;
using std::string;

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

time_t GetMtimeMillisec(const string& path) {
  struct stat buf;
  if (stat(path.c_str(), &buf)) {
    return -1;
  } else {
    return buf.st_mtime;
  }
}

bool SetMtimeMillisec(const string& path, time_t mtime) {
  struct utimbuf times = { mtime, mtime };
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
