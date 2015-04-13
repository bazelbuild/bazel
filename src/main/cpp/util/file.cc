// Copyright 2014 Google Inc. All rights reserved.
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
#include "util/file.h"

#include <errno.h>   // EINVAL
#include <limits.h>  // PATH_MAX
#include <sys/stat.h>
#include <unistd.h>  // access
#include <cstdlib>
#include <vector>

#include "blaze_exit_code.h"
#include "util/errors.h"
#include "util/strings.h"

using std::pair;

namespace blaze_util {

pair<string, string> SplitPath(const string &path) {
  size_t pos = path.rfind('/');

  // Handle the case with no '/' in 'path'.
  if (pos == string::npos) return std::make_pair("", path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0) return std::make_pair(string(path, 0, 1), string(path, 1));

  return std::make_pair(string(path, 0, pos), string(path, pos + 1));
}

string Dirname(const string &path) {
  return SplitPath(path).first;
}

string Basename(const string &path) {
  return SplitPath(path).second;
}

string JoinPath(const string &path1, const string &path2) {
  if (path1.empty()) {
    // "" + "/bar"
    return path2;
  }

  if (path1[path1.size() - 1] == '/') {
    if (path2.find('/') == 0) {
      // foo/ + /bar
      return path1 + path2.substr(1);
    } else {
      // foo/ + bar
      return path1 + path2;
    }
  } else {
    if (path2.find('/') == 0) {
      // foo + /bar
      return path1 + path2;
    } else {
      // foo + bar
      return path1 + "/" + path2;
    }
  }
}

string Which(const string &executable) {
  string path(getenv("PATH"));
  if (path.empty()) {
    die(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR,
        "Could not get PATH to find %s", executable.c_str());
  }

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

}  // namespace blaze_util
