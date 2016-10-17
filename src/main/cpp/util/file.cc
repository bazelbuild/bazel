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
#include "src/main/cpp/util/file.h"

#include <limits.h>  // PATH_MAX
#include <sys/stat.h>
#include <cstdlib>
#include <vector>

#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/strings.h"

using std::pair;

namespace blaze_util {

using std::string;

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

}  // namespace blaze_util
