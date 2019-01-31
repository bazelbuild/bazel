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

#include "src/main/cpp/util/path_platform.h"

#include <regex>

#include <limits.h>  // PATH_MAX

#include <string.h>  // strncmp
#include <unistd.h>  // access, open, close, fsync
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"

namespace blaze_util {

std::string ConvertPath(const std::string &path) { return path; }

std::string PathAsJvmFlag(const std::string &path) { return path; }

bool CompareAbsolutePaths(const std::string &a, const std::string &b) {
  return a == b;
}

std::pair<std::string, std::string> SplitPath(const std::string &path) {
  size_t pos = path.rfind('/');

  // Handle the case with no '/' in 'path'.
  if (pos == std::string::npos) return std::make_pair("", path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(std::string(path, 0, 1), std::string(path, 1));

  return std::make_pair(std::string(path, 0, pos), std::string(path, pos + 1));
}

bool IsDevNull(const char *path) {
  return path != NULL && *path != 0 && strncmp("/dev/null\0", path, 10) == 0;
}

bool IsRootDirectory(const std::string &path) {
  return path.size() == 1 && path[0] == '/';
}

bool IsAbsolute(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string MakeAbsolute(const std::string &path) {
  if (blaze_util::IsAbsolute(path) || path.empty()) {
    return path;
  }

  return JoinPath(blaze_util::GetCwd(), path);
}

std::string ResolveEnvvars(const std::string &path) {
  std::string result = path;
  static std::regex env("\\$\\{([^}]+)}");
  std::smatch m;
  while (std::regex_search(result, m, env)) {
    const char *value = getenv(m[1].str().c_str());
    const std::string replacement = std::string(value ? value : "");
    result.replace(m.position(0), m.length(0), replacement);
  }
  return result;
}

std::string MakeAbsoluteAndResolveEnvvars(const std::string &path) {
  return MakeAbsolute(ResolveEnvvars(path));
}

}  // namespace blaze_util
