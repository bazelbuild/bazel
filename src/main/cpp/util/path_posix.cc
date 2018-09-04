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

#include <sstream>
#include <vector>

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

std::string MakeAbsoluteAndResolveWindowsEnvvars(const std::string &path) {
  return MakeAbsolute(path);
}

}  // namespace blaze_util

namespace path {

static bool IsNormalized(const Path::string_type& path) {
  if (path.empty()) {
    return true;
  }
  Path::char_type prev = 0;
  int dots = 0;
  for (const auto& c : path) {
    if (c == '/') {
      if (prev == '/' || dots == 1 || dots == 2) {
        return false;
      }
      dots = 0;
    } else if (c == '.' && (prev == 0 || prev == '/' || prev == '.') &&
               (dots == 0 || dots == 1)) {
      dots++;
    } else {
      dots = -1;
    }
    prev = c;
  }
  return (path.size() == 1 && prev != '.') ||
         (path.size() > 1 && prev != '/' && dots != 1 && dots != 2);
}

static Path::string_type Normalize(const Path::string_type& path) {
  if (IsNormalized(path)) {
    return path;
  }

  static const Path::string_type dot(1, '.');
  static const Path::string_type dotdot(2, '.');

  const bool is_absolute = path[0] == '/';
  bool seg_started = false;
  Path::string_type::size_type seg_start;
  std::vector<Path::string_type> segments;
  for (Path::string_type::size_type i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (seg_started && i > seg_start) {
        Path::string_type segment = path.substr(seg_start, i - seg_start);
        seg_started = false;
        if (segment == dotdot) {
          if (!segments.empty()) {
            segments.pop_back();
          }
        } else if (segment != dot) {
          segments.push_back(segment);
        }
      }
    } else if (!seg_started) {
      seg_started = true;
      seg_start = i;
    }
  }

  if (segments.empty()) {
    return is_absolute ? Path::string_type(1, '/') : Path::string_type();
  }

  bool first = true;
  std::basic_ostringstream<Path::char_type> result;
  if (is_absolute) {
    result << '/';
  }
  for (const auto& s : segments) {
    if (!first) {
      result << '/';
    }
    first = false;
    result << s;
  }
  return result.str();
}

namespace testing {

bool TestOnly_IsNormalized(const Path::string_type& path) {
  return IsNormalized(path);
}

Path::string_type TestOnly_Normalize(const Path::string_type& path) {
  return Normalize(path);
}

}  // namespace testing

}  // namespace path
