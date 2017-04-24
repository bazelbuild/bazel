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

#include "src/main/cpp/blaze_util.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/util/port.h"

using blaze_util::die;

namespace blaze {

using std::string;
using std::vector;

const char kServerPidFile[] = "server.pid.txt";
const char kServerPidSymlink[] = "server.pid";

string MakeAbsolute(const string &path) {
  // Check if path is already absolute.
  // TODO(laszlocsomor): remove the "path.empty() ||" clause; empty paths are
  // not absolute!
  if (path.empty() || blaze_util::IsAbsolute(path)) {
    return path;
  }

  return blaze_util::JoinPath(blaze_util::GetCwd(), path);
}

const char* GetUnaryOption(const char *arg,
                           const char *next_arg,
                           const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == NULL) {
    return NULL;
  } else if (value[0] == '=') {
    return value + 1;
  } else if (value[0]) {
    return NULL;  // trailing garbage in key name
  }

  return next_arg;
}

bool GetNullaryOption(const char *arg, const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == NULL) {
    return false;
  } else if (value[0] == '=') {
    die(blaze_exit_code::BAD_ARGV,
        "In argument '%s': option '%s' does not take a value.", arg, key);
  } else if (value[0]) {
    return false;  // trailing garbage in key name
  }

  return true;
}

const char* SearchUnaryOption(const vector<string>& args,
                              const char *key) {
  if (args.empty()) {
    return NULL;
  }

  vector<string>::size_type i = 0;
  for (; i < args.size() - 1; ++i) {
    if (args[i] == "--") {
      return NULL;
    }
    const char* result = GetUnaryOption(args[i].c_str(),
                                        args[i + 1].c_str(),
                                        key);
    if (result != NULL) {
      return result;
    }
  }
  return GetUnaryOption(args[i].c_str(), NULL, key);
}

bool SearchNullaryOption(const vector<string>& args, const char *key) {
  for (vector<string>::size_type i = 0; i < args.size(); i++) {
    if (args[i] == "--") {
      return false;
    }
    if (GetNullaryOption(args[i].c_str(), key)) {
      return true;
    }
  }
  return false;
}

bool VerboseLogging() { return !GetEnv("VERBOSE_BLAZE_CLIENT").empty(); }

// Read the Jvm version from a file descriptor. The read fd
// should contains a similar output as the java -version output.
string ReadJvmVersion(const string& version_string) {
  // try to look out for 'version "'
  static const string version_pattern = "version \"";
  size_t found = version_string.find(version_pattern);
  if (found != string::npos) {
    found += version_pattern.size();
    // If we found "version \"", process until next '"'
    size_t end = version_string.find("\"", found);
    if (end == string::npos) {  // consider end of string as a '"'
      end = version_string.size();
    }
    return version_string.substr(found, end - found);
  }

  return "";
}

bool CheckJavaVersionIsAtLeast(const string &jvm_version,
                               const string &version_spec) {
  vector<string> jvm_version_vect = blaze_util::Split(jvm_version, '.');
  int jvm_version_size = static_cast<int>(jvm_version_vect.size());
  vector<string> version_spec_vect = blaze_util::Split(version_spec, '.');
  int version_spec_size = static_cast<int>(version_spec_vect.size());
  int i;
  for (i = 0; i < jvm_version_size && i < version_spec_size; i++) {
    int jvm = blaze_util::strto32(jvm_version_vect[i].c_str(), NULL, 10);
    int spec = blaze_util::strto32(version_spec_vect[i].c_str(), NULL, 10);
    if (jvm > spec) {
      return true;
    } else if (jvm < spec) {
      return false;
    }
  }
  if (i < version_spec_size) {
    for (; i < version_spec_size; i++) {
      if (version_spec_vect[i] != "0") {
        return false;
      }
    }
  }
  return true;
}

bool IsArg(const string& arg) {
  return blaze_util::starts_with(arg, "-") && (arg != "--help")
      && (arg != "-help") && (arg != "-h");
}

}  // namespace blaze
