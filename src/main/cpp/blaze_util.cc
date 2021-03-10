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
#include <cassert>
#include <iostream>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/numbers.h"
#include "src/main/cpp/util/path_platform.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"

namespace blaze {

using std::map;
using std::string;
using std::vector;

const char kServerPidFile[] = "server.pid.txt";

const unsigned int kPostShutdownGracePeriodSeconds = 60;

const unsigned int kPostKillGracePeriodSeconds = 10;

const char* GetUnaryOption(const char *arg,
                           const char *next_arg,
                           const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == nullptr) {
    return nullptr;
  } else if (value[0] == '=') {
    return value + 1;
  } else if (value[0]) {
    return nullptr;  // trailing garbage in key name
  }

  return next_arg;
}

bool GetNullaryOption(const char *arg, const char *key) {
  const char *value = blaze_util::var_strprefix(arg, key);
  if (value == nullptr) {
    return false;
  } else if (value[0] == '=') {
    BAZEL_DIE(blaze_exit_code::BAD_ARGV)
        << "In argument '" << arg << "': option '" << key
        << "' does not take a value.";
  } else if (value[0]) {
    return false;  // trailing garbage in key name
  }

  return true;
}

const char* SearchUnaryOption(const vector<string>& args,
                              const char *key, bool warn_if_dupe) {
  if (args.empty()) {
    return nullptr;
  }

  const char* value = nullptr;
  bool found_dupe = false;  // true if 'key' was found twice
  vector<string>::size_type i = 0;

  // Examine the first N-1 arguments. (N-1 because we examine the i'th and
  // i+1'th together, in case a flag is defined "--name value" style and not
  // "--name=value" style.)
  for (; i < args.size() - 1; ++i) {
    if (args[i] == "--") {
      // If the current argument is "--", all following args are target names.
      // If 'key' was not found, 'value' is nullptr and we can return that.
      // If 'key' was found exactly once, then 'value' has the value and again
      // we can return that.
      // If 'key' was found more than once then we could not have reached this
      // line, because we would have broken out of the loop when 'key' was found
      // the second time.
      return value;
    }
    const char* result = GetUnaryOption(args[i].c_str(),
                                        args[i + 1].c_str(),
                                        key);
    if (result != nullptr) {
      // 'key' was found and 'result' has its value.
      if (value) {
        // 'key' was found once before, because 'value' is not empty.
        found_dupe = true;
        break;
      } else {
        // 'key' was not found before, so store the value in 'value'.
        value = result;
      }
    }
  }

  if (value) {
    // 'value' is not empty, so 'key' was found at least once in the first N-1
    // arguments.
    if (warn_if_dupe) {
      if (!found_dupe) {
        // We did not find a duplicate in the first N-1 arguments. Examine the
        // last argument, it may be a duplicate.
        found_dupe = (GetUnaryOption(args[i].c_str(), nullptr, key) != nullptr);
      }
      if (found_dupe) {
        BAZEL_LOG(WARNING) << key << " is given more than once, "
                           << "only the first occurrence is used";
      }
    }
    return value;
  } else {
    // 'value' is empty, so 'key' was not yet found in the first N-1 arguments.
    // If 'key' is in the last argument, we'll parse and return the value from
    // that, and if it isn't, we'll return NULL.
    return GetUnaryOption(args[i].c_str(), nullptr, key);
  }
}

bool SearchNullaryOption(const vector<string>& args,
                         const string& flag_name,
                         const bool default_value) {
  const string positive_flag = "--" + flag_name;
  const string negative_flag = "--no" + flag_name;
  bool result = default_value;
  for (vector<string>::size_type i = 0; i < args.size(); i++) {
    if (args[i] == "--") {
      break;
    }
    if (GetNullaryOption(args[i].c_str(), positive_flag.c_str())) {
      result = true;
    } else if (GetNullaryOption(args[i].c_str(), negative_flag.c_str())) {
      result = false;
    }
  }
  return result;
}

bool IsArg(const string& arg) {
  return blaze_util::starts_with(arg, "-") && (arg != "--help")
      && (arg != "-help") && (arg != "-h");
}

std::string AbsolutePathFromFlag(const std::string& value) {
  if (value.empty()) {
    return blaze_util::GetCwd();
  } else if (!value.empty() && value[0] == '~') {
    return blaze_util::JoinPath(GetHomeDir(), value.substr(1));
  } else {
    return blaze_util::MakeAbsolute(value);
  }
}

void LogWait(unsigned int elapsed_seconds, unsigned int wait_seconds) {
  SigPrintf("WARNING: Waiting for server process to terminate "
            "(waited %d seconds, waiting at most %d)\n",
            elapsed_seconds, wait_seconds);
}

bool AwaitServerProcessTermination(int pid, const blaze_util::Path& output_base,
                                   unsigned int wait_seconds) {
  uint64_t st = GetMillisecondsMonotonic();
  const unsigned int first_seconds = 5;
  bool logged_first = false;
  const unsigned int second_seconds = 10;
  bool logged_second = false;
  const unsigned int third_seconds = 30;
  bool logged_third = false;

  while (VerifyServerProcess(pid, output_base)) {
    TrySleep(100);
    uint64_t elapsed_millis = GetMillisecondsMonotonic() - st;
    if (!logged_first && elapsed_millis > first_seconds * 1000) {
      LogWait(first_seconds, wait_seconds);
      logged_first = true;
    }
    if (!logged_second && elapsed_millis > second_seconds * 1000) {
      LogWait(second_seconds, wait_seconds);
      logged_second = true;
    }
    if (!logged_third && elapsed_millis > third_seconds * 1000) {
      LogWait(third_seconds, wait_seconds);
      logged_third = true;
    }
    if (elapsed_millis > wait_seconds * 1000) {
      SigPrintf("INFO: Waited %d seconds for server process (pid=%d) to"
                " terminate.\n",
                wait_seconds, pid);
      return false;
    }
  }
  return true;
}

// For now, we don't have the client set up to log to a file. If --client_debug
// is passed, however, all BAZEL_LOG statements will be output to stderr.
// If/when we switch to logging these to a file, care will have to be taken to
// either log to both stderr and the file in the case of --client_debug, or be
// ok that these log lines will only go to one stream.
void SetDebugLog(bool enabled) {
  if (enabled) {
    blaze_util::SetLoggingOutputStreamToStderr();
  } else {
    blaze_util::SetLoggingOutputStream(nullptr);
  }
}

bool IsRunningWithinTest() { return ExistsEnv("TEST_TMPDIR"); }

void WithEnvVars::SetEnvVars(const map<string, EnvVarValue>& vars) {
  for (const auto& var : vars) {
    switch (var.second.action) {
      case EnvVarAction::UNSET:
        UnsetEnv(var.first);
        break;

      case EnvVarAction::SET:
        SetEnv(var.first, var.second.value);
        break;

      default:
        assert(false);
    }
  }
}

WithEnvVars::WithEnvVars(const map<string, EnvVarValue>& vars) {
  for (const auto& v : vars) {
    if (ExistsEnv(v.first)) {
      _old_values[v.first] = EnvVarValue(EnvVarAction::SET, GetEnv(v.first));
    } else {
      _old_values[v.first] = EnvVarValue(EnvVarAction::UNSET, "");
    }
  }

  SetEnvVars(vars);
}

WithEnvVars::~WithEnvVars() {
  SetEnvVars(_old_values);
}

}  // namespace blaze
