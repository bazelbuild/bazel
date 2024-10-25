// Copyright 2024 The Bazel Authors. All rights reserved.
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

#include "src/main/cpp/option_processor-internal.h"

#include <algorithm>
#include <string>

#include "src/main/cpp/util/strings.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace blaze::internal {

#if defined(__CYGWIN__)

static void PreprocessEnvString(std::string *env_str) {
  int pos = env_str->find_first_of('=');
  if (pos == string::npos) return;
  std::string name = env_str->substr(0, pos);
  if (name == "PATH") {
    env_str->assign("PATH=" + env_str->substr(pos + 1));
  } else if (name == "TMP") {
    // A valid Windows path "c:/foo" is also a valid Unix path list of
    // ["c", "/foo"] so must use ConvertPath here. See GitHub issue #1684.
    env_str->assign("TMP=" + blaze_util::ConvertPath(env_str->substr(pos + 1)));
  }
}

#else // not defined(__CYGWIN__)

static void PreprocessEnvString(std::string *env_str) {
  static constexpr const char *vars_to_uppercase[] = {"PATH", "SYSTEMROOT",
                                                      "SYSTEMDRIVE",
                                                      "TEMP", "TEMPDIR", "TMP"};

  std::size_t pos = env_str->find_first_of('=');
  if (pos == std::string::npos) return;

  std::string name = env_str->substr(0, pos);
  // We do not care about locale. All variable names are ASCII.
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (std::find(std::begin(vars_to_uppercase), std::end(vars_to_uppercase),
                name) != std::end(vars_to_uppercase)) {
    env_str->assign(name + "=" + env_str->substr(pos + 1));
  }
}

#endif  // defined(__CYGWIN__)

static bool IsValidEnvName(const char* p) {
  for (; *p && *p != '='; ++p) {
    if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
          (*p >= '0' && *p <= '9') || *p == '_' || *p == '(' || *p == ')')) {
      return false;
    }
  }
  return true;
}

// Use GetEnvironmentStringsW to get the environment variables to support
// Unicode regardless of the current code page.
std::vector<std::string> GetProcessedEnv() {
  std::vector<std::string> processed_env;
  wchar_t* env = GetEnvironmentStringsW();
  if (env == nullptr) {
    return processed_env;
  }

  for (wchar_t* p = env; *p != L'\0'; p += wcslen(p) + 1) {
    std::string env_str = blaze_util::WstringToCstring(p);
    if (IsValidEnvName(env_str.c_str())) {
      PreprocessEnvString(&env_str);
      processed_env.push_back(std::move(env_str));
    }
  }

  FreeEnvironmentStringsW(env);
  return processed_env;
}

} // namespace blaze::internal
