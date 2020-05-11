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

// The "srcs_for_embedded_tools" rule in the same package sets the line below to
// include runfiles.h from the correct path. Do not modify the line below.
#include "wrunfiles_src.h"

#ifdef _WIN32
#include <windows.h>
#else  // not _WIN32
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // _WIN32

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <memory>
#endif  // _WIN32

#include "src/main/cpp/util/path_platform.h"

namespace bazel {
namespace tools {
namespace cpp {
namespace wrunfiles {

using std::function;
using std::map;
using std::pair;
using std::vector;

namespace {

bool starts_with(const std::wstring& s, const wchar_t* prefix) {
  if (!prefix || !*prefix) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.find(prefix) == 0;
}

bool contains(const std::wstring& s, const wchar_t* substr) {
  if (!substr || !*substr) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.find(substr) != std::wstring::npos;
}

bool ends_with(const std::wstring& s, const std::wstring& suffix) {
  if (suffix.empty()) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.rfind(suffix) == s.size() - suffix.size();
}

bool IsReadableFile(const std::wstring& path) {
  return std::wifstream(path).is_open();
}

bool IsDirectory(const std::wstring& path) {
#ifdef _WIN32
  DWORD attrs = GetFileAttributesW(path.c_str());
  return (attrs != INVALID_FILE_ATTRIBUTES) &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat buf;
  return stat(path.c_str(), &buf) == 0 && S_ISDIR(buf.st_mode);
#endif
}

bool PathsFrom(const std::wstring& argv0, std::wstring runfiles_manifest_file,
               std::wstring runfiles_dir, std::wstring* out_manifest,
               std::wstring* out_directory);

bool PathsFrom(const std::wstring& argv0, std::wstring runfiles_manifest_file,
               std::wstring runfiles_dir,
               std::function<bool(const std::wstring&)> is_runfiles_manifest,
               std::function<bool(const std::wstring&)> is_runfiles_directory,
               std::wstring* out_manifest, std::wstring* out_directory);

bool ParseManifest(const std::wstring& path, map<std::wstring, std::wstring>* result,
                   std::wstring* error);

}  // namespace

Runfiles* Runfiles::Create(const std::wstring& argv0,
                           const std::wstring& runfiles_manifest_file,
                           const std::wstring& runfiles_dir, std::wstring* error) {
  std::wstring manifest, directory;
  if (!PathsFrom(argv0, runfiles_manifest_file, runfiles_dir, &manifest,
                 &directory)) {
    if (error) {
      std::wostringstream err;
      err << L"ERROR: " << __FILE__ << L"(" << __LINE__
          << L"): cannot find runfiles (argv0=\"" << argv0 << L"\")";
      *error = err.str();
    }
    return nullptr;
  }

  const vector<pair<std::wstring, std::wstring> > envvars = {
      {L"RUNFILES_MANIFEST_FILE", manifest},
      {L"RUNFILES_DIR", directory},
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can
      // pick up RUNFILES_DIR.
      {L"JAVA_RUNFILES", directory}};

  map<std::wstring, std::wstring> runfiles;
  if (!manifest.empty()) {
    if (!ParseManifest(manifest, &runfiles, error)) {
      return nullptr;
    }
  }

  return new Runfiles(std::move(runfiles), std::move(directory),
                      std::move(envvars));
}

bool IsAbsolute(const std::wstring& path) {
  if (path.empty()) {
    return false;
  }
  wchar_t c = path.front();
  return (c == '/' && (path.size() < 2 || path[1] != '/')) ||
         (path.size() >= 3 &&
          ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) &&
          path[1] == ':' && (path[2] == '\\' || path[2] == '/'));
}

std::wstring GetEnv(const std::wstring& key) {
#ifdef _WIN32
  DWORD size = ::GetEnvironmentVariableW(key.c_str(), NULL, 0);
  if (size == 0) {
    return std::wstring();  // unset or empty envvar
  }
  std::unique_ptr<wchar_t[]> value(new wchar_t[size]);
  ::GetEnvironmentVariableW(key.c_str(), value.get(), size);
  return value.get();
#else
  wchar_t* result = getenv(key.c_str());
  return (result == NULL) ? std::wstring() : std::wstring(result);
#endif
}

std::wstring Runfiles::Rlocation(const std::wstring& path) const {
  if (path.empty() || starts_with(path, L"../") || contains(path, L"/..") ||
      starts_with(path, L"./") || contains(path, L"/./") ||
      ends_with(path, L"/.") || contains(path, L"//")) {
    return std::wstring();
  }
  if (IsAbsolute(path)) {
    return path;
  }
  const auto value = runfiles_map_.find(path);
  if (value != runfiles_map_.end()) {
    return value->second;
  }
  if (!directory_.empty()) {
    return directory_ + L"/" + path;
  }
  return L"";
}

namespace {

bool ParseManifest(const std::wstring& path, map<std::wstring, std::wstring>* result,
                   std::wstring* error) {
  std::wifstream stm(path);
  if (!stm.is_open()) {
    if (error) {
      std::wostringstream err;
      err << L"ERROR: " << __FILE__ << L"(" << __LINE__
          << L"): cannot open runfiles manifest \"" << path << L"\"";
      *error = err.str();
    }
    return false;
  }
  std::wstring line;
  std::getline(stm, line);
  size_t line_count = 1;
  while (!line.empty()) {
    std::wstring::size_type idx = line.find_first_of(' ');
    if (idx == std::wstring::npos) {
      if (error) {
        std::wostringstream err;
        err << L"ERROR: " << __FILE__ << L"(" << __LINE__
            << L"): bad runfiles manifest entry in \"" << path << L"\" line #"
            << line_count << L": \"" << line << L"\"";
        *error = err.str();
      }
      return false;
    }
    (*result)[line.substr(0, idx)] = line.substr(idx + 1);
    std::getline(stm, line);
    ++line_count;
  }
  return true;
}

}  // namespace

namespace testing {

bool TestOnly_PathsFrom(const std::wstring& argv0, std::wstring mf, std::wstring dir,
                        function<bool(const std::wstring&)> is_runfiles_manifest,
                        function<bool(const std::wstring&)> is_runfiles_directory,
                        std::wstring* out_manifest, std::wstring* out_directory) {
  return PathsFrom(argv0, mf, dir, is_runfiles_manifest, is_runfiles_directory,
                   out_manifest, out_directory);
}

bool TestOnly_IsAbsolute(const std::wstring& path) { return IsAbsolute(path); }

}  // namespace testing

Runfiles* Runfiles::Create(const std::wstring& argv0, std::wstring* error) {
  return Runfiles::Create(argv0, GetEnv(L"RUNFILES_MANIFEST_FILE"),
                          GetEnv(L"RUNFILES_DIR"), error);
}

Runfiles* Runfiles::CreateForTest(std::wstring* error) {
  return Runfiles::Create(std::wstring(), GetEnv(L"RUNFILES_MANIFEST_FILE"),
                          GetEnv(L"TEST_SRCDIR"), error);
}

namespace {

bool PathsFrom(const std::wstring& argv0, std::wstring mf, std::wstring dir, std::wstring* out_manifest,
               std::wstring* out_directory) {
  return PathsFrom(argv0, mf, dir,
                   [](const std::wstring& path) { return IsReadableFile(path); },
                   [](const std::wstring& path) { return IsDirectory(path); },
                   out_manifest, out_directory);
}

bool PathsFrom(const std::wstring& argv0, std::wstring mf, std::wstring dir,
               function<bool(const std::wstring&)> is_runfiles_manifest,
               function<bool(const std::wstring&)> is_runfiles_directory,
               std::wstring* out_manifest, std::wstring* out_directory) {
  out_manifest->clear();
  out_directory->clear();

  std::string error;
  blaze_util::AsAbsoluteWindowsPath(mf, &mf, &error);
  blaze_util::AsAbsoluteWindowsPath(dir, &dir, &error);

  bool mfValid = is_runfiles_manifest(mf);
  bool dirValid = is_runfiles_directory(dir);

  if (!argv0.empty() && !mfValid && !dirValid) {
    mf = argv0 + L".runfiles/MANIFEST";
    dir = argv0 + L".runfiles";
    mfValid = is_runfiles_manifest(mf);
    dirValid = is_runfiles_directory(dir);
    if (!mfValid) {
      mf = argv0 + L".runfiles_manifest";
      mfValid = is_runfiles_manifest(mf);
    }
  }

  if (!mfValid && !dirValid) {
    return false;
  }

  if (!mfValid) {
    mf = dir + L"/MANIFEST";
    mfValid = is_runfiles_manifest(mf);
    if (!mfValid) {
      mf = dir + L"_manifest";
      mfValid = is_runfiles_manifest(mf);
    }
  }

  if (!dirValid &&
      (ends_with(mf, L".runfiles_manifest") || ends_with(mf, L"/MANIFEST"))) {
    static const size_t kSubstrLen = 9;  // "_manifest" or "/MANIFEST"
    dir = mf.substr(0, mf.size() - kSubstrLen);
    dirValid = is_runfiles_directory(dir);
  }

  if (mfValid) {
    *out_manifest = mf;
  }

  if (dirValid) {
    *out_directory = dir;
  }

  return true;
}

}  // namespace

}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
