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
#include "tools/runfiles/runfiles.h"

namespace bazel {
namespace runfiles {

using std::string;

namespace {

class RunfilesImpl : public Runfiles {
 public:
  // TODO(laszlocsomor): implement Create(
  //   const string& argv0, function<string(const string&)> env_lookup, string*
  //   error);

  string Rlocation(const string& path) const override;

  // Returns the runtime-location of a given runfile.
  //
  // This method assumes that the caller already validated the `path`. See
  // Runfiles::Rlocation for requirements.
  virtual string RlocationChecked(const string& path) const = 0;

 protected:
  RunfilesImpl() {}
  virtual ~RunfilesImpl() {}
};

// TODO(laszlocsomor): derive a ManifestBased class from RunfilesImpl.

// Runfiles implementation that appends runfiles paths to the runfiles root.
class DirectoryBased : public RunfilesImpl {
 public:
  DirectoryBased(string runfiles_path)
      : runfiles_path_(std::move(runfiles_path)) {}
  string RlocationChecked(const string& path) const override;

 private:
  DirectoryBased(const DirectoryBased&) = delete;
  DirectoryBased(DirectoryBased&&) = delete;
  DirectoryBased& operator=(const DirectoryBased&) = delete;
  DirectoryBased& operator=(DirectoryBased&&) = delete;

  const string runfiles_path_;
};

bool IsAbsolute(const string& path) {
  if (path.empty()) {
    return false;
  }
  char c = path.front();
  return (c == '/') || (path.size() >= 3 &&
                        ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) &&
                        path[1] == ':' && (path[2] == '\\' || path[2] == '/'));
}

string RunfilesImpl::Rlocation(const string& path) const {
  if (path.empty() || path.find("..") != string::npos) {
    return std::move(string());
  }
  if (IsAbsolute(path)) {
    return path;
  }
  return RlocationChecked(path);
}

string DirectoryBased::RlocationChecked(const string& path) const {
  return std::move(runfiles_path_ + "/" + path);
}

}  // namespace

namespace testing {

bool TestOnly_IsAbsolute(const string& path) { return IsAbsolute(path); }

}  // namespace testing

Runfiles* Runfiles::CreateDirectoryBased(const string& directory_path,
                                         string* error) {
  // Note: `error` is intentionally unused because we don't expect any errors
  // here. We expect an `error` pointer so that we may use it in the future if
  // need be, without having to change the API.
  return new DirectoryBased(directory_path);
}

}  // namespace runfiles
}  // namespace bazel
