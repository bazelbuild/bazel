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

// Runfiles lookup library for Bazel-built C++ binaries and tests.
//
// TODO(laszlocsomor): add usage information and examples.

#ifndef BAZEL_SRC_TOOLS_RUNFILES_RUNFILES_H_
#define BAZEL_SRC_TOOLS_RUNFILES_RUNFILES_H_ 1

#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace runfiles {

class Runfiles {
 public:
  virtual ~Runfiles() {}

  // Returns a new manifest-based `Runfiles` instance.
  // Returns nullptr on error. If `error` is provided, the method prints an
  // error message into it.
  static Runfiles* CreateManifestBased(const std::string& manifest_path,
                                       std::string* error = nullptr);

  // Returns a new directory-based `Runfiles` instance.
  // Returns nullptr on error. If `error` is provided, the method prints an
  // error message into it.
  static Runfiles* CreateDirectoryBased(const std::string& directory_path,
                                        std::string* error = nullptr);

  // Returns the runtime path of a runfile.
  //
  // Runfiles are data-dependencies of Bazel-built binaries and tests.
  //
  // The returned path may not be valid. The caller should check the path's
  // validity and that the path exists.
  //
  // The function may return an empty string. In that case the caller can be
  // sure that the Runfiles object does not know about this data-dependency.
  //
  // Args:
  //   path: runfiles-root-relative path of the runfile; must not be empty and
  //     must not contain uplevel references.
  // Returns:
  //   the path to the runfile, which the caller should check for existence, or
  //   an empty string if the method doesn't know about this runfile
  virtual std::string Rlocation(const std::string& path) const = 0;

  // Returns environment variables for subprocesses.
  //
  // The caller should set the returned key-value pairs in the environment of
  // subprocesses in case those subprocesses are also Bazel-built binaries that
  // need to use runfiles.
  virtual std::vector<std::pair<std::string, std::string> > EnvVars() const = 0;

 protected:
  Runfiles() {}

 private:
  Runfiles(const Runfiles&) = delete;
  Runfiles(Runfiles&&) = delete;
  Runfiles& operator=(const Runfiles&) = delete;
  Runfiles& operator=(Runfiles&&) = delete;
};

// The "testing" namespace contains functions that allow unit testing the code.
// Do not use these outside of runfiles_test.cc, they are only part of the
// public API for the benefit of the tests.
// These functions and their interface may change without notice.
namespace testing {

// For testing only.
// Returns true if `path` is an absolute Unix or Windows path.
// For Windows paths, this function does not regard drive-less absolute paths
// (i.e. absolute-on-current-drive, e.g. "\foo\bar") as absolute and returns
// false for these.
bool TestOnly_IsAbsolute(const std::string& path);

}  // namespace testing
}  // namespace runfiles
}  // namespace bazel

#endif  // BAZEL_SRC_TOOLS_RUNFILES_RUNFILES_H_
