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
// Usage:
//
//   #include "tools/cpp/runfiles/runfiles.h"
//
//   using bazel::tools::cpp::runfiles::Runfiles;
//
//   int main(int argc, char** argv) {
//     std::string error;
//     std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
//     if (runfiles == nullptr) {
//       ...  // error handling
//     }
//     std::string path(runfiles->Rlocation("io_bazel/src/bazel"));
//     std::ifstream data(path);
//     if (data.is_open()) {
//       ...  // use the runfile
//
// The code above creates a manifest- or directory-based implementations
// depending on it finding a runfiles manifest or -directory near argv[0] or
// finding appropriate environment variables that tell it where to find the
// manifest or directory. See `Runfiles::Create` for more info.
//
// If you want to explicitly create a manifest- or directory-based
// implementation, you can do so as follows:
//
//   std::unique_ptr<Runfiles> runfiles1(
//       Runfiles::CreateManifestBased(path/to/foo.runfiles/MANIFEST", &error));
//
//   std::unique_ptr<Runfiles> runfiles2(
//       Runfiles::CreateDirectoryBased(path/to/foo.runfiles", &error));
//
// If you want to start child processes that also need runfiles, you need to set
// the right environment variables for them:
//
//   std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
//
//   for (const auto i : runfiles->EnvVars()) {
//     setenv(i.first, i.second, 1);
//   }
//   std::string path(runfiles->Rlocation("path/to/binary"));
//   if (!path.empty()) {
//     pid_t child = fork();
//     ...

#ifndef TOOLS_CPP_RUNFILES_RUNFILES_H_
#define TOOLS_CPP_RUNFILES_RUNFILES_H_ 1

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace tools {
namespace cpp {
namespace runfiles {

class Runfiles {
 public:
  virtual ~Runfiles() {}

  // Returns a new `Runfiles` instance.
  //
  // The returned object is either:
  // - manifest-based, meaning it looks up runfile paths from a manifest file,
  //   or
  // - directory-based, meaning it looks up runfile paths under a given
  //   directory path
  //
  // This method:
  // 1. checks the RUNFILES_MANIFEST_FILE or RUNFILES_DIR environment variables;
  //    if either is non-empty, returns a manifest- or directory-based Runfiles
  //    object; otherwise
  // 2. checks if there's a runfiles manifest (argv0 + ".runfiles_manifest") or
  //    runfiles directory (argv0 + ".runfiles") next to this binary; if so,
  //    returns a manifest- or directory-based Runfiles object; otherwise
  // 3. returns nullptr.
  //
  // The manifest-based Runfiles object eagerly reads and caches the whole
  // manifest file upon instantiation; this may be relevant for performance
  // consideration.
  //
  // Returns nullptr on error. If `error` is provided, the method prints an
  // error message into it.
  static Runfiles* Create(const std::string& argv0,
                          std::string* error = nullptr);

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

  // Computes the path of the runfiles manifest and the runfiles directory.
  //
  // If the method finds both a valid manifest and valid directory according to
  // `is_runfiles_manifest` and `is_runfiles_directory`, then the method sets
  // the corresponding values to `out_manifest` and `out_directory` and returns
  // true.
  //
  // If the method only finds a valid manifest or a valid directory, but not
  // both, then it sets the corresponding output variable (`out_manifest` or
  // `out_directory`) to the value while clearing the other output variable. The
  // method still returns true in this case.
  //
  // If the method cannot find either a valid manifest or valid directory, it
  // clears both output variables and returns false.
  static bool PathsFrom(
      const std::string& argv0,
      std::function<std::string(std::string)> env_lookup,
      std::function<bool(const std::string&)> is_runfiles_manifest,
      std::function<bool(const std::string&)> is_runfiles_directory,
      std::string* out_manifest, std::string* out_directory);

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
//
// Create a new Runfiles instance, looking up environment variables using
// `env_lookup`.
//
// Args:
//   argv0: name of the binary; if this string is not empty, then the function
//     looks for a runfiles manifest or directory next to this
//   env_lookup: a function that returns envvar values if an envvar is known, or
//     empty string otherwise
Runfiles* TestOnly_CreateRunfiles(
    const std::string& argv0,
    std::function<std::string(const std::string&)> env_lookup,
    std::string* error);

// For testing only.
// Returns true if `path` is an absolute Unix or Windows path.
// For Windows paths, this function does not regard drive-less absolute paths
// (i.e. absolute-on-current-drive, e.g. "\foo\bar") as absolute and returns
// false for these.
bool TestOnly_IsAbsolute(const std::string& path);

}  // namespace testing
}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // TOOLS_CPP_RUNFILES_RUNFILES_H_
