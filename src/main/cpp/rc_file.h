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
#ifndef BAZEL_SRC_MAIN_CPP_RC_FILE_H_
#define BAZEL_SRC_MAIN_CPP_RC_FILE_H_

#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/workspace_layout.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"

namespace blaze {

// Single option in an rc file.
struct RcOption {
  std::string option;
  // Only keep the index of the source file to avoid copying the paths all over.
  // This index points into the RcFile's canonical_source_paths() vector.
  int source_index;
};

// Reads and parses a single rc file with all its imports.
class RcFile {
 public:
  // Constructs a parsed rc file object, or returns a nullptr and sets the
  // error and error text on failure.
  using ReadFileFn =
      absl::FunctionRef<bool(const std::string&, std::string*, std::string*)>;
  using CanonicalizePathFn = absl::FunctionRef<std::string(const std::string&)>;
  enum class ParseError { NONE, UNREADABLE_FILE, INVALID_FORMAT, IMPORT_LOOP };
  static std::unique_ptr<RcFile> Parse(
      const std::string& filename, const WorkspaceLayout* workspace_layout,
      const std::string& workspace, ParseError* error, std::string* error_text,
      ReadFileFn read_file = &ReadFileDefault,
      CanonicalizePathFn canonicalize_path = &CanonicalizePathDefault);

  // Movable and copyable.
  RcFile(const RcFile&) = default;
  RcFile(RcFile&&) = default;
  RcFile& operator=(const RcFile&) = default;
  RcFile& operator=(RcFile&&) = default;

  // Returns all relevant rc sources for this file (including itself).
  const std::vector<std::string>& canonical_source_paths() const {
    return canonical_rcfile_paths_;
  }

  // Command -> all options for that command (in order of appearance).
  using OptionMap = absl::flat_hash_map<std::string, std::vector<RcOption>>;
  const OptionMap& options() const { return options_; }

 private:
  RcFile() = default;

  // Recursive call to parse a file and its imports.
  ParseError ParseFile(const std::string& filename,
                       const std::string& workspace,
                       const WorkspaceLayout& workspace_layout,
                       ReadFileFn read_file,
                       CanonicalizePathFn canonicalize_path,
                       std::vector<std::string>& import_stack,
                       std::string* error_text);

  static bool ReadFileDefault(const std::string& filename,
                              std::string* contents, std::string* error_msg);
  static std::string CanonicalizePathDefault(const std::string& filename);

  // Full closure of rcfile paths imported from this file (including itself).
  // These are all canonical paths, created with blaze_util::MakeCanonical.
  // This also means all of these paths should exist.
  std::vector<std::string> canonical_rcfile_paths_;
  // All options parsed from the file.
  OptionMap options_;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_RC_FILE_H_
