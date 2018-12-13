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

#include <deque>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "src/main/cpp/workspace_layout.h"

namespace blaze {

// Single option in an rc file.
struct RcOption {
  // Keep a pointer to the path string to avoid copying it over and over.
  std::string* source_path;
  std::string option;
};

// Reads and parses a single rc file with all its imports.
class RcFile {
 public:
  // Constructs a parsed rc file object, or returns a nullptr and sets the
  // error and error text on failure.
  enum class ParseError { NONE, UNREADABLE_FILE, INVALID_FORMAT, IMPORT_LOOP };
  static std::unique_ptr<RcFile> Parse(
      std::string filename, const WorkspaceLayout* workspace_layout,
      std::string workspace, ParseError* error, std::string* error_text);

  // Returns all relevant rc sources for this file (including itself).
  const std::deque<std::string>& canonical_source_paths() const {
    return canonical_rcfile_paths_;
  }

  // Command -> all options for that command (in order of appearance).
  using OptionMap = std::unordered_map<std::string, std::vector<RcOption>>;
  const OptionMap& options() const { return options_; }

 private:
  RcFile(std::string filename, const WorkspaceLayout* workspace_layout,
         std::string workspace);
  // Don't allow copying or moving because it can be tricky with the RcOption
  // string pointers.
  RcFile(const RcFile&) = delete;
  RcFile& operator=(const RcFile&) = delete;

  // Recursive call to parse a file and its imports.
  ParseError ParseFile(const std::string& filename,
                       std::deque<std::string>* import_stack,
                       std::string* error_text);

  const std::string filename_;

  // Workspace definition.
  const WorkspaceLayout* const workspace_layout_;
  const std::string workspace_;

  // Full closure of rcfile paths imported from this file (including itself).
  // These are all canonical paths, created with blaze_util::MakeCanonical.
  // This also means all of these paths should exist.
  //
  // The RcOption structs point to the strings in here so they need to be stored
  // in a container that offers stable pointers, like a deque (and not vector).
  std::deque<std::string> canonical_rcfile_paths_;
  // All options parsed from the file.
  OptionMap options_;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_RC_FILE_H_
