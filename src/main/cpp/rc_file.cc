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

#include "src/main/cpp/rc_file.h"

#include <algorithm>
#include <utility>

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/strings.h"
#include "src/main/cpp/workspace_layout.h"

namespace blaze {

using std::deque;
using std::string;
using std::vector;

static constexpr const char* kCommandImport = "import";
static constexpr const char* kCommandTryImport = "try-import";

RcFile::RcFile(string filename, const WorkspaceLayout* workspace_layout,
               string workspace)
    : filename_(std::move(filename)),
      workspace_layout_(workspace_layout),
      workspace_(std::move(workspace)) {}

/*static*/ std::unique_ptr<RcFile> RcFile::Parse(
    std::string filename, const WorkspaceLayout* workspace_layout,
    std::string workspace, ParseError* error, std::string* error_text) {
  std::unique_ptr<RcFile> rcfile(new RcFile(
      std::move(filename), workspace_layout, std::move(workspace)));
  deque<string> initial_import_stack = {rcfile->filename_};
  *error = rcfile->ParseFile(
      rcfile->filename_, &initial_import_stack, error_text);
  return (*error == ParseError::NONE) ? std::move(rcfile) : nullptr;
}

RcFile::ParseError RcFile::ParseFile(const string& filename,
                                     deque<string>* import_stack,
                                     string* error_text) {
  BAZEL_LOG(INFO) << "Parsing the RcFile " << filename;
  string contents;
  string error_message;
  if (!blaze_util::ReadFile(filename, &contents, &error_message)) {
    blaze_util::StringPrintf(error_text,
                             "Unexpected error reading config file '%s': %s",
                             filename.c_str(), error_message.c_str());
    return ParseError::UNREADABLE_FILE;
  }
  const std::string canonical_filename =
      blaze_util::MakeCanonical(filename.c_str());

  int rcfile_index = canonical_rcfile_paths_.size();
  canonical_rcfile_paths_.push_back(canonical_filename);

  // A '\' at the end of a line continues the line.
  blaze_util::Replace("\\\r\n", "", &contents);
  blaze_util::Replace("\\\n", "", &contents);

  vector<string> lines = blaze_util::Split(contents, '\n');
  for (string& line : lines) {
    blaze_util::StripWhitespace(&line);

    // Check for an empty line.
    if (line.empty()) {
      continue;
    }

    vector<string> words;

    // This will treat "#" as a comment, and properly
    // quote single and double quotes, and treat '\'
    // as an escape character.
    // TODO(bazel-team): This function silently ignores
    // dangling backslash escapes and missing end-quotes.
    blaze_util::Tokenize(line, '#', &words);

    if (words.empty()) {
      // Could happen if line starts with "#"
      continue;
    }

    string command = words[0];

    if (command == kCommandImport || command == kCommandTryImport) {
      if (words.size() != 2 ||
          (words[1].compare(0, workspace_layout_->WorkspacePrefixLength,
                            workspace_layout_->WorkspacePrefix) == 0 &&
           !workspace_layout_->WorkspaceRelativizeRcFilePath(workspace_,
                                                             &words[1]))) {
        blaze_util::StringPrintf(
            error_text,
            "Invalid import declaration in config file '%s': '%s'"
            " (are you in your source checkout/WORKSPACE?)",
            canonical_filename.c_str(), line.c_str());
        return ParseError::INVALID_FORMAT;
      }
      if (std::find(import_stack->begin(), import_stack->end(), words[1]) !=
          import_stack->end()) {
        string loop;
        for (const string& imported_rc : *import_stack) {
          loop += "  " + imported_rc + "\n";
        }
        loop += "  " + words[1] + "\n";  // Include the loop.
        blaze_util::StringPrintf(error_text,
            "Import loop detected:\n%s", loop.c_str());
        return ParseError::IMPORT_LOOP;
      }

      import_stack->push_back(words[1]);
      ParseError parse_error = ParseFile(words[1], import_stack, error_text);
      if (parse_error != ParseError::NONE) {
        if (parse_error == ParseError::UNREADABLE_FILE &&
            command == kCommandTryImport) {
          // For try-import, we ignore it if we couldn't find a file.
          BAZEL_LOG(INFO) << "Skipped optional import of " << words[1]
                          << ", the specified rc file either does not exist or "
                             "is not readable.";
          *error_text = "";
        } else {
          // Files that are there but are malformed or introduce a loop are
          // still a problem, though, so perpetuate those errors as we would
          // for a normal import statement.
          return parse_error;
        }
      }
      import_stack->pop_back();
    } else {
      auto words_it = words.begin();
      words_it++;  // Advance past command.
      for (; words_it != words.end(); words_it++) {
        options_[command].push_back({*words_it, rcfile_index});
      }
    }
  }

  return ParseError::NONE;
}

}  // namespace blaze
