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
#include "src/main/cpp/util/file.h"

#include <errno.h>
#include <limits.h>  // PATH_MAX

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/strings.h"

namespace blaze_util {

using std::pair;
using std::string;
using std::vector;

bool ReadFrom(const std::function<int(void *, int)> &read_func, string *content,
              int max_size) {
  content->clear();
  char buf[4096];
  // OPT:  This loop generates one spurious read on regular files.
  while (int r = read_func(
             buf, max_size > 0
                      ? std::min(max_size, static_cast<int>(sizeof buf))
                      : sizeof buf)) {
    if (r == -1) {
      if (errno == EINTR || errno == EAGAIN) continue;
      return false;
    }
    content->append(buf, r);
    if (max_size > 0) {
      if (max_size > r) {
        max_size -= r;
      } else {
        break;
      }
    }
  }
  return true;
}

bool WriteTo(const std::function<int(const void *, size_t)> &write_func,
             const void *data, size_t size) {
  int r = write_func(data, size);
  if (r == -1) {
    return false;
  }
  return r == static_cast<int>(size);
}

bool WriteFile(const std::string &content, const std::string &filename) {
  return WriteFile(content.c_str(), content.size(), filename);
}

string Dirname(const string &path) {
  return SplitPath(path).first;
}

string Basename(const string &path) {
  return SplitPath(path).second;
}

string JoinPath(const string &path1, const string &path2) {
  if (path1.empty()) {
    // "" + "/bar"
    return path2;
  }

  if (path1[path1.size() - 1] == '/') {
    if (path2.find('/') == 0) {
      // foo/ + /bar
      return path1 + path2.substr(1);
    } else {
      // foo/ + bar
      return path1 + path2;
    }
  } else {
    if (path2.find('/') == 0) {
      // foo + /bar
      return path1 + path2;
    } else {
      // foo + bar
      return path1 + "/" + path2;
    }
  }
}

class DirectoryTreeWalker : public DirectoryEntryConsumer {
 public:
  DirectoryTreeWalker(vector<string> *files,
                      _ForEachDirectoryEntry walk_entries)
      : _files(files), _walk_entries(walk_entries) {}

  void Consume(const string &path, bool is_directory) override {
    if (is_directory) {
      Walk(path);
    } else {
      _files->push_back(path);
    }
  }

  void Walk(const string &path) { _walk_entries(path, this); }

 private:
  vector<string> *_files;
  _ForEachDirectoryEntry _walk_entries;
};

void GetAllFilesUnder(const string &path, vector<string> *result) {
  _GetAllFilesUnder(path, result, &ForEachDirectoryEntry);
}

void _GetAllFilesUnder(const string &path,
                       vector<string> *result,
                       _ForEachDirectoryEntry walk_entries) {
  DirectoryTreeWalker(result, walk_entries).Walk(path);
}

}  // namespace blaze_util
