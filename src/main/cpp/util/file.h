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
#ifndef BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_

#include <functional>
#include <string>
#include <vector>

namespace blaze_util {

class IPipe {
 public:
  virtual ~IPipe() {}

  // Sends `size` bytes from `buffer` through the pipe.
  // Returns true if `size` is not negative and could send all the data.
  virtual bool Send(const void *buffer, int size) = 0;

  // Receives at most `size` bytes into `buffer` from the pipe.
  // Returns the number of bytes received; sets `errno` upon error.
  // If `size` is negative, returns -1.
  virtual int Receive(void *buffer, int size) = 0;
};

// Replaces 'content' with data read from a source using `read_func`.
// If `max_size` is positive, the method reads at most that many bytes;
// otherwise the method reads everything.
// Returns false on error. Can be called from a signal handler.
bool ReadFrom(const std::function<int(void *, int)> &read_func,
              std::string *content, int max_size = 0);

// Writes `size` bytes from `data` into file 'filename' and makes it executable.
// Returns false on failure, sets errno.
bool WriteFile(const void *data, size_t size, const std::string &filename);

// Writes `size` bytes from `data` into a destination using `write_func`.
// Returns false on failure, sets errno.
bool WriteTo(const std::function<int(const void *, size_t)> &write_func,
             const void *data, size_t size);

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input (i.e., "").
std::string Dirname(const std::string &path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
std::string Basename(const std::string &path);

std::string JoinPath(const std::string &path1, const std::string &path2);

// Lists all files in `path` and all of its subdirectories.
//
// Does not follow symlinks / junctions.
//
// Populates `result` with the full paths of the files. Every entry will have
// `path` as its prefix. If `path` is a file, `result` contains just this file.
void GetAllFilesUnder(const std::string &path,
                      std::vector<std::string> *result);

class DirectoryEntryConsumer;

// Visible for testing only.
typedef void (*_ForEachDirectoryEntry)(const std::string &path,
                                       DirectoryEntryConsumer *consume);

// Visible for testing only.
void _GetAllFilesUnder(const std::string &path,
                       std::vector<std::string> *result,
                       _ForEachDirectoryEntry walk_entries);

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_
