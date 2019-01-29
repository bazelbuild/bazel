// Copyright 2016 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_H_ 1

#include <string>

#include "src/tools/singlejar/port.h"

/*
 * A mapped read-only file with auto closing.
 *
 * MappedFile::Open maps a file with specified name to memory as read-only.
 * It is assumed that the address space is large enough for that.
 * MappedFile::Close deletes the mapping. The destructor calls it, too.
 * A predictable set of methods provide conversion between file offsets and
 * mapped addresses, returns map size, etc.
 */
class MappedFile {
 public:
  MappedFile();

  ~MappedFile() { Close(); }

  bool Open(const std::string &path);

  void Close();

  bool mapped(const void *addr) const {
    return mapped_start_ <= addr && addr < mapped_end_;
  }

  const unsigned char *start() const { return mapped_start_; }
  const unsigned char *end() const { return mapped_end_; }
  const unsigned char *address(off64_t offset) const {
    return mapped_start_ + offset;
  }
  off64_t offset(const void *address) const {
    return reinterpret_cast<const unsigned char *>(address) - mapped_start_;
  }

#ifndef _WIN32
  // Not used on Windows, only in Google's own code. Don't add more usage of it.
  // It is not available on Windows because Windows' implementation does not
  // use fd at all and adding it would just make the implementation too
  // complicated.
  int fd() const { return fd_; }
#endif

  size_t size() const { return mapped_end_ - mapped_start_; }
  bool is_open() const;

 private:
  unsigned char *mapped_start_;
  unsigned char *mapped_end_;
#ifdef _WIN32
  /* HANDLE */ void *hFile_;
  /* HANDLE */ void *hMapFile_;
#else
  int fd_;
#endif
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_H_
