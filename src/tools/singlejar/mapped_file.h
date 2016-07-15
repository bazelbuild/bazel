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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "src/tools/singlejar/diag.h"

/*
 * A mapped read-only file with auto closing.
 *
 * MappedFile::Open maps a file with specified name to memory as read-only.
 * It is assumed that the address space is large enough for that.
 * MappedFile::Close deletes the mapping. The destructor calls it, too.
 * A predictable set of methods provide conversion between file offsets and
 * mapped addresses, returns map size, etc.
 *
 * The implementation is 64-bit Linux or OSX specific.
 */
#if !((defined(__linux) || defined(__APPLE__)) && __SIZEOF_POINTER__ == 8)
#error This code for 64 bit Unix.
#endif

class MappedFile {
 public:
  MappedFile() : mapped_start_(nullptr), mapped_end_(nullptr), fd_(-1) {}

  ~MappedFile() { Close(); }

  bool Open(const char *filename) {
    if (is_open()) {
      diag_errx(1, "%s:%d: This instance is already open", __FILE__, __LINE__);
    }
    if ((fd_ = open(filename, O_RDONLY)) < 0) {
      diag_warn("%s:%d: open %s:", __FILE__, __LINE__, filename);
      return false;
    }
    // Map the file, even if it is empty (in which case allocate 1 byte to it).
    struct stat st;
    if (fstat(fd_, &st) ||
        (mapped_start_ = static_cast<char *>(
             mmap(nullptr, st.st_size ? st.st_size : 1, PROT_READ, MAP_PRIVATE,
                  fd_, 0))) == MAP_FAILED) {
      diag_warn("%s:%d: mmap %s:", __FILE__, __LINE__, filename);
      close(fd_);
      fd_ = -1;
      return false;
    }
    mapped_end_ = mapped_start_ + st.st_size;
    return true;
  }

  void Close() {
    if (is_open()) {
      munmap(mapped_start_, mapped_end_ - mapped_start_);
      mapped_start_ = mapped_end_ = nullptr;
      close(fd_);
      fd_ = -1;
    }
  }

  bool mapped(const void *addr) const {
    return mapped_start_ <= addr && addr < mapped_end_;
  }

  const char *start() const { return mapped_start_; }
  const char *end() const { return mapped_end_; }
  const char *address(off_t offset) const { return mapped_start_ + offset; }
  off_t offset(const char *address) const { return address - mapped_start_; }
  int fd() const { return fd_; }
  size_t size() const { return mapped_end_ - mapped_start_; }
  bool is_open() { return fd_ >= 0; }

 private:
  char *mapped_start_;
  char *mapped_end_;
  int fd_;
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_H_
