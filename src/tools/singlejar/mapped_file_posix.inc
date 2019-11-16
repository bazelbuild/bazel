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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_POSIX_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_POSIX_H_ 1

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

#include "src/tools/singlejar/diag.h"

// The implementation is specific to 64-bit Linux / OS X / BSD.
#if !((defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)) && \
      __SIZEOF_POINTER__ == 8)
#error This code for 64 bit Unix.
#endif

MappedFile::MappedFile()
    : mapped_start_(nullptr), mapped_end_(nullptr), fd_(-1) {}

bool MappedFile::Open(const std::string& path) {
  if (is_open()) {
    diag_errx(1, "%s:%d: This instance is already open", __FILE__, __LINE__);
  }
  if ((fd_ = open(path.c_str(), O_RDONLY)) < 0) {
    diag_warn("%s:%d: open %s:", __FILE__, __LINE__, path.c_str());
    return false;
  }
  // Map the file, even if it is empty (in which case allocate 1 byte to it).
  struct stat st;
  if (fstat(fd_, &st) ||
      (mapped_start_ = static_cast<unsigned char *>(
           mmap(nullptr, st.st_size ? st.st_size : 1, PROT_READ, MAP_PRIVATE,
                fd_, 0))) == MAP_FAILED) {
    if (S_ISDIR(st.st_mode)) {
      diag_warn("%s:%d: %s is a directory", __FILE__, __LINE__, path.c_str());
    } else {
      diag_warn("%s:%d: mmap %s:", __FILE__, __LINE__, path.c_str());
    }
    close(fd_);
    fd_ = -1;
    return false;
  }
  mapped_end_ = mapped_start_ + st.st_size;
  return true;
}

void MappedFile::Close() {
  if (is_open()) {
    munmap(mapped_start_, mapped_end_ - mapped_start_);
    mapped_start_ = mapped_end_ = nullptr;
    close(fd_);
    fd_ = -1;
  }
}

bool MappedFile::is_open() const { return fd_ >= 0; }

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_MAPPED_FILE_POSIX_H_
