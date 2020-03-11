// Copyright 2015 The Bazel Authors. All rights reserved.
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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include <algorithm>

#include "third_party/ijar/mapped_file.h"

#define MAX_ERROR 2048

namespace devtools_ijar {

static char errmsg[MAX_ERROR];

struct MappedInputFileImpl {
  size_t discarded_;
  int fd_;
};

MappedInputFile::MappedInputFile(const char* name) {
  impl_ = NULL;
  opened_ = false;

  int fd = open(name, O_RDONLY);
  if (fd < 0) {
    snprintf(errmsg, MAX_ERROR, "open(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  off_t length = lseek(fd, 0, SEEK_END);
  if (length < 0) {
    snprintf(errmsg, MAX_ERROR, "lseek(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  void* buffer = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buffer == MAP_FAILED) {
    snprintf(errmsg, MAX_ERROR, "mmap(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  impl_ = new MappedInputFileImpl();
  impl_->fd_ = fd;
  impl_->discarded_ = 0;
  buffer_ = reinterpret_cast<u1*>(buffer);
  length_ = length;
  opened_ = true;
}

MappedInputFile::~MappedInputFile() {
  delete impl_;
}

void MappedInputFile::Discard(size_t bytes) {
  munmap(buffer_ + impl_->discarded_, bytes);
  impl_->discarded_ += bytes;
}

int MappedInputFile::Close() {
  if (close(impl_->fd_) < 0) {
    snprintf(errmsg, MAX_ERROR, "close(): %s", strerror(errno));
    errmsg_ = errmsg;
    return -1;
  }

  return 0;
}

struct MappedOutputFileImpl {
  int fd_;
  int mmap_length_;
};

MappedOutputFile::MappedOutputFile(const char* name, size_t estimated_size)
    : estimated_size_(estimated_size) {
  impl_ = NULL;
  opened_ = false;
  int fd = open(name, O_CREAT|O_RDWR|O_TRUNC, 0644);
  if (fd < 0) {
    snprintf(errmsg, MAX_ERROR, "open(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  // Create mmap-able sparse file
  if (ftruncate(fd, estimated_size) < 0) {
    snprintf(errmsg, MAX_ERROR, "ftruncate(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  // Ensure that any buffer overflow in JarStripper will result in
  // SIGSEGV or SIGBUS by over-allocating beyond the end of the file.
  size_t mmap_length =
      std::min(static_cast<size_t>(estimated_size + sysconf(_SC_PAGESIZE)),
               std::numeric_limits<size_t>::max());
  void* mapped =
      mmap(NULL, mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    snprintf(errmsg, MAX_ERROR, "mmap(): %s", strerror(errno));
    errmsg_ = errmsg;
    return;
  }

  impl_ = new MappedOutputFileImpl();
  impl_->fd_ = fd;
  impl_->mmap_length_ = mmap_length;
  buffer_ = reinterpret_cast<u1*>(mapped);
  opened_ = true;
}


MappedOutputFile::~MappedOutputFile() {
  delete impl_;
}

int MappedOutputFile::Close(size_t size) {
  if (size > estimated_size_) {
    snprintf(errmsg, MAX_ERROR, "size %zu > estimated size %zu", size,
             estimated_size_);
    errmsg_ = errmsg;
    return -1;
  }
  munmap(buffer_, impl_->mmap_length_);
  if (ftruncate(impl_->fd_, size) < 0) {
    snprintf(errmsg, MAX_ERROR, "ftruncate(): %s", strerror(errno));
    errmsg_ = errmsg;
    return -1;
  }

  if (close(impl_->fd_) < 0) {
    snprintf(errmsg, MAX_ERROR, "close(): %s", strerror(errno));
    errmsg_ = errmsg;
    return -1;
  }

  return 0;
}

}  // namespace devtools_ijar
