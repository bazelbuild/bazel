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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_ 1

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#include <inttypes.h>
#include <stdlib.h>

#include <string>

#include "src/tools/singlejar/diag.h"
#include "src/tools/singlejar/mapped_file.h"
#include "src/tools/singlejar/zip_headers.h"

/*
 * An input jar. The usage pattern is:
 *   InputJar input_jar;
 *   if (!input_jar.Open("path/to/file")) { fail...}
 *   CDH *dir_entry;
 *   LH *local_header;
 *   while (dir_entry = input_jar.NextEntry(&local_header)) {
 *     // process entry.
 *   }
 *   input_jar.Close(); // actually, called by destructor, too.
 */
class InputJar {
 public:
  InputJar() {}

  ~InputJar() { Close(); }

#ifndef _WIN32
  // Used by Google-internal only. Do not add more usage of it.
  int fd() const { return mapped_file.fd(); }
#endif

  // Opens the file, memory maps it and locates Central Directory.
  bool Open(const std::string& path);

  // Returns the next Central Directory Header or NULL.
  const CDH *NextEntry(const LH **local_header_ptr) {
    if (path_.empty()) {
      diag_errx(1, "%s:%d: call Open() first!", __FILE__, __LINE__);
    }
    if (!cdh_->is()) {
      return nullptr;
    }
    const CDH *current_cdh = cdh_;
    const uint8_t *new_cdr = ziph::byte_ptr(cdh_) + cdh_->size();
    if (!mapped_file_.mapped(new_cdr)) {
      diag_errx(
          1,
          "Bad directory record at offset 0x%" PRIx64 " of %s\n"
          "file name length = %u, extra_field length = %u, comment length = %u",
          CentralDirectoryRecordOffset(cdh_), path_.c_str(),
          cdh_->file_name_length(), cdh_->extra_fields_length(),
          cdh_->comment_length());
    }
    cdh_ = reinterpret_cast<const CDH *>(new_cdr);
    *local_header_ptr = LocalHeader(current_cdh);
    return current_cdh;
  }

  // Closes the file.
  bool Close();

  uint64_t CentralDirectoryRecordOffset(const void *cdr) const {
    return mapped_file_.offset(cdr);
  }

  const LH *LocalHeader(const CDH *cdh) const {
    return reinterpret_cast<const LH *>(
        mapped_file_.address(cdh->local_header_offset() + preamble_size_));
  }

  uint64_t LocalHeaderOffset(const LH *lh) const {
    return mapped_file_.offset(lh);
  }

  const uint8_t *mapped_start() const {
    return mapped_file_.address(0);
  }

 private:
  std::string path_;
  MappedFile mapped_file_;
  const CDH *cdh_;  // current directory entry
  uint64_t preamble_size_;  // Bytes before the Zip proper.
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_
