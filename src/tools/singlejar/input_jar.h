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

#include <inttypes.h>
#include <stdlib.h>

#include "src/tools/singlejar/diag.h"
#include "src/tools/singlejar/mapped_file.h"
#include "src/tools/singlejar/zip_headers.h"

/*
 * An input jar. The usage pattern is:
 *   InputJar input_jar("path/to/file");
 *   if (!input_jar.Open()) { fail...}
 *   CDH *dir_entry;
 *   LH *local_header;
 *   while (dir_entry = input_jar.NextExtry(&local_header)) {
 *     // process entry.
 *   }
 *   input_jar.Close(); // actually, called by destructor, too.
 */
class InputJar {
 public:
  InputJar() : path_(nullptr) {}

  ~InputJar() { Close(); }

  int fd() const { return mapped_file_.fd(); }

  // Opens the file, memory maps it and locates Central Directory.
  bool Open(const char *path) {
    if (path_ != nullptr) {
      diag_errx(1, "%s:%d: This instance is already handling %s\n", __FILE__,
                __LINE__, path_);
    }
    if (!mapped_file_.Open(path)) {
      diag_warn("%s:%d: Cannot open input jar %s", __FILE__, __LINE__, path);
      mapped_file_.Close();
      return false;
    }
    if (mapped_file_.size() < sizeof(ECD)) {
      diag_warnx(
          "%s:%d: %s is only %ld bytes long, should be at least %lu bytes long",
          __FILE__, __LINE__, path_, mapped_file_.size(), sizeof(ECD));
      mapped_file_.Close();
      return false;
    }

    // Now locate End of Central Directory (ECD) record.
    const char *ecd_min = mapped_file_.end() - 65536 - sizeof(ECD);
    if (ecd_min < mapped_file_.start()) {
      ecd_min = mapped_file_.start();
    }

    const ECD *ecd = nullptr;
    for (const char *ecd_ptr = mapped_file_.end() - sizeof(ECD);
         ecd_ptr >= ecd_min; --ecd_ptr) {
      ecd = reinterpret_cast<const ECD *>(ecd_ptr);
      if (ecd->is() && ecd) {
        break;
      }
    }
    if (!ecd) {
      diag_warnx("%s:%d: Cannot locate ECD record in %s", __FILE__, __LINE__,
                 path);
      mapped_file_.Close();
      return false;
    }
    uint64_t offset_to_dir = ecd->cen_offset32();
    if (offset_to_dir == 0xFFFFFFFF) {
      const ECD64 *ecd64 = reinterpret_cast<const ECD64 *>(
          mapped_file_.address(ecd->ecd64_offset()));
      offset_to_dir = ecd64->cen_offset();
    }
    cdh_ = reinterpret_cast<const CDH *>(mapped_file_.address(offset_to_dir));
    if (!cdh_->is()) {
      diag_warnx("in %s, expected central file header signature at 0x%" PRIx64,
                 path, offset_to_dir);
      mapped_file_.Close();
      return false;
    }
    path_ = strdup(path);
    return true;
  }

  // Returns the next Central Directory Header or NULL.
  const CDH *NextEntry(const LH **local_header_ptr) {
    if (!path_) {
      diag_errx(1, "%s:%d: call Open() first!", __FILE__, __LINE__);
    }
    if (!cdh_->is()) {
      return nullptr;
    }
    const CDH *current_cdh = cdh_;
    const uint8_t *new_cdr = byte_ptr(cdh_) + cdh_->size();
    if (!mapped_file_.mapped(new_cdr)) {
      diag_errx(
          1,
          "Bad directory record at offset 0x%" PRIx64 " of %s\n"
          "file name length = %u, extra_field length = %u, comment length = %u",
          CentralDirectoryRecordOffset(cdh_), path_, cdh_->file_name_length(),
          cdh_->extra_fields_length(), cdh_->comment_length());
    }
    cdh_ = reinterpret_cast<const CDH *>(new_cdr);
    *local_header_ptr = LocalHeader(current_cdh);
    return current_cdh;
  }

  // Closes the file.
  bool Close() {
    mapped_file_.Close();
    if (path_ != nullptr) {
      free(path_);
      path_ = nullptr;
    }
    return true;
  }

  uint64_t CentralDirectoryRecordOffset(const void *cdr) const {
    return mapped_file_.offset(static_cast<const char *>(cdr));
  }

  const LH *LocalHeader(const CDH *cdh) const {
    return reinterpret_cast<const LH *>(
        mapped_file_.address(cdh->local_header_offset()));
  }

 private:
  char *path_;
  MappedFile mapped_file_;
  const CDH *cdh_;  // current directory entry
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_
