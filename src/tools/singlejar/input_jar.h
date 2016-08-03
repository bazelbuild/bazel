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
      diag_warnx("%s:%d: %s is only 0x%lx"
                 " bytes long, should be at least 0x%lx bytes long",
                 __FILE__, __LINE__, path_, mapped_file_.size(), sizeof(ECD));
      mapped_file_.Close();
      return false;
    }

    // Now locate End of Central Directory (ECD) record.
    auto ecd_min = mapped_file_.end() - 65536 - sizeof(ECD);
    if (ecd_min < mapped_file_.start()) {
      ecd_min = mapped_file_.start();
    }

    const ECD *ecd = nullptr;
    for (auto ecd_ptr = mapped_file_.end() - sizeof(ECD); ecd_ptr >= ecd_min;
         --ecd_ptr) {
      if (reinterpret_cast<const ECD *>(ecd_ptr)->is()) {
        ecd = reinterpret_cast<const ECD *>(ecd_ptr);
        break;
      }
    }
    if (ecd == nullptr) {
      diag_warnx("%s:%d: Cannot locate ECD record in %s", __FILE__, __LINE__,
                 path);
      mapped_file_.Close();
      return false;
    }

    /* Find Central Directory and preamble size. We want to handle the case
     * where a Jar/Zip file contains a preamble (an arbitrary data before the
     * first entry) and 'zip -A' was not called to adjust the offsets, so all
     * the offsets are off by the preamble size. In the 32-bit case (that is,
     * there is no ECD64Locator+ECD64), ECD immediately follows the last CDH,
     * ECD immediately follows the Central Directory, and contains its size, so
     * Central Directory can be found reliably. We then use its stated location,
     * which ECD contains, too, to calculate the preamble size.  In the 64-bit
     * case, there are ECD64 and ECD64Locator records between the end of the
     * Central Directory and the ECD, the calculation is similar, with the
     * exception of the logic to find the actual start of the ECD64.
     * ECD64Locator contains only its position in the file, which is off by
     * preamble size, but does not contain the actual size of ECD64, which in
     * theory is variable (the fixed fields may be followed by some custom data,
     * with the total size saved in ECD64::remaining_size and thus unavailable
     * until we find ECD64.  We assume that the custom data is missing.
     */

    // First, sanity checks.
    uint64_t cen_position = ecd->cen_offset32();
    if (cen_position != 0xFFFFFFFF) {
      if (!mapped_file_.mapped(mapped_file_.address(cen_position))) {
        diag_warnx("%s:%d: %s is corrupt: Central Directory location 0x%" PRIx64
                   " is invalid",
                   __FILE__, __LINE__, path, cen_position);
        mapped_file_.Close();
        return false;
      }
      if (mapped_file_.offset(ecd) <= cen_position) {
        diag_warnx(
            "%s:%d: %s is corrupt: End of Central Directory at 0x%" PRIx64
            " precedes Central Directory at 0x%" PRIx64,
            __FILE__, __LINE__, path, mapped_file_.offset(ecd), cen_position);
        mapped_file_.Close();
        return false;
      }
    }
    uint64_t cen_size = ecd->cen_size32();
    if (cen_size != 0xFFFFFFFF) {
      if (cen_size > mapped_file_.offset(ecd)) {
        diag_warnx("%s:%d: %s is corrupt: Central Directory size 0x%" PRIx64
                   " is too large",
                   __FILE__, __LINE__, path, cen_size);
        mapped_file_.Close();
        return false;
      }
    }

    auto ecd64loc = reinterpret_cast<const ECD64Locator *>(
        byte_ptr(ecd) - sizeof(ECD64Locator));
    if (ecd64loc->is()) {
      auto ecd64 =
          reinterpret_cast<const ECD64 *>(byte_ptr(ecd64loc) - sizeof(ECD64));
      if (!ecd64->is()) {
        diag_warnx(
            "%s:%d: %s is corrupt, expected ECD64 record at offset 0x%" PRIx64
            " is missing",
            __FILE__, __LINE__, path, mapped_file_.offset(ecd64));
        mapped_file_.Close();
        return false;
      }
      cdh_ = reinterpret_cast<const CDH *>(byte_ptr(ecd64) - ecd64->cen_size());
      preamble_size_ = mapped_file_.offset(cdh_) - ecd64->cen_offset();
      // Find CEN and preamble size.
    } else {
      if (cen_size == 0xFFFFFFFF || cen_position == 0xFFFFFFFF) {
        diag_warnx(
            "%s:%d: %s is corrupt, expected ECD64 locator record at "
            "offset 0x%" PRIx64 " is missing",
            __FILE__, __LINE__, path, mapped_file_.offset(ecd64loc));
        return false;
      }
      cdh_ = reinterpret_cast<const CDH *>(byte_ptr(ecd) - cen_size);
      preamble_size_ = mapped_file_.offset(cdh_) - cen_position;
    }
    if (!cdh_->is()) {
      diag_warnx(
          "%s:%d: In %s, expected central file header signature at "
          "offset0x%" PRIx64,
          __FILE__, __LINE__, path, mapped_file_.offset(cdh_));
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
        mapped_file_.address(cdh->local_header_offset() + preamble_size_));
  }

 private:
  char *path_;
  MappedFile mapped_file_;
  const CDH *cdh_;  // current directory entry
  uint64_t preamble_size_;  // Bytes before the Zip proper.
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_
