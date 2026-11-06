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

#include <cstddef>
#include <cstdint>
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
  InputJar() : cdh_(nullptr), cen_end_(nullptr), preamble_size_(0) {}

  ~InputJar() { Close(); }

#ifndef _WIN32
  // Not used on Windows, only in Google's own code. Don't add more usage of it.
  int fd() const { return mapped_file_.fd(); }
#endif

  // Opens the file, memory maps it and locates Central Directory. populate
  // defaults to true to keep the existing behavior for callers like singlejar
  // that read every entry; set it to false if only the Central Directory will
  // be read. See MappedFile::Open.
  bool Open(const std::string& path, bool populate = true);

  // Creates an input jar from data that's already in memory.
  // Requires a non-empty path for use in diagnostics.
  bool Open(const std::string& path, unsigned char* data, size_t length);

  // Returns the next Central Directory Header or nullptr. If local_header_ptr
  // is non-null, it is set to the entry's (validated) Local Header. Callers
  // that only read the Central Directory should pass nullptr, which avoids
  // touching the local header's page in the mapped file.
  const CDH* NextEntry(const LH** local_header_ptr = nullptr) {
    if (path_.empty()) {
      diag_errx(1, "%s:%d: call Open() first!", __FILE__, __LINE__);
    }
    if (ziph::byte_ptr(cdh_) >= cen_end_) {
      return nullptr;
    }
    if (static_cast<size_t>(cen_end_ - ziph::byte_ptr(cdh_)) < sizeof(CDH) ||
        !cdh_->is()) {
      diag_errx(1, "Bad directory record at offset 0x%" PRIx64 " of %s",
                CentralDirectoryRecordOffset(cdh_), path_.c_str());
    }
    const CDH* current_cdh = cdh_;
    const uint8_t* new_cdr = ziph::byte_ptr(cdh_) + cdh_->size();
    if (new_cdr > cen_end_ || !mapped_file_.mapped(new_cdr)) {
      diag_errx(
          1,
          "Bad directory record at offset 0x%" PRIx64
          " of %s\n"
          "file name length = %u, extra_field length = %u, comment length = %u",
          CentralDirectoryRecordOffset(cdh_), path_.c_str(),
          cdh_->file_name_length(), cdh_->extra_fields_length(),
          cdh_->comment_length());
    }
    cdh_ = reinterpret_cast<const CDH*>(new_cdr);
    if (local_header_ptr != nullptr) {
      *local_header_ptr = LocalHeader(current_cdh);
    }
    return current_cdh;
  }

  // Closes the file.
  bool Close();

  uint64_t CentralDirectoryRecordOffset(const void* cdr) const {
    return mapped_file_.offset(cdr);
  }

  const LH* LocalHeader(const CDH* cdh) const {
    uint64_t lh_offset = cdh->local_header_offset() + preamble_size_;
    if (lh_offset < preamble_size_ || lh_offset > mapped_file_.size() ||
        mapped_file_.size() - lh_offset < sizeof(LH)) {
      diag_errx(1,
                "%s:%d: Bad local header offset 0x%" PRIx64
                " at central directory record offset 0x%" PRIx64 " of %s",
                __FILE__, __LINE__, cdh->local_header_offset(),
                CentralDirectoryRecordOffset(cdh), path_.c_str());
    }
    const LH* lh = reinterpret_cast<const LH*>(
        mapped_file_.address(static_cast<int64_t>(lh_offset)));
    if (!lh->is() || lh->size() > mapped_file_.size() - lh_offset) {
      diag_errx(1, "%s:%d: Bad local header at offset 0x%" PRIx64 " of %s",
                __FILE__, __LINE__, lh_offset, path_.c_str());
    }
    uint64_t max_payload = mapped_file_.size() - (lh_offset + lh->size());
    if (cdh->no_size_in_local_header()) {
      uint64_t compressed_size = cdh->compressed_file_size();
      if (compressed_size > max_payload ||
          (lh->compression_method() == 0 &&
           cdh->uncompressed_file_size() > max_payload) ||
          max_payload - compressed_size < sizeof(DDR)) {
        diag_errx(1,
                  "%s:%d: Bad entry size at local header offset 0x%" PRIx64
                  " of %s",
                  __FILE__, __LINE__, lh_offset, path_.c_str());
      }
      const DDR* ddr =
          reinterpret_cast<const DDR*>(lh->data() + compressed_size);
      size_t ddr_size =
          ddr->size(ziph::zfield_has_ext64(cdh->compressed_file_size32()),
                    ziph::zfield_has_ext64(cdh->uncompressed_file_size32()));
      if (max_payload - compressed_size < ddr_size) {
        diag_errx(1,
                  "%s:%d: Bad data descriptor at local header offset 0x%" PRIx64
                  " of %s",
                  __FILE__, __LINE__, lh_offset, path_.c_str());
      }
    } else {
      if (lh->compressed_file_size() > max_payload ||
          cdh->compressed_file_size() > max_payload ||
          (lh->compression_method() == 0 &&
           (lh->uncompressed_file_size() > max_payload ||
            cdh->uncompressed_file_size() > max_payload))) {
        diag_errx(1,
                  "%s:%d: Bad entry size at local header offset 0x%" PRIx64
                  " of %s",
                  __FILE__, __LINE__, lh_offset, path_.c_str());
      }
    }
    return lh;
  }

  uint64_t LocalHeaderOffset(const LH* lh) const {
    return mapped_file_.offset(lh);
  }

  const uint8_t* mapped_start() const { return mapped_file_.address(0); }

 private:
  bool LocateCentralDirectory(const std::string& path);

  std::string path_;
  MappedFile mapped_file_;
  const CDH* cdh_;          // current directory entry
  const uint8_t* cen_end_;  // end of central directory
  uint64_t preamble_size_;  // Bytes before the Zip proper.
};

#endif  //  BAZEL_SRC_TOOLS_SINGLEJAR_INPUT_JAR_H_
