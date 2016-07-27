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

#ifndef SRC_TOOLS_SINGLEJAR_COMBINERS_H_
#define SRC_TOOLS_SINGLEJAR_COMBINERS_H_ 1

#include <memory>
#include <string>

#include "src/tools/singlejar/transient_bytes.h"
#include "src/tools/singlejar/zip_headers.h"
#include "src/tools/singlejar/zlib_interface.h"

// An output jar entry consisting of a concatenation of the input jar
// entries. Byte sequences can be appended to it, too.
class Concatenator {
 public:
  Concatenator(const std::string &filename) : filename_(filename) {}

  // Appends the contents of the given input entry.
  bool Merge(const CDH *cdh, const LH *lh) {
    CreateBuffer();
    if (Z_NO_COMPRESSION == lh->compression_method()) {
      buffer_->ReadEntryContents(lh);
    } else if (Z_DEFLATED == lh->compression_method()) {
      if (!inflater_.get()) {
        inflater_.reset(new Inflater());
      }
      buffer_->DecompressEntryContents(cdh, lh, inflater_.get());
    } else {
      errx(2, "%s is neither stored nor deflated", filename_.c_str());
    }
    return true;
  }

  // Returns a point to the buffer  containing Local Header followed by the
  // payload. The caller is responsible of freeing the buffer.
  void *OutputEntry() {
    if (!buffer_.get()) {
      return nullptr;
    }

    // Allocate a contiguous buffer for the local file header and
    // deflated data. We assume that deflate decreases the size, so if
    //  the deflater reports overflow, we just save original data.
    size_t deflated_buffer_size =
        sizeof(LH) + filename_.size() + buffer_->data_size();

    // Huge entry (>4GB) needs Zip64 extension field with 64-bit original
    // and compressed size values.
    uint8_t
        zip64_extension_buffer[sizeof(Zip64ExtraField) + 2 * sizeof(uint64_t)];
    bool huge_buffer = (buffer_->data_size() >= 0xFFFFFFFF);
    if (huge_buffer) {
      deflated_buffer_size += sizeof(zip64_extension_buffer);
    }
    LH *lh = reinterpret_cast<LH *>(malloc(deflated_buffer_size));
    if (lh == nullptr) {
      return nullptr;
    }
    lh->signature();
    lh->version(20);
    lh->bit_flag(0x0);
    lh->last_mod_file_time(1);   // 00:00:01
    lh->last_mod_file_date(33);  // 1980-01-01
    lh->crc32(0x12345678);
    lh->compressed_file_size32(0);
    lh->file_name(filename_.c_str(), filename_.size());

    if (huge_buffer) {
      // Add Z64 extension if this is a huge entry.
      lh->uncompressed_file_size32(0xFFFFFFFF);
      Zip64ExtraField *z64 =
          reinterpret_cast<Zip64ExtraField *>(zip64_extension_buffer);
      z64->signature();
      z64->payload_size(2 * sizeof(uint64_t));
      z64->attr64(0, buffer_->data_size());
      lh->extra_fields(reinterpret_cast<uint8_t *>(z64), z64->size());
    } else {
      lh->uncompressed_file_size32(buffer_->data_size());
      lh->extra_fields(nullptr, 0);
    }

    uint32_t checksum;
    uint64_t compressed_size;
    uint16_t method = buffer_->Write(lh->data(), &checksum, &compressed_size);
    lh->crc32(checksum);
    lh->compression_method(method);
    if (huge_buffer) {
      lh->compressed_file_size32(compressed_size < 0xFFFFFFFF ? compressed_size
                                                              : 0xFFFFFFFF);
      // Not sure if this has to be written in the small case, but it shouldn't
      // hurt.
      const_cast<Zip64ExtraField *>(lh->zip64_extra_field())
          ->attr64(1, compressed_size);
    } else {
      // If original data is <4GB, the compressed one is, too.
      lh->compressed_file_size32(compressed_size);
    }
    return reinterpret_cast<void *>(lh);
  }

  void Append(const char *s, size_t n) {
    CreateBuffer();
    buffer_->Append(reinterpret_cast<const uint8_t *>(s), n);
  }

  void Append(const char *s) { Append(s, strlen(s)); }

  void Append(const std::string &str) { Append(str.c_str(), str.size()); }

  const std::string &filename() const { return filename_; }

 private:
  void CreateBuffer() {
    if (!buffer_.get()) {
      buffer_.reset(new TransientBytes());
    }
  }
  const std::string filename_;
  std::unique_ptr<TransientBytes> buffer_;
  std::unique_ptr<Inflater> inflater_;
};

// Combines the contents of the multiple input entries which are XML
// files into a single XML output entry with given top level XML tag.
class XmlCombiner {
 public:
  XmlCombiner(const std::string &filename, const char *xml_tag)
      : filename_(filename), xml_tag_(xml_tag) {}

  bool Merge(const CDH *cdh, const LH *lh) {
    if (!concatenator_.get()) {
      concatenator_.reset(new Concatenator(filename_));
      concatenator_->Append("<");
      concatenator_->Append(xml_tag_);
      concatenator_->Append(">\n");
    }
    return concatenator_->Merge(cdh, lh);
  }

  // Returns a pointer to the buffer containing LocalHeader for the entry,
  // immediately followed by entry payload. The caller is responsible for
  // freeing the buffer.
  void *OutputEntry() {
    if (!concatenator_.get()) {
      return nullptr;
    }
    concatenator_->Append("</");
    concatenator_->Append(xml_tag_);
    concatenator_->Append(">\n");
    return concatenator_->OutputEntry();
  }

  const std::string filename() const { return filename_; }

 private:
  const std::string filename_;
  const char *xml_tag_;
  std::unique_ptr<Concatenator> concatenator_;
  std::unique_ptr<Inflater> inflater_;
};

// A wrapper around Concatenator allowing to append
//   NAME=VALUE
// lines to the contents.
class PropertyCombiner : public Concatenator {
 public:
  PropertyCombiner(const std::string &filename) : Concatenator(filename) {}
  void AddProperty(const char *key, const char *value) {
    // TODO(asmundak): deduplicate properties.
    Append(key);
    Append("=", 1);
    Append(value);
    Append("\n", 1);
  }

  void AddProperty(const std::string &key, const std::string &value) {
    // TODO(asmundak): deduplicate properties.
    Append(key);
    Append("=", 1);
    Append(value);
    Append("\n", 1);
  }
};

#endif  //  SRC_TOOLS_SINGLEJAR_COMBINERS_H_
