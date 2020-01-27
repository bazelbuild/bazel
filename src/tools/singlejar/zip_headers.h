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

#ifndef BAZEL_SRC_TOOLS_SINGLEJAR_ZIP_HEADERS_H_
#define BAZEL_SRC_TOOLS_SINGLEJAR_ZIP_HEADERS_H_

/*
 * Zip file headers, as described in .ZIP File Format Specification
 * http://www.pkware.com/documents/casestudies/APPNOTE.TXT
 */

#include <stdlib.h>
#include <string.h>

#include <cinttypes>

#if defined(__linux__)
#include <endian.h>
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/endian.h>
#elif defined(__APPLE__) || defined(_WIN32)
// Hopefully OSX and Windows will keep running solely on little endian CPUs, so:
#define le16toh(x) (x)
#define le32toh(x) (x)
#define le64toh(x) (x)
#define htole16(x) (x)
#define htole32(x) (x)
#define htole64(x) (x)
#else
#error "This platform is not supported."
#endif

#include <string>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma pack(push, 1)
#define attr_packed
#else
#define attr_packed  __attribute__((packed))
#endif

class ziph {
 public:
  static const uint8_t *byte_ptr(const void *ptr) {
    return reinterpret_cast<const uint8_t *>(ptr);
  }

  /* Utility functions to handle Zip64 extensions. Size and position fields in
   * the Zip headers are 32-bit wide. If field's value does not fit into 32
   * bits (more precisely, it is >= 0xFFFFFFFF), the field contains 0xFFFFFFFF
   * and the actual value is saved in the corresponding 64-bit extension field.
   * The first function returns true if there is an extension for the given
   * field value, and the second returns true if given field value needs
   * extension.
   */
  static bool zfield_has_ext64(uint32_t v) { return v == 0xFFFFFFFF; }
  static bool zfield_needs_ext64(uint64_t v) { return v >= 0xFFFFFFFF; }
};

/* Overall .ZIP file format (section 4.3.6), and the corresponding classes
 *    [local file header 1]                          class LH
 *    [encryption header 1]
 *    [file data 1]
 *    [data descriptor 1]
 *    .
 *    .
 *    .
 *    [local file header n]
 *    [encryption header n]
 *    [file data n]
 *    [data descriptor n]
 *    [archive decryption header]
 *    [archive extra data record]
 *    [central directory header 1]                   class CDH
 *    .
 *    .
 *    .
 *    [central directory header n]
 *    [zip64 end of central directory record]        class ECD64
 *    [zip64 end of central directory locator]       class ECDLocator
 *    [end of central directory record]              class ECD
 */

class ExtraField {
 public:
  static const ExtraField *find(uint16_t tag, const uint8_t *start,
                                const uint8_t *end) {
    while (start < end) {
      auto extra_field = reinterpret_cast<const ExtraField *>(start);
      if (extra_field->is(tag)) {
        return extra_field;
      }
      start = ziph::byte_ptr(start) + extra_field->size();
    }
    return nullptr;
  }
  bool is(uint16_t tag) const { return htole16(tag_) == tag; }
  bool is_zip64() const { return is(1); }
  bool is_unix_time() const { return is(0x5455); }
  void signature(uint16_t tag) { tag_ = le16toh(tag); }

  uint16_t payload_size() const { return le16toh(payload_size_); }
  void payload_size(uint16_t v) { payload_size_ = htole16(v); }

  uint16_t size() const { return sizeof(ExtraField) + payload_size(); }

  const ExtraField *next() const {
    return reinterpret_cast<const ExtraField *>(ziph::byte_ptr(this) + size());
  }

 protected:
  uint16_t tag_;
  uint16_t payload_size_;
} attr_packed;

static_assert(4 == sizeof(ExtraField),
              "ExtraField class fields layout is incorrect.");

/* Zip64 Extra Field (section 4.5.3 of the .ZIP format spec)
 *
 * It is present if a value of a uncompressed_size/compressed_size/file_offset
 * exceeds 32 bits. It consists of a 4-byte header followed by
 * [64-bit uncompressed_size] [64-bit compressed_size] [64-bit file_offset]
 * Only the entities whose value exceed 32 bits are present, and the present
 * ones are always in the order shown above. The originating 32-bit field
 * contains 0xFFFFFFFF to indicate that the value is 64-bit and is in
 * Zip64 Extra Field. Section 4.5.3 of the spec mentions that Zip64 extra field
 * of the Local Header MUST have both uncompressed and compressed sizes present.
 */
class Zip64ExtraField : public ExtraField {
 public:
  static const Zip64ExtraField *find(const uint8_t *start, const uint8_t *end) {
    return reinterpret_cast<const Zip64ExtraField *>(
        ExtraField::find(1, start, end));
  }

  bool is() const { return is_zip64(); }
  void signature() { ExtraField::signature(1); }

  // The value of i-th attribute
  uint64_t attr64(int index) const { return le64toh(attr_[index]); }
  void attr64(int index, uint64_t v) { attr_[index] = htole64(v); }

  // Attribute count
  int attr_count() const { return payload_size() / sizeof(attr_[0]); }
  void attr_count(int n) { payload_size(n * sizeof(attr_[0])); }

  // Space needed for this field to accommodate n_attr attributes
  static uint16_t space_needed(int n_attrs) {
    return n_attrs > 0 ? sizeof(Zip64ExtraField) + n_attrs * sizeof(uint64_t)
                       : 0;
  }

 private:
  uint64_t attr_[];
} attr_packed;
static_assert(4 == sizeof(Zip64ExtraField),
              "Zip64ExtraField class fields layout is incorrect.");

/* Extended Timestamp Extra Field.
 * This field in the Central Directory Header contains only the modification
 * time, whereas in the Local Header up to three timestamps (modification.
 * access, creation) may be present.
 * The time values are in standard Unix signed-long format, indicating the
 * number of seconds since 1 January 1970 00:00:00.  The times are relative to
 * Coordinated Universal Time (UTC).
 */
class UnixTimeExtraField : public ExtraField {
 public:
  static const UnixTimeExtraField *find(const uint8_t *start,
                                        const uint8_t *end) {
    return reinterpret_cast<const UnixTimeExtraField *>(
        ExtraField::find(0x5455, start, end));
  }
  bool is() const { return is_unix_time(); }
  void signature() { ExtraField::signature(0x5455); }

  void flags(uint8_t v) { flags_ = v; }
  bool has_modification_time() const { return flags_ & 1; }
  bool has_access_time() const { return flags_ & 2; }
  bool has_creation_time() const { return flags_ & 4; }

  uint32_t timestamp(int index) const { return le32toh(timestamp_[index]); }
  void timestamp(int index, uint32_t v) { timestamp_[index] = htole32(v); }

  int timestamp_count() const {
    return (payload_size() - sizeof(flags_)) / sizeof(timestamp_[0]);
  }

 private:
  uint8_t flags_;
  uint32_t timestamp_[];
} attr_packed;
static_assert(5 == sizeof(UnixTimeExtraField),
              "UnixTimeExtraField layout is incorrect");

/* Local Header precedes each archive file data (section 4.3.7).  */
class LH {
 public:
  bool is() const { return 0x04034b50 == le32toh(signature_); }
  void signature() { signature_ = htole32(0x04034b50); }

  uint16_t version() const { return le16toh(version_); }
  void version(uint16_t v) { version_ = htole16(v); }

  void bit_flag(uint16_t v) { bit_flag_ = htole16(v); }
  uint16_t bit_flag() const { return le16toh(bit_flag_); }

  uint16_t compression_method() const { return le16toh(compression_method_); }
  void compression_method(uint16_t v) { compression_method_ = htole16(v); }

  uint16_t last_mod_file_time() const { return le16toh(last_mod_file_time_); }
  void last_mod_file_time(uint16_t v) { last_mod_file_time_ = htole16(v); }

  uint16_t last_mod_file_date() const { return le16toh(last_mod_file_date_); }
  void last_mod_file_date(uint16_t v) { last_mod_file_date_ = htole16(v); }

  uint32_t crc32() const { return le32toh(crc32_); }
  void crc32(uint32_t v) { crc32_ = htole32(v); }

  size_t compressed_file_size() const {
    size_t size32 = compressed_file_size32();
    if (ziph::zfield_has_ext64(size32)) {
      const Zip64ExtraField *z64 = zip64_extra_field();
      return z64 == nullptr ? 0xFFFFFFFF : z64->attr64(1);
    }
    return size32;
  }
  size_t compressed_file_size32() const {
    return le32toh(compressed_file_size32_);
  }
  void compressed_file_size32(uint32_t v) {
    compressed_file_size32_ = htole32(v);
  }

  size_t uncompressed_file_size() const {
    size_t size32 = uncompressed_file_size32();
    if (ziph::zfield_has_ext64(size32)) {
      const Zip64ExtraField *z64 = zip64_extra_field();
      return z64 == nullptr ? 0xFFFFFFFF : z64->attr64(0);
    }
    return size32;
  }
  size_t uncompressed_file_size32() const {
    return le32toh(uncompressed_file_size32_);
  }
  void uncompressed_file_size32(uint32_t v) {
    uncompressed_file_size32_ = htole32(v);
  }

  uint16_t file_name_length() const { return le16toh(file_name_length_); }
  const char *file_name() const { return file_name_; }
  void file_name(const char *filename, uint16_t len) {
    file_name_length_ = htole16(len);
    if (len) {
      memcpy(file_name_, filename, file_name_length_);
    }
  }
  bool file_name_is(const char *name) const {
    size_t name_len = strlen(name);
    return file_name_length() == name_len &&
           0 == strncmp(file_name(), name, name_len);
  }
  std::string file_name_string() const {
    return std::string(file_name(), file_name_length());
  }

  uint16_t extra_fields_length() const { return le16toh(extra_fields_length_); }
  const uint8_t *extra_fields() const {
    return ziph::byte_ptr(file_name_ + file_name_length());
  }
  uint8_t *extra_fields() {
    return reinterpret_cast<uint8_t *>(file_name_) + file_name_length();
  }
  void extra_fields(const uint8_t *data, uint16_t data_length) {
    extra_fields_length_ = htole16(data_length);
    if (data_length) {
      memcpy(extra_fields(), data, data_length);
    }
  }

  size_t size() const {
    return sizeof(LH) + file_name_length() + extra_fields_length();
  }
  const uint8_t *data() const { return extra_fields() + extra_fields_length(); }
  uint8_t *data() { return extra_fields() + extra_fields_length(); }

  size_t in_zip_size() const {
    return compression_method() ? compressed_file_size()
                                : uncompressed_file_size();
  }

  const Zip64ExtraField *zip64_extra_field() const {
    return Zip64ExtraField::find(extra_fields(),
                                 extra_fields() + extra_fields_length());
  }

  const UnixTimeExtraField *unix_time_extra_field() const {
    return UnixTimeExtraField::find(extra_fields(),
                                    extra_fields() + extra_fields_length());
  }

 private:
  uint32_t signature_;
  uint16_t version_;
  uint16_t bit_flag_;
  uint16_t compression_method_;
  uint16_t last_mod_file_time_;
  uint16_t last_mod_file_date_;
  uint32_t crc32_;
  uint32_t compressed_file_size32_;
  uint32_t uncompressed_file_size32_;
  uint16_t file_name_length_;
  uint16_t extra_fields_length_;
  char file_name_[0];
  // Followed by extra_fields.
} attr_packed;
static_assert(30 == sizeof(LH), "The fields layout for class LH is incorrect");

/* Data descriptor Record:
 *    4.3.9  Data descriptor:
 *
 *      crc-32                          4 bytes
 *      compressed size                 4 bytes
 *       uncompressed size               4 bytes
 *
 *    4.3.9.1 This descriptor MUST exist if bit 3 of the general purpose bit
 *    flag is set (see below).  It is byte aligned and immediately follows the
 *    last byte of compressed data. This descriptor SHOULD be used only when it
 *    was not possible to seek in the output .ZIP file, e.g., when the output
 *    .ZIP file was standard output or a non-seekable device.  For ZIP64(tm)
 *    format archives, the compressed and uncompressed sizes are 8 bytes each.
 *
 *    4.3.9.2 When compressing files, compressed and uncompressed sizes should
 *    be stored in ZIP64 format (as 8 byte values) when a file's size exceeds
 *    0xFFFFFFFF.   However ZIP64 format may be used regardless of the size of a
 *    file.  When extracting, if the zip64 extended information extra field is
 *    present for the file the compressed and uncompressed sizes will be 8 byte
 *    values.
 *
 *    4.3.9.3 Although not originally assigned a signature, the value 0x08074b50
 *    has commonly been adopted as a signature value for the data descriptor
 *    record.  Implementers should be aware that ZIP files may be encountered
 *    with or without this signature marking data descriptors and SHOULD account
 *    for either case when reading ZIP files to ensure compatibility.
 */
class DDR {
 public:
  size_t size(bool compressed_size_is_64bits,
              bool original_size_is_64bits) const {
    return (0x08074b50 == le32toh(optional_signature_) ? 8 : 4) +
           (compressed_size_is_64bits ? 8 : 4) +
           (original_size_is_64bits ? 8 : 4);
  }

 private:
  uint32_t optional_signature_;
} attr_packed;

/* Central Directory Header.  */
class CDH {
 public:
  void signature() { signature_ = htole32(0x02014b50); }
  bool is() const { return 0x02014b50 == le32toh(signature_); }

  uint16_t version() const { return le16toh(version_); }
  void version(uint16_t v) { version_ = htole16(v); }

  uint16_t version_to_extract() const { return le16toh(version_to_extract_); }
  void version_to_extract(uint16_t v) { version_to_extract_ = htole16(v); }

  void bit_flag(uint16_t v) { bit_flag_ = htole16(v); }
  uint16_t bit_flag() const { return le16toh(bit_flag_); }

  uint16_t compression_method() const { return le16toh(compression_method_); }
  void compression_method(uint16_t v) { compression_method_ = htole16(v); }

  uint16_t last_mod_file_time() const { return le16toh(last_mod_file_time_); }
  void last_mod_file_time(uint16_t v) { last_mod_file_time_ = htole16(v); }

  uint16_t last_mod_file_date() const { return le16toh(last_mod_file_date_); }
  void last_mod_file_date(uint16_t v) { last_mod_file_date_ = htole16(v); }

  void crc32(uint32_t v) { crc32_ = htole32(v); }
  uint32_t crc32() const { return le32toh(crc32_); }

  size_t compressed_file_size() const {
    size_t size32 = compressed_file_size32();
    if (ziph::zfield_has_ext64(size32)) {
      const Zip64ExtraField *z64 = zip64_extra_field();
      return z64 == nullptr ? 0xFFFFFFFF
                            : z64->attr64(ziph::zfield_has_ext64(
                                  uncompressed_file_size32()));
    }
    return size32;
  }
  size_t compressed_file_size32() const {
    return le32toh(compressed_file_size32_);
  }
  void compressed_file_size32(uint32_t v) {
    compressed_file_size32_ = htole32(v);
  }

  size_t uncompressed_file_size() const {
    uint32_t size32 = uncompressed_file_size32();
    if (ziph::zfield_has_ext64(size32)) {
      const Zip64ExtraField *z64 = zip64_extra_field();
      return z64 == nullptr ? 0xFFFFFFFF : z64->attr64(0);
    }
    return size32;
  }
  size_t uncompressed_file_size32() const {
    return le32toh(uncompressed_file_size32_);
  }

  void uncompressed_file_size32(uint32_t v) {
    uncompressed_file_size32_ = htole32(v);
  }

  uint16_t file_name_length() const { return le16toh(file_name_length_); }
  const char *file_name() const { return file_name_; }
  void file_name(const char *filename, uint16_t filename_len) {
    file_name_length_ = htole16(filename_len);
    if (filename_len) {
      memcpy(file_name_, filename, filename_len);
    }
  }
  bool file_name_is(const char *name) const {
    size_t name_len = strlen(name);
    return file_name_length() == name_len &&
           0 == strncmp(file_name(), name, name_len);
  }
  std::string file_name_string() const {
    return std::string(file_name(), file_name_length());
  }

  uint16_t extra_fields_length() const { return le16toh(extra_fields_length_); }
  const uint8_t *extra_fields() const {
    return ziph::byte_ptr(file_name_ + file_name_length());
  }
  uint8_t *extra_fields() {
    return reinterpret_cast<uint8_t *>(file_name_) + file_name_length();
  }
  void extra_fields(const uint8_t *data, uint16_t data_length) {
    extra_fields_length_ = htole16(data_length);
    if (data_length && data != extra_fields()) {
      memcpy(extra_fields(), data, data_length);
    }
  }

  uint16_t comment_length() const { return le16toh(comment_length_); }
  void comment_length(uint16_t v) { comment_length_ = htole16(v); }

  uint16_t start_disk_nr() const { return le16toh(start_disk_nr_); }
  void start_disk_nr(uint16_t v) { start_disk_nr_ = htole16(v); }

  uint16_t internal_attributes() const { return le16toh(internal_attributes_); }
  void internal_attributes(uint16_t v) { internal_attributes_ = htole16(v); }

  uint32_t external_attributes() const { return le32toh(external_attributes_); }
  void external_attributes(uint32_t v) { external_attributes_ = htole32(v); }

  uint64_t local_header_offset() const {
    uint32_t size32 = local_header_offset32();
    if (ziph::zfield_has_ext64(size32)) {
      const Zip64ExtraField *z64 = zip64_extra_field();
      int attr_no = ziph::zfield_has_ext64(uncompressed_file_size32());
      if (ziph::zfield_has_ext64(compressed_file_size32())) {
        ++attr_no;
      }
      return z64 == nullptr ? 0xFFFFFFFF : z64->attr64(attr_no);
    }
    return size32;
  }

  uint32_t local_header_offset32() const {
    return le32toh(local_header_offset32_);
  }
  void local_header_offset32(uint32_t v) {
    local_header_offset32_ = htole32(v);
  }
  bool no_size_in_local_header() const { return bit_flag() & 0x08; }
  size_t size() const {
    return sizeof(*this) + file_name_length() + extra_fields_length() +
           comment_length();
  }

  const Zip64ExtraField *zip64_extra_field() const {
    return Zip64ExtraField::find(extra_fields(),
                                 extra_fields() + extra_fields_length());
  }

  const UnixTimeExtraField *unix_time_extra_field() const {
    return UnixTimeExtraField::find(extra_fields(),
                                    extra_fields() + extra_fields_length());
  }

 private:
  uint32_t signature_;
  uint16_t version_;
  uint16_t version_to_extract_;
  uint16_t bit_flag_;
  uint16_t compression_method_;
  uint16_t last_mod_file_time_;
  uint16_t last_mod_file_date_;
  uint32_t crc32_;
  uint32_t compressed_file_size32_;
  uint32_t uncompressed_file_size32_;
  uint16_t file_name_length_;
  uint16_t extra_fields_length_;
  uint16_t comment_length_;
  uint16_t start_disk_nr_;
  uint16_t internal_attributes_;
  uint32_t external_attributes_;
  uint32_t local_header_offset32_;
  char file_name_[0];
  // Followed by extra fields and then comment.
} attr_packed;
static_assert(46 == sizeof(CDH), "Class CDH fields layout is incorrect.");

/* Zip64 End of Central Directory Locator.  */
class ECD64Locator {
 public:
  void signature() { signature_ = htole32(0x07064b50); }
  bool is() const { return 0x07064b50 == le32toh(signature_); }

  void ecd64_disk_nr(uint32_t nr) { ecd64_disk_nr_ = htole32(nr); }
  uint32_t ecd64_disk_nr() const { return le32toh(ecd64_disk_nr_); }

  void ecd64_offset(uint64_t v) { ecd64_offset_ = htole64(v); }
  uint64_t ecd64_offset() const { return le64toh(ecd64_offset_); }

  void total_disks(uint32_t v) { total_disks_ = htole32(v); }
  uint32_t total_disks() const { return le32toh(total_disks_); }

 private:
  uint32_t signature_;
  uint32_t ecd64_disk_nr_;
  uint64_t ecd64_offset_;
  uint32_t total_disks_;
} attr_packed;
static_assert(20 == sizeof(ECD64Locator),
              "ECD64Locator class fields layout is incorrect.");

/* End of Central Directory.  */
class ECD {
 public:
  void signature() { signature_ = htole32(0x06054b50); }
  bool is() const { return 0x06054b50 == le32toh(signature_); }

  void this_disk_nr(uint16_t v) { this_disk_nr_ = htole16(v); }
  uint16_t this_disk_nr() const { return le16toh(this_disk_nr_); }

  void cen_disk_nr(uint16_t v) { cen_disk_nr_ = htole16(v); }
  uint16_t cen_disk_nr() const { return le16toh(cen_disk_nr_); }

  void this_disk_entries16(uint16_t v) { this_disk_entries16_ = htole16(v); }
  uint16_t this_disk_entries16() const { return le16toh(this_disk_entries16_); }

  void total_entries16(uint16_t v) { total_entries16_ = htole16(v); }
  uint16_t total_entries16() const { return le16toh(total_entries16_); }

  void cen_size32(uint32_t v) { cen_size32_ = htole32(v); }
  uint32_t cen_size32() const { return le32toh(cen_size32_); }

  void cen_offset32(uint32_t v) { cen_offset32_ = htole32(v); }
  uint32_t cen_offset32() const { return le32toh(cen_offset32_); }

  void comment(uint8_t *data, uint16_t data_size) {
    comment_length_ = htole16(data_size);
    if (data_size) {
      memcpy(comment_, data, data_size);
    }
  }
  uint16_t comment_length() const { return le16toh(comment_length_); }
  const uint8_t *comment() const { return comment_; }

  uint64_t ecd64_offset() const {
    const ECD64Locator *locator = reinterpret_cast<const ECD64Locator *>(
        ziph::byte_ptr(this) - sizeof(ECD64Locator));
    return locator->is() ? locator->ecd64_offset() : 0xFFFFFFFFFFFFFFFF;
  }

 private:
  uint32_t signature_;
  uint16_t this_disk_nr_;
  uint16_t cen_disk_nr_;
  uint16_t this_disk_entries16_;
  uint16_t total_entries16_;
  uint32_t cen_size32_;
  uint32_t cen_offset32_;
  uint16_t comment_length_;
  uint8_t comment_[0];
} attr_packed;
static_assert(22 == sizeof(ECD), "ECD class fields layout is incorrect.");

/* Zip64 end of central directory.  */
class ECD64 {
 public:
  bool is() const { return 0x06064b50 == le32toh(signature_); }
  void signature() { signature_ = htole32(0x06064b50); }

  void remaining_size(uint64_t v) { remaining_size_ = htole64(v); }
  uint64_t remaining_size() const { return le64toh(remaining_size_); }

  void version(uint16_t v) { version_ = htole16(v); }
  uint16_t version() const { return le16toh(version_); }

  void version_to_extract(uint16_t v) { version_to_extract_ = htole16(v); }
  uint16_t version_to_extract() const { return le16toh(version_to_extract_); }

  void this_disk_nr(uint32_t v) { this_disk_nr_ = htole32(v); }
  uint32_t this_disk_nr() const { return le32toh(this_disk_nr_); }

  void cen_disk_nr(uint32_t v) { cen_disk_nr_ = htole32(v); }
  uint32_t cen_disk_nr() const { return le32toh(cen_disk_nr_); }

  void this_disk_entries(uint64_t v) { this_disk_entries_ = htole64(v); }
  uint64_t this_disk_entries() const { return le64toh(this_disk_entries_); }

  void total_entries(uint64_t v) { total_entries_ = htole64(v); }
  uint64_t total_entries() const { return le64toh(total_entries_); }

  void cen_size(uint64_t v) { cen_size_ = htole64(v); }
  uint64_t cen_size() const { return le64toh(cen_size_); }

  void cen_offset(uint64_t v) { cen_offset_ = htole64(v); }
  uint64_t cen_offset() const { return le64toh(cen_offset_); }

 private:
  uint32_t signature_;
  uint64_t remaining_size_;
  uint16_t version_;
  uint16_t version_to_extract_;
  uint32_t this_disk_nr_;
  uint32_t cen_disk_nr_;
  uint64_t this_disk_entries_;
  uint64_t total_entries_;
  uint64_t cen_size_;
  uint64_t cen_offset_;
} attr_packed;
static_assert(56 == sizeof(ECD64), "ECD64 class fields layout is incorrect.");

#if defined(_MSC_VER) && !defined(__clang__)
#pragma pack(pop)
#endif

#undef attr_packed

#endif  // BAZEL_SRC_TOOLS_SINGLEJAR_ZIP_HEADERS_H_
