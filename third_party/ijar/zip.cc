// Copyright 2007 Alan Donovan. All rights reserved.
//
// Author: Alan Donovan <adonovan@google.com>
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
//
// zip.cc -- .zip (.jar) file reading/writing routines.
//

// See README.txt for details.
//
// See http://www.pkware.com/documents/casestudies/APPNOTE.TXT
// for definition of PKZIP file format.

#define _FILE_OFFSET_BITS 64  // Support zip files larger than 2GB

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits>
#include <vector>

#include "common.h"
#include <zlib.h>

#define LOCAL_FILE_HEADER_SIGNATURE           0x04034b50
#define CENTRAL_FILE_HEADER_SIGNATURE         0x02014b50
#define END_OF_CENTRAL_DIR_SIGNATURE          0x06054b50
#define DATA_DESCRIPTOR_SIGNATURE             0x08074b50

// version to extract: 1.0 - default value from APPNOTE.TXT.
// Output JAR files contain no extra ZIP features, so this is enough.
#define ZIP_VERSION_TO_EXTRACT                10
#define COMPRESSION_METHOD_STORED             0   // no compression
#define COMPRESSION_METHOD_DEFLATED           8

#define GENERAL_PURPOSE_BIT_FLAG_COMPRESSED (1 << 3)
#define GENERAL_PURPOSE_BIT_FLAG_UTF8_ENCODED (1 << 11)
#define GENERAL_PURPOSE_BIT_FLAG_SUPPORTED \
  (GENERAL_PURPOSE_BIT_FLAG_COMPRESSED | GENERAL_PURPOSE_BIT_FLAG_UTF8_ENCODED)

#define STRINGIFY(x) #x
#define SYSCALL(expr)  do { \
                         if ((expr) < 0) { \
                           perror(STRINGIFY(expr)); \
                           abort(); \
                         } \
                       } while (0)

namespace devtools_ijar {

bool verbose = false;

// In the absence of ZIP64 support, zip files are limited to 4GB.
// http://www.info-zip.org/FAQ.html#limits
static const u8 kMaximumOutputSize = std::numeric_limits<uint32_t>::max();

static bool ProcessCentralDirEntry(const u1 *&p,
                                   size_t *compressed_size,
                                   size_t *uncompressed_size,
                                   bool *is_class_file);

struct JarStripper {
  JarStripper(const u1 * const zipdata_in,
              u1 * const zipdata_out,
              size_t in_length,
              const u1 * central_dir) :
      zipdata_in_(zipdata_in),
      zipdata_out_(zipdata_out),
      zipdata_in_mapped_(zipdata_in),
      zipdata_out_mapped_(zipdata_out),
      central_dir_(central_dir),
      in_length_(in_length),
      p(zipdata_in),
      q(zipdata_out),
      central_dir_current_(central_dir) {
    uncompressed_data_allocated_ = INITIAL_BUFFER_SIZE;
    uncompressed_data_ =
        reinterpret_cast<u1*>(malloc(uncompressed_data_allocated_));
  }

  ~JarStripper()  {
    free(uncompressed_data_);
  }

  // Scan through the input .jar file, stripping each class file and
  // emitting it to the output .jar file.  Returns the size of the
  // output.
  off_t Run();

 private:
  struct LocalFileEntry {
    // Start of the local header (in the output buffer).
    size_t local_header_offset;
    size_t uncompressed_length;

    // Start/length of the file_name in the local header.
    u1 *file_name;
    u2 file_name_length;

    // Start/length of the extra_field in the local header.
    const u1 *extra_field;
    u2 extra_field_length;
  };

  // Buffer size is initially INITIAL_BUFFER_SIZE. It doubles in size every
  // time it is found too small, until it reaches MAX_BUFFER_SIZE. If that is
  // not enough, we bail out. We only decompress class files, so they should
  // be smaller than 64K anyway, but we give a little leeway.
  static const size_t INITIAL_BUFFER_SIZE = 256 * 1024;  // 256K
  static const size_t MAX_BUFFER_SIZE = 16 * 1024 * 1024;
  static const size_t MAX_MAPPED_REGION = 32 * 1024 * 1024;

  const u1 * const zipdata_in_;   // start of input file mmap
  u1 * const zipdata_out_;        // start of output file mmap
  const u1 * zipdata_in_mapped_;  // start of still mapped region
  u1 * zipdata_out_mapped_;       // start of still mapped region
  const u1 * const central_dir_;  // central directory in input file

  size_t in_length_;  // size of the input file

  const u1 *p;  // input cursor
  u1 *q;  // output cursor
  const u1* central_dir_current_;  // central dir input cursor

  std::vector<LocalFileEntry*> entries_;

  // These metadata fields are the fields of the ZIP header of the file being
  // processed.
  u2 extract_version_;
  u2 general_purpose_bit_flag_;
  u2 compression_method_;
  u4 uncompressed_size_;
  u4 compressed_size_;
  u2 file_name_length_;
  u2 extra_field_length_;
  const u1 *file_name_;
  const u1 *extra_field_;

  // Administration of memory reserved for decompressed data. We use the same
  // buffer for each file to avoid some malloc()/free() calls and free the
  // memory only in the dtor. C-style memory management is used so that we
  // can call realloc.
  u1 *uncompressed_data_;
  size_t uncompressed_data_allocated_;

  // Read one entry from input zip file, and emit corresponding entry
  // in output zip file.
  void ProcessLocalFileEntry(size_t compressed_size, size_t uncompressed_size,
                             bool is_class_file);

  // Add a zero-byte file called "dummy" to the output zip file.
  void AddDummyEntry();

  // Write the ZIP central directory structure for each local file
  // entry in "entries".
  void WriteCentralDirectory();

  // Check that at least n bytes remain in the input file, otherwise
  // abort with an error message.  "state" is the name of the field
  // we're about to read, for diagnostics.
  void EnsureRemaining(size_t n, const char *state) {
    size_t in_offset = p - zipdata_in_;
    size_t remaining = in_length_ - in_offset;
    if (n > remaining) {
      fprintf(stderr, "Premature end of file (at offset %zd, state=%s); "
              "expected %zd more bytes but found %zd.\n",
              in_offset, state, n, remaining);
      abort();
    }
  }

  // Returns the offset of the pointer relative to the start of the
  // output zip file.
  size_t Offset(const u1 *const x) {
    return x - zipdata_out_;
  }

  // Uncompress a file from the archive using zlib. The pointer returned
  // is owned by JarStripper, so it must not be freed. Advances the input cursor
  // to the first byte after the compressed data.
  u1* UncompressFile();

  // Write ZIP file header in the output. Since the compressed size is not
  // known in advance, it must be recorded later. This method returns a pointer
  // to "compressed size" in the file header that should be passed to
  // WriteFileSizeInLocalFileHeader() later.
  u1* WriteLocalFileHeader();

  // Fill in the "compressed size" and "uncompressed size" fields in a local
  // file header previously written by WriteLocalFileHeader().
  void WriteFileSizeInLocalFileHeader(u1 *compressed_size_ptr,
                                      size_t out_length);

  // Process raw class data. Expects that metadata fields are filled out, i.e.
  // extract_version_, general_purpose_bit_flag and their kin.
  void ProcessRawClassData(const u1 *classdata_in);

  // Process a compressed file as a class
  void ProcessCompressedFile();

  // Skip a compressed file
  void SkipCompressedFile();

  // Process an uncompressed file
  void ProcessUncompressedFile();

  // Skip an uncompressed file
  void SkipUncompressedFile();
};

off_t JarStripper::Run() {
  // Process all the entries in the central directory. Also make sure that the
  // content pointer is in sync.
  for (int i = 0; true; i++) {
    size_t compressed, uncompressed;
    bool is_class_file;
    if (!ProcessCentralDirEntry(central_dir_current_,
                                &compressed, &uncompressed, &is_class_file)) {
        break;
    }

    EnsureRemaining(4, "signature");
    u4 signature = get_u4le(p);
    if (signature == LOCAL_FILE_HEADER_SIGNATURE) {
      ProcessLocalFileEntry(compressed, uncompressed, is_class_file);
    } else {
      fprintf(stderr,
              "local file header signature for file %d not found\n", i);
      abort();
    }
  }

  // Add dummy file, since javac doesn't like truly empty jars.
  if (entries_.empty()) AddDummyEntry();

  WriteCentralDirectory();
  return Offset(q);  // = output length
}

void JarStripper::AddDummyEntry() {
  const u1* file_name = (const u1*) "dummy";
  size_t file_name_length = strlen("dummy");

  LocalFileEntry *entry = new LocalFileEntry;
  entry->local_header_offset = Offset(q);

  // Output the ZIP local_file_header:
  put_u4le(q, LOCAL_FILE_HEADER_SIGNATURE);
  put_u2le(q, 10);  // extract_version
  put_u2le(q, 0);  // general_purpose_bit_flag
  put_u2le(q, 0);  // compression_method
  put_u2le(q, 0);  // last_mod_file_time
  put_u2le(q, 0);  // last_mod_file_date
  put_u4le(q, 0);  // crc32
  put_u4le(q, 0);  // compressed_size
  put_u4le(q, 0);  // uncompressed_size
  put_u2le(q, file_name_length);
  put_u2le(q, 0);  // extra_field_length
  put_n(q, file_name, file_name_length);

  entry->file_name_length = file_name_length;
  entry->extra_field_length = 0;
  entry->extra_field = (const u1*) "";
  entry->file_name = (u1*) strdup((const char *) file_name);
  entries_.push_back(entry);
}

void JarStripper::ProcessLocalFileEntry(
    size_t compressed_size, size_t uncompressed_size, bool is_class_file) {
  EnsureRemaining(26, "extract_version");
  extract_version_ = get_u2le(p);
  general_purpose_bit_flag_ = get_u2le(p);

  if ((general_purpose_bit_flag_ & ~GENERAL_PURPOSE_BIT_FLAG_SUPPORTED) != 0) {
    fprintf(stderr, "Unsupported value (0x%04x) in general purpose bit flag.\n",
            general_purpose_bit_flag_);
    abort();
  }

  compression_method_ = get_u2le(p);

  if (compression_method_ != COMPRESSION_METHOD_DEFLATED &&
      compression_method_ != COMPRESSION_METHOD_STORED) {
    fprintf(stderr, "Unsupported compression method (%d).\n",
            compression_method_);
    abort();
  }

  // skip over: last_mod_file_time, last_mod_file_date, crc32
  p += 2 + 2 + 4;
  compressed_size_ = get_u4le(p);
  uncompressed_size_ = get_u4le(p);
  file_name_length_ = get_u2le(p);
  extra_field_length_ = get_u2le(p);

  EnsureRemaining(file_name_length_, "file_name");
  file_name_ = p;
  p += file_name_length_;

  EnsureRemaining(extra_field_length_, "extra_field");
  extra_field_ = p;
  p += extra_field_length_;

  bool is_compressed = compression_method_ == COMPRESSION_METHOD_DEFLATED;

  // If the zip is compressed, compressed and uncompressed size members are
  // zero in the local file header. If not, check that they are the same as the
  // lengths from the central directory, otherwise, just believe the central
  // directory
  if (compressed_size_ == 0) {
    compressed_size_ = compressed_size;
  } else {
    if (compressed_size_ != compressed_size) {
      fprintf(stderr, "central directory and file header inconsistent\n");
      abort();
    }
  }

  if (uncompressed_size_ == 0) {
    uncompressed_size_ = uncompressed_size;
  } else {
    if (uncompressed_size_ != uncompressed_size) {
      fprintf(stderr, "central directory and file header inconsistent\n");
      abort();
    }
  }

  if (is_class_file) {
    if (is_compressed) {
      ProcessCompressedFile();
    } else {
      ProcessUncompressedFile();
    }
  } else {
    if (is_compressed) {
      SkipCompressedFile();
    } else {
      SkipUncompressedFile();
    }
  }

  if (general_purpose_bit_flag_ & GENERAL_PURPOSE_BIT_FLAG_COMPRESSED) {
    // Skip the data descriptor. Some implementations do not put the signature
    // here, so check if the next 4 bytes are a signature, and if so, skip the
    // next 12 bytes (for CRC, compressed/uncompressed size), otherwise skip
    // the next 8 bytes (because the value just read was the CRC).
    u4 signature = get_u4le(p);
    if (signature == DATA_DESCRIPTOR_SIGNATURE) {
      p += 4 * 3;
    } else {
      p += 4 * 2;
    }
  }

  if (q - zipdata_out_mapped_ > MAX_MAPPED_REGION) {
    munmap(zipdata_out_mapped_, MAX_MAPPED_REGION);
    zipdata_out_mapped_ += MAX_MAPPED_REGION;
  }

  if (p - zipdata_in_mapped_ > MAX_MAPPED_REGION) {
    munmap(const_cast<u1*>(zipdata_in_mapped_), MAX_MAPPED_REGION);
    zipdata_in_mapped_ += MAX_MAPPED_REGION;
  }
}

void JarStripper::SkipUncompressedFile() {
  // In this case, compressed_size_ == uncompressed_size_ (since the file is
  // uncompressed), so we can use either.
  if (compressed_size_ != uncompressed_size_) {
    fprintf(stderr, "compressed size != uncompressed size, although the file "
            "is uncompressed.\n");
    abort();
  }

  EnsureRemaining(compressed_size_, "file_data");
  p += compressed_size_;
}

u1* JarStripper::UncompressFile() {
  size_t in_offset = p - zipdata_in_;
  size_t remaining = in_length_ - in_offset;
  z_stream stream;

  stream.zalloc = Z_NULL;
  stream.zfree = Z_NULL;
  stream.opaque = Z_NULL;
  stream.avail_in = remaining;
  stream.next_in = (Bytef *) p;

  int ret = inflateInit2(&stream, -MAX_WBITS);
  if (ret != Z_OK) {
    fprintf(stderr, "inflateInit: %d\n", ret);
    abort();
  }

  int uncompressed_until_now = 0;

  while (true) {
    stream.avail_out = uncompressed_data_allocated_ - uncompressed_until_now;
    stream.next_out = uncompressed_data_ + uncompressed_until_now;
    int old_avail_out = stream.avail_out;

    ret = inflate(&stream, Z_SYNC_FLUSH);
    int uncompressed_now = old_avail_out - stream.avail_out;
    uncompressed_until_now += uncompressed_now;

    switch (ret) {
      case Z_STREAM_END: {
        // zlib said that there is no more data to decompress.

        u1 *new_p = reinterpret_cast<u1*>(stream.next_in);
        compressed_size_ = new_p - p;
        uncompressed_size_ = uncompressed_until_now;
        p = new_p;
        inflateEnd(&stream);
        return uncompressed_data_;
      }

      case Z_OK: {
        // zlib said that there is no more room in the buffer allocated for
        // the decompressed data. Enlarge that buffer and try again.

        if (uncompressed_data_allocated_ == MAX_BUFFER_SIZE) {
          fprintf(stderr,
                  "ijar does not support decompressing files "
                  "larger than %dMB.\n",
                  (int) (MAX_BUFFER_SIZE/(1024*1024)));
          abort();
        }

        uncompressed_data_allocated_ *= 2;
        if (uncompressed_data_allocated_ > MAX_BUFFER_SIZE) {
          uncompressed_data_allocated_ = MAX_BUFFER_SIZE;
        }

        uncompressed_data_ = reinterpret_cast<u1*>(
            realloc(uncompressed_data_, uncompressed_data_allocated_));
        break;
      }

      case Z_DATA_ERROR:
      case Z_BUF_ERROR:
      case Z_STREAM_ERROR:
      case Z_NEED_DICT:
      default: {
        fprintf(stderr, "zlib returned error code %d during inflate.\n", ret);
        abort();
      }
    }
  }
}

void JarStripper::SkipCompressedFile() {
  EnsureRemaining(compressed_size_, "file_data");
  p += compressed_size_;
}

u1* JarStripper::WriteLocalFileHeader() {
    LocalFileEntry *entry = new LocalFileEntry;
    entry->local_header_offset = Offset(q);
    entry->file_name_length = file_name_length_;
    entry->file_name = new u1[file_name_length_];
    memcpy(entry->file_name, file_name_, file_name_length_);
    entry->extra_field_length = 0;
    entry->extra_field = (const u1*)"";

    // Output the ZIP local_file_header:
    put_u4le(q, LOCAL_FILE_HEADER_SIGNATURE);
    put_u2le(q, ZIP_VERSION_TO_EXTRACT);  // version to extract
    put_u2le(q, 0);  // general purpose bit flag
    put_u2le(q, COMPRESSION_METHOD_STORED);  // compression method:
    put_u2le(q, 0);  // last_mod_file_time
    put_u2le(q, 0);  // last_mod_file_date
    put_u4le(q, 0);  // crc32 (jar/javac tools don't care)
    u1 *compressed_size_ptr = q;
    put_u4le(q, 0);  // compressed_size = placeholder
    put_u4le(q, 0);  // uncompressed_size = placeholder
    put_u2le(q, entry->file_name_length);
    put_u2le(q, entry->extra_field_length);

    put_n(q, entry->file_name, entry->file_name_length);
    put_n(q, entry->extra_field, entry->extra_field_length);
    entries_.push_back(entry);

    return compressed_size_ptr;
}

void JarStripper::WriteFileSizeInLocalFileHeader(u1 *compressed_size_ptr,
                                                 size_t out_length) {
  // uncompressed size and compressed size are the same, since the output
  // ijar is uncompressed.
  put_u4le(compressed_size_ptr, out_length);  // compressed_size
  put_u4le(compressed_size_ptr, out_length);  // uncompressed_size
}

void JarStripper::ProcessRawClassData(const u1 *classdata_in) {
  if (verbose) {
    // file_name_ is not NUL-terminated.
    fprintf(stderr, "INFO: StripClass: %.*s\n", file_name_length_, file_name_);
  }
  u1 *compressed_size_ptr = WriteLocalFileHeader();

  u1 *classdata_out = q;
  StripClass(q, classdata_in, uncompressed_size_);  // actually process it
  size_t out_length = q - classdata_out;

  WriteFileSizeInLocalFileHeader(compressed_size_ptr, out_length);
  entries_.back()->uncompressed_length = out_length;
}

void JarStripper::ProcessCompressedFile() {
  u1 *classdata_in = UncompressFile();
  ProcessRawClassData(classdata_in);
}

void JarStripper::ProcessUncompressedFile() {
  // In this case, compressed_size_ == uncompressed_size_ (since the file is
  // uncompressed), so we can use either.
  if (compressed_size_ != uncompressed_size_) {
    fprintf(stderr, "compressed size != uncompressed size, although the file "
            "is uncompressed.\n");
    abort();
  }

  EnsureRemaining(compressed_size_, "file_data");
  const u1 *file_data = p;
  p += compressed_size_;
  ProcessRawClassData(file_data);
}

void JarStripper::WriteCentralDirectory() {
  // central directory:
  const u1 *central_directory_start = q;
  for (int ii = 0; ii < entries_.size(); ++ii) {
    LocalFileEntry *entry = entries_[ii];
    put_u4le(q, CENTRAL_FILE_HEADER_SIGNATURE);
    put_u2le(q, 0);  // version made by

    put_u2le(q, ZIP_VERSION_TO_EXTRACT);  // version to extract
    put_u2le(q, 0);  // general purpose bit flag
    put_u2le(q, COMPRESSION_METHOD_STORED);  // compression method:
    put_u2le(q, 0);  // last_mod_file_time
    put_u2le(q, 0);  // last_mod_file_date
    put_u4le(q, 0);  // crc32 (jar/javac tools don't care)
    put_u4le(q, entry->uncompressed_length);  // compressed_size
    put_u4le(q, entry->uncompressed_length);  // uncompressed_size
    put_u2le(q, entry->file_name_length);
    put_u2le(q, entry->extra_field_length);

    put_u2le(q, 0);  // file comment length
    put_u2le(q, 0);  // disk number start
    put_u2le(q, 0);  // internal file attributes
    put_u4le(q, 0);  // external file attributes
    // relative offset of local header:
    put_u4le(q, entry->local_header_offset);

    put_n(q, entry->file_name, entry->file_name_length);
    put_n(q, entry->extra_field, entry->extra_field_length);
  }
  u4 central_directory_size = q - central_directory_start;

  put_u4le(q, END_OF_CENTRAL_DIR_SIGNATURE);
  put_u2le(q, 0);  // number of this disk
  put_u2le(q, 0);  // number of the disk with the start of the central directory
  put_u2le(q, entries_.size());  // # central dir entries on this disk
  put_u2le(q, entries_.size());  // total # entries in the central directory
  put_u4le(q, central_directory_size);  // size of the central directory
  put_u4le(q, Offset(central_directory_start));  // offset of start of central
                                                 // directory wrt starting disk
  put_u2le(q, 0);  // .ZIP file comment length
}

// Reads and returns some metadata of the next file from the central directory:
// - compressed size
// - uncompressed size
// - whether the entry is a class file (to be included in the output).
// Precondition: p points to the beginning of an entry in the central dir
// Postcondition: p points to the beginning of the next entry in the central dir
// Returns true if the central directory contains another file and false if not.
// Of course, in the latter case, the size output variables are not changed.
// Note that the central directory is always followed by another data structure
// that has a signature, so parsing it this way is safe.
static bool ProcessCentralDirEntry(
    const u1 *&p, size_t *compressed_size, size_t *uncompressed_size,
    bool *is_class_file) {
  u4 signature = get_u4le(p);
  if (signature != CENTRAL_FILE_HEADER_SIGNATURE) {
    return false;
  }

  p += 16;  // skip to 'compressed size' field
  *compressed_size = get_u4le(p);
  *uncompressed_size = get_u4le(p);
  u2 file_name_length = get_u2le(p);
  u2 extra_field_length = get_u2le(p);
  u2 file_comment_length = get_u2le(p);
  p += 12;  // skip to file name field
  {
    static const int len = strlen(".class");
    *is_class_file = file_name_length >= len &&
        memcmp(".class", p + file_name_length - len, len) == 0;
  }
  p += file_name_length;
  p += extra_field_length;
  p += file_comment_length;
  return true;
}

// Given the data in the zip file, returns the offset of the central directory
// and the number of files contained in it.
bool FindZipCentralDirectory(const u1* bytes, size_t in_length, u4* offset) {
  static const int MAX_COMMENT_LENGTH = 0xffff;
  static const int CENTRAL_DIR_LOCATOR_SIZE = 22;
  // Maximum distance of start of central dir locator from end of file
  static const int MAX_DELTA = MAX_COMMENT_LENGTH + CENTRAL_DIR_LOCATOR_SIZE;
  const u1* last_pos_to_check = in_length < MAX_DELTA
      ? bytes
      : bytes + (in_length - MAX_DELTA);
  const u1* current;
  bool found = false;

  for (current = bytes + in_length - CENTRAL_DIR_LOCATOR_SIZE;
       current >= last_pos_to_check;
       current-- ) {
    const u1* p = current;
    if (get_u4le(p) != END_OF_CENTRAL_DIR_SIGNATURE) {
      continue;
    }

    p += 16;  // skip to comment length field
    u2 comment_length = get_u2le(p);

    // Does the comment go exactly till the end of the file?
    if (current + comment_length + CENTRAL_DIR_LOCATOR_SIZE
        != bytes + in_length) {
      continue;
    }

    // Hooray, we found it!
    found = true;
    break;
  }

  if (!found) {
    fprintf(stderr, "file is invalid or corrupted (missing end of central "
                    "directory record)\n");
    return false;
  }

  get_u4le(current);  // central directory locator signature, already checked
  u2 number_of_this_disk = get_u2le(current);
  u2 disk_with_central_dir = get_u2le(current);
  u2 central_dir_entries_on_this_disk = get_u2le(current);
  u2 central_dir_entries = get_u2le(current);
  get_u4le(current);  // central directory size
  u4 central_dir_offset = get_u4le(current);

  if (number_of_this_disk != 0
    || disk_with_central_dir != 0
    || central_dir_entries_on_this_disk != central_dir_entries) {
    fprintf(stderr, "multi-disk JAR files are not supported\n");
    return false;
  }

  // Do not change output values before determining that they are OK.
  *offset = central_dir_offset;
  return true;
}

// Gives a maximum bound on the size of the interface JAR. Basically, adds
// the difference between the compressed and uncompressed sizes to the size
// of the input file.
static u8 CalculateOutputLength(const u1* central_dir, size_t in_length) {
  const u1* current = central_dir;

  u8 compressed_size = 0;
  u8 uncompressed_size = 0;
  u8 skipped_compressed_size = 0;

  while (true) {
    size_t file_compressed, file_uncompressed;
    bool is_class_file;
    if (!ProcessCentralDirEntry(current,
                                &file_compressed, &file_uncompressed,
                                &is_class_file)) {
      break;
    }

    if (is_class_file) {
      compressed_size += (u8) file_compressed;
      uncompressed_size += (u8) file_uncompressed;
    } else {
      skipped_compressed_size += file_compressed;
    }
  }

  // The worst case is when the output is simply the input uncompressed. The
  // metadata in the zip file will stay the same, so the file will grow by the
  // difference between the compressed and uncompressed sizes.
  return (u8) in_length - skipped_compressed_size
      + (uncompressed_size - compressed_size);
}

int OpenFilesAndProcessJar(const char *file_out, const char *file_in) {
  int fd_in = open(file_in, O_RDONLY);
  if (fd_in < 0) {
    fprintf(stderr, "Can't open file %s for reading: %s.\n",
            file_in, strerror(errno));
    return 1;
  }

  off_t length;
  SYSCALL(length = lseek(fd_in, 0, SEEK_END));

  void *zipdata_in = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd_in, 0);
  if (zipdata_in == MAP_FAILED) {
    perror("mmap(in)");
    return 1;
  }

  u4 central_dir_offset;

  if (!devtools_ijar::FindZipCentralDirectory(
          static_cast<const u1*>(zipdata_in), length, &central_dir_offset)) {
    abort();
  }

  const u1* central_dir =
      static_cast<const u1*>(zipdata_in) + central_dir_offset;

  u8 output_length = CalculateOutputLength(central_dir, length);
  if (output_length > kMaximumOutputSize) {
    fprintf(stderr,
            "Uncompressed input jar has size %llu, "
            "which exceeds the maximum supported output size %llu.\n"
            "Assuming that ijar will be smaller and hoping for the best.\n",
            output_length, kMaximumOutputSize);
    output_length = kMaximumOutputSize;
  }

  int fd_out = open(file_out, O_CREAT|O_RDWR|O_TRUNC, 0644);
  if (fd_out < 0) {
    fprintf(stderr, "Can't create file %s: %s.\n",
            file_out, strerror(errno));
    return 1;
  }
  // Create mmap-able sparse file
  SYSCALL(ftruncate(fd_out, output_length));

  // Ensure that any buffer overflow in JarStripper will result in
  // SIGSEGV or SIGBUS by over-allocating beyond the end of the file.
  size_t mmap_length = std::min(output_length + sysconf(_SC_PAGESIZE),
                                (u8) std::numeric_limits<size_t>::max());

  void *zipdata_out = mmap(NULL, mmap_length, PROT_WRITE,
                           MAP_SHARED, fd_out, 0);
  if (zipdata_out == MAP_FAILED) {
    fprintf(stderr, "output_length=%llu\n", output_length);
    perror("mmap(out)");
    return 1;
  }

  JarStripper stripper((const u1*) zipdata_in, (u1*) zipdata_out,
                       length, (const u1*) central_dir);
  off_t out_length = stripper.Run();
  SYSCALL(ftruncate(fd_out, out_length));
  SYSCALL(close(fd_out));
  SYSCALL(close(fd_in));

  if (verbose) {
    fprintf(stderr, "INFO: produced interface jar: %s -> %s (%d%%).\n",
            file_in, file_out, (int) (100.0 * out_length / length));
  }

  return 0;
}

}  // namespace devtools_ijar
