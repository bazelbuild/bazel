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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits.h>
#include <limits>
#include <vector>

#include "third_party/ijar/zip.h"
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
#define GENERAL_PURPOSE_BIT_FLAG_COMPRESSION_SPEED ((1 << 2) | (1 << 1))
#define GENERAL_PURPOSE_BIT_FLAG_SUPPORTED \
  (GENERAL_PURPOSE_BIT_FLAG_COMPRESSED \
  | GENERAL_PURPOSE_BIT_FLAG_UTF8_ENCODED \
  | GENERAL_PURPOSE_BIT_FLAG_COMPRESSION_SPEED)

namespace devtools_ijar {
// In the absence of ZIP64 support, zip files are limited to 4GB.
// http://www.info-zip.org/FAQ.html#limits
static const u8 kMaximumOutputSize = std::numeric_limits<uint32_t>::max();

static bool ProcessCentralDirEntry(const u1 *&p,
                                   size_t *compressed_size,
                                   size_t *uncompressed_size,
                                   char *filename,
                                   size_t filename_size,
                                   u4 *attr,
                                   u4 *offset);

//
// A class representing a ZipFile for reading. Its public API is exposed
// using the ZipExtractor abstract class.
//
class InputZipFile : public ZipExtractor {
 public:
  InputZipFile(ZipExtractorProcessor *processor, int fd, off_t in_length,
               off_t in_offset, const u1* zipdata_in, const u1* central_dir);
  virtual ~InputZipFile();

  virtual const char* GetError() {
    if (errmsg[0] == 0) {
      return NULL;
    }
    return errmsg;
  }

  virtual bool ProcessNext();
  virtual void Reset();
  virtual size_t GetSize() {
    return in_length_;
  }

  virtual u8 CalculateOutputLength();

 private:
  ZipExtractorProcessor *processor;

  int fd_in;  // Input file descripor

  // InputZipFile is responsible for maintaining the following
  // pointers. They are allocated by the Create() method before
  // the object is actually created using mmap.
  const u1 * const zipdata_in_;   // start of input file mmap
  const u1 * zipdata_in_mapped_;  // start of still mapped region
  const u1 * const central_dir_;  // central directory in input file

  size_t in_length_;  // size of the input file
  size_t in_offset_;  // offset  the input file

  const u1 *p;  // input cursor

  const u1* central_dir_current_;  // central dir input cursor

  // Buffer size is initially INITIAL_BUFFER_SIZE. It doubles in size every
  // time it is found too small, until it reaches MAX_BUFFER_SIZE. If that is
  // not enough, we bail out. We only decompress class files, so they should
  // be smaller than 64K anyway, but we give a little leeway.
  // MAX_BUFFER_SIZE must be bigger than the size of the biggest file in the
  // ZIP. It is set to 128M here so we can uncompress the Bazel server with
  // this library.
  static const size_t INITIAL_BUFFER_SIZE = 256 * 1024;  // 256K
  static const size_t MAX_BUFFER_SIZE = 128 * 1024 * 1024;
  static const size_t MAX_MAPPED_REGION = 32 * 1024 * 1024;

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

  // Copy of the last filename entry - Null-terminated.
  char filename[PATH_MAX];
  // The external file attribute field
  u4 attr;

  // last error
  char errmsg[4*PATH_MAX];

  int error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errmsg, 4*PATH_MAX, fmt, ap);
    va_end(ap);
    return -1;
  }

  // Check that at least n bytes remain in the input file, otherwise
  // abort with an error message.  "state" is the name of the field
  // we're about to read, for diagnostics.
  int EnsureRemaining(size_t n, const char *state) {
    size_t in_offset = p - zipdata_in_;
    size_t remaining = in_length_ - in_offset;
    if (n > remaining) {
      return error("Premature end of file (at offset %zd, state=%s); "
                   "expected %zd more bytes but found %zd.\n",
                   in_offset, state, n, remaining);
    }
    return 0;
  }

  // Read one entry from input zip file
  int ProcessLocalFileEntry(size_t compressed_size, size_t uncompressed_size);

  // Uncompress a file from the archive using zlib. The pointer returned
  // is owned by InputZipFile, so it must not be freed. Advances the input
  // cursor to the first byte after the compressed data.
  u1* UncompressFile();

  // Skip a file
  int SkipFile(const bool compressed);

  // Process a file
  int ProcessFile(const bool compressed);
};

//
// A class implementing ZipBuilder that represent an open zip file for writing.
//
class OutputZipFile : public ZipBuilder {
 public:
  OutputZipFile(int fd, u1 * const zipdata_out) :
      fd_out(fd),
      zipdata_out_(zipdata_out),
      q(zipdata_out) {
    errmsg[0] = 0;
  }

  virtual const char* GetError() {
    if (errmsg[0] == 0) {
      return NULL;
    }
    return errmsg;
  }

  virtual ~OutputZipFile() { Finish(); }
  virtual u1* NewFile(const char* filename, const u4 attr);
  virtual int FinishFile(size_t filelength, bool compress = false);
  virtual int WriteEmptyFile(const char *filename);
  virtual size_t GetSize() {
    return Offset(q);
  }
  virtual int GetNumberFiles() {
    return entries_.size();
  }
  virtual int Finish();

 private:
  struct LocalFileEntry {
    // Start of the local header (in the output buffer).
    size_t local_header_offset;

    // Sizes of the file entry
    size_t uncompressed_length;
    size_t compressed_length;

    // Compression method
    u2 compression_method;

    // external attributes field
    u4 external_attr;

    // Start/length of the file_name in the local header.
    u1 *file_name;
    u2 file_name_length;

    // Start/length of the extra_field in the local header.
    const u1 *extra_field;
    u2 extra_field_length;
  };

  int fd_out;  // file descriptor for the output file

  // OutputZipFile is responsible for maintaining the following
  // pointers. They are allocated by the Create() method before
  // the object is actually created using mmap.
  u1 * const zipdata_out_;        // start of output file mmap
  u1 *q;  // output cursor

  u1 *header_ptr;  // Current pointer to "compression method" entry.

  // List of entries to write the central directory
  std::vector<LocalFileEntry*> entries_;

  // last error
  char errmsg[4*PATH_MAX];

  int error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(errmsg, 4*PATH_MAX, fmt, ap);
    va_end(ap);
    return -1;
  }

  // Write the ZIP central directory structure for each local file
  // entry in "entries".
  void WriteCentralDirectory();

  // Returns the offset of the pointer relative to the start of the
  // output zip file.
  size_t Offset(const u1 *const x) {
    return x - zipdata_out_;
  }

  // Write ZIP file header in the output. Since the compressed size is not
  // known in advance, it must be recorded later. This method returns a pointer
  // to "compressed size" in the file header that should be passed to
  // WriteFileSizeInLocalFileHeader() later.
  u1* WriteLocalFileHeader(const char *filename, const u4 attr);

  // Fill in the "compressed size" and "uncompressed size" fields in a local
  // file header previously written by WriteLocalFileHeader().
  size_t WriteFileSizeInLocalFileHeader(u1 *header_ptr, size_t out_length,
                                        bool compress = false);
};

//
// Implementation of InputZipFile
//
bool InputZipFile::ProcessNext() {
  // Process the next entry in the central directory. Also make sure that the
  // content pointer is in sync.
  size_t compressed, uncompressed;
  u4 offset;
  if (!ProcessCentralDirEntry(central_dir_current_, &compressed, &uncompressed,
                              filename, PATH_MAX, &attr, &offset)) {
    return false;
  }

  // There might be an offset specified in the central directory that does
  // not match the file offset, if so, correct the pointer.
  if (offset != 0 && (p != (zipdata_in_ + in_offset_ + offset))) {
    p = zipdata_in_ + offset;
  }

  if (EnsureRemaining(4, "signature") < 0) {
    return false;
  }
  u4 signature = get_u4le(p);
  if (signature == LOCAL_FILE_HEADER_SIGNATURE) {
    if (ProcessLocalFileEntry(compressed, uncompressed) < 0) {
      return false;
    }
  } else {
    error("local file header signature for file %s not found\n", filename);
    return false;
  }

  return true;
}

int InputZipFile::ProcessLocalFileEntry(
    size_t compressed_size, size_t uncompressed_size) {
  if (EnsureRemaining(26, "extract_version") < 0) {
    return -1;
  }
  extract_version_ = get_u2le(p);
  general_purpose_bit_flag_ = get_u2le(p);

  if ((general_purpose_bit_flag_ & ~GENERAL_PURPOSE_BIT_FLAG_SUPPORTED) != 0) {
    return error("Unsupported value (0x%04x) in general purpose bit flag.\n",
                 general_purpose_bit_flag_);
  }

  compression_method_ = get_u2le(p);

  if (compression_method_ != COMPRESSION_METHOD_DEFLATED &&
      compression_method_ != COMPRESSION_METHOD_STORED) {
    return error("Unsupported compression method (%d).\n",
                 compression_method_);
  }

  // skip over: last_mod_file_time, last_mod_file_date, crc32
  p += 2 + 2 + 4;
  compressed_size_ = get_u4le(p);
  uncompressed_size_ = get_u4le(p);
  file_name_length_ = get_u2le(p);
  extra_field_length_ = get_u2le(p);

  if (EnsureRemaining(file_name_length_, "file_name") < 0) {
    return -1;
  }
  file_name_ = p;
  p += file_name_length_;

  if (EnsureRemaining(extra_field_length_, "extra_field") < 0) {
    return -1;
  }
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
      return error("central directory and file header inconsistent\n");
    }
  }

  if (uncompressed_size_ == 0) {
    uncompressed_size_ = uncompressed_size;
  } else {
    if (uncompressed_size_ != uncompressed_size) {
      return error("central directory and file header inconsistent\n");
    }
  }

  if (processor->Accept(filename, attr)) {
    if (ProcessFile(is_compressed) < 0) {
      return -1;
    }
  } else {
    if (SkipFile(is_compressed) < 0) {
      return -1;
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

  if (p > zipdata_in_mapped_ + MAX_MAPPED_REGION) {
    munmap(const_cast<u1 *>(zipdata_in_mapped_), MAX_MAPPED_REGION);
    zipdata_in_mapped_ += MAX_MAPPED_REGION;
  }

  return 0;
}

int InputZipFile::SkipFile(const bool compressed) {
  if (!compressed) {
    // In this case, compressed_size_ == uncompressed_size_ (since the file is
    // uncompressed), so we can use either.
    if (compressed_size_ != uncompressed_size_) {
      return error("compressed size != uncompressed size, although the file "
                   "is uncompressed.\n");
    }
  }

  if (EnsureRemaining(compressed_size_, "file_data") < 0) {
    return -1;
  }
  p += compressed_size_;
  return 0;
}

u1* InputZipFile::UncompressFile() {
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
    error("inflateInit: %d\n", ret);
    return NULL;
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
          error("ijar does not support decompressing files "
                "larger than %dMB.\n",
                (int) (MAX_BUFFER_SIZE/(1024*1024)));
          return NULL;
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
        error("zlib returned error code %d during inflate.\n", ret);
        return NULL;
      }
    }
  }
}

int InputZipFile::ProcessFile(const bool compressed) {
  const u1 *file_data;
  if (compressed) {
    file_data = UncompressFile();
    if (file_data == NULL) {
      return -1;
    }
  } else {
    // In this case, compressed_size_ == uncompressed_size_ (since the file is
    // uncompressed), so we can use either.
    if (compressed_size_ != uncompressed_size_) {
      return error("compressed size != uncompressed size, although the file "
                   "is uncompressed.\n");
    }

    if (EnsureRemaining(compressed_size_, "file_data") < 0) {
      return -1;
    }
    file_data = p;
    p += compressed_size_;
  }
  processor->Process(filename, attr, file_data, uncompressed_size_);
  return 0;
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
    char *filename, size_t filename_size, u4 *attr, u4 *offset) {
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
  p += 4;  // skip to external file attributes field
  *attr = get_u4le(p);
  *offset = get_u4le(p);
  {
    size_t len = (file_name_length < filename_size)
      ? file_name_length
      : (filename_size - 1);
    memcpy(reinterpret_cast<void*>(filename), p, len);
    filename[len] = 0;
  }
  p += file_name_length;
  p += extra_field_length;
  p += file_comment_length;
  return true;
}

// Gives a maximum bound on the size of the interface JAR. Basically, adds
// the difference between the compressed and uncompressed sizes to the size
// of the input file.
u8 InputZipFile::CalculateOutputLength() {
  const u1* current = central_dir_;

  u8 compressed_size = 0;
  u8 uncompressed_size = 0;
  u8 skipped_compressed_size = 0;
  u4 attr;
  u4 offset;
  char filename[PATH_MAX];

  while (true) {
    size_t file_compressed, file_uncompressed;
    if (!ProcessCentralDirEntry(current,
                                &file_compressed, &file_uncompressed,
                                filename, PATH_MAX, &attr, &offset)) {
      break;
    }

    if (processor->Accept(filename, attr)) {
      compressed_size += (u8) file_compressed;
      uncompressed_size += (u8) file_uncompressed;
    } else {
      skipped_compressed_size += file_compressed;
    }
  }

  // The worst case is when the output is simply the input uncompressed. The
  // metadata in the zip file will stay the same, so the file will grow by the
  // difference between the compressed and uncompressed sizes.
  return (u8) in_length_ - skipped_compressed_size
      + (uncompressed_size - compressed_size);
}

// Given the data in the zip file, returns the offset of the central directory
// and the number of files contained in it.
bool FindZipCentralDirectory(const u1* bytes, size_t in_length,
                             u4* offset, const u1** central_dir) {
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

  const u1* end_of_central_dir = current;
  get_u4le(current);  // central directory locator signature, already checked
  u2 number_of_this_disk = get_u2le(current);
  u2 disk_with_central_dir = get_u2le(current);
  u2 central_dir_entries_on_this_disk = get_u2le(current);
  u2 central_dir_entries = get_u2le(current);
  u4 central_dir_size = get_u4le(current);
  u4 central_dir_offset = get_u4le(current);
  u2 file_comment_length = get_u2le(current);
  current += file_comment_length;  // set current to the end of the central dir

  if (number_of_this_disk != 0
    || disk_with_central_dir != 0
    || central_dir_entries_on_this_disk != central_dir_entries) {
    fprintf(stderr, "multi-disk JAR files are not supported\n");
    return false;
  }

  // Do not change output values before determining that they are OK.
  *offset = central_dir_offset;
  // Central directory start can then be used to determine the actual
  // starts of the zip file (which can be different in case of a non-zip
  // header like for auto-extractable binaries).
  *central_dir = end_of_central_dir - central_dir_size;
  return true;
}

void InputZipFile::Reset() {
  central_dir_current_ = central_dir_;
  zipdata_in_mapped_ = zipdata_in_;
  p = zipdata_in_ + in_offset_;
}

int ZipExtractor::ProcessAll() {
  while (ProcessNext()) {}
  if (GetError() != NULL) {
    return -1;
  }
  return 0;
}

ZipExtractor* ZipExtractor::Create(const char* filename,
                                   ZipExtractorProcessor *processor) {
  int fd_in = open(filename, O_RDONLY);
  if (fd_in < 0) {
    return NULL;
  }

  off_t length = lseek(fd_in, 0, SEEK_END);
  if (length < 0) {
    return NULL;
  }

  void *zipdata_in = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd_in, 0);
  if (zipdata_in == MAP_FAILED) {
    return NULL;
  }

  u4 central_dir_offset;
  const u1 *central_dir = NULL;

  if (!devtools_ijar::FindZipCentralDirectory(
          static_cast<const u1*>(zipdata_in), length,
          &central_dir_offset, &central_dir)) {
    errno = EIO;  // we don't really have a good error number
    return NULL;
  }
  const u1 *zipdata_start = static_cast<const u1*>(zipdata_in);
  off_t offset = - static_cast<off_t>(zipdata_start
                                      + central_dir_offset
                                      - central_dir);

  return new InputZipFile(processor, fd_in, length, offset,
                          zipdata_start, central_dir);
}

InputZipFile::InputZipFile(ZipExtractorProcessor *processor, int fd,
                           off_t in_length, off_t in_offset,
                           const u1* zipdata_in, const u1* central_dir)
  : processor(processor), fd_in(fd),
    zipdata_in_(zipdata_in), zipdata_in_mapped_(zipdata_in),
    central_dir_(central_dir), in_length_(in_length), in_offset_(in_offset),
    p(zipdata_in + in_offset), central_dir_current_(central_dir) {
  uncompressed_data_allocated_ = INITIAL_BUFFER_SIZE;
  uncompressed_data_ =
    reinterpret_cast<u1*>(malloc(uncompressed_data_allocated_));
  errmsg[0] = 0;
}

InputZipFile::~InputZipFile() {
  free(uncompressed_data_);
  close(fd_in);
}


//
// Implementation of OutputZipFile
//
int OutputZipFile::WriteEmptyFile(const char *filename) {
  const u1* file_name = (const u1*) filename;
  size_t file_name_length = strlen(filename);

  LocalFileEntry *entry = new LocalFileEntry;
  entry->local_header_offset = Offset(q);
  entry->external_attr = 0;

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
  entry->compressed_length = 0;
  entry->uncompressed_length = 0;
  entry->compression_method = 0;
  entry->extra_field = (const u1 *)"";
  entry->file_name = (u1*) strdup((const char *) file_name);
  entries_.push_back(entry);

  return 0;
}

void OutputZipFile::WriteCentralDirectory() {
  // central directory:
  const u1 *central_directory_start = q;
  for (size_t ii = 0; ii < entries_.size(); ++ii) {
    LocalFileEntry *entry = entries_[ii];
    put_u4le(q, CENTRAL_FILE_HEADER_SIGNATURE);
    put_u2le(q, 0);  // version made by

    put_u2le(q, ZIP_VERSION_TO_EXTRACT);  // version to extract
    put_u2le(q, 0);  // general purpose bit flag
    put_u2le(q, entry->compression_method);  // compression method:
    put_u2le(q, 0);                          // last_mod_file_time
    put_u2le(q, 0);  // last_mod_file_date
    put_u4le(q, 0);  // crc32 (jar/javac tools don't care)
    put_u4le(q, entry->compressed_length);    // compressed_size
    put_u4le(q, entry->uncompressed_length);  // uncompressed_size
    put_u2le(q, entry->file_name_length);
    put_u2le(q, entry->extra_field_length);

    put_u2le(q, 0);  // file comment length
    put_u2le(q, 0);  // disk number start
    put_u2le(q, 0);  // internal file attributes
    put_u4le(q, entry->external_attr);  // external file attributes
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

u1* OutputZipFile::WriteLocalFileHeader(const char* filename, const u4 attr) {
  off_t file_name_length_ = strlen(filename);
  LocalFileEntry *entry = new LocalFileEntry;
  entry->local_header_offset = Offset(q);
  entry->file_name_length = file_name_length_;
  entry->file_name = new u1[file_name_length_];
  entry->external_attr = attr;
  memcpy(entry->file_name, filename, file_name_length_);
  entry->extra_field_length = 0;
  entry->extra_field = (const u1 *)"";

  // Output the ZIP local_file_header:
  put_u4le(q, LOCAL_FILE_HEADER_SIGNATURE);
  put_u2le(q, ZIP_VERSION_TO_EXTRACT);     // version to extract
  put_u2le(q, 0);                          // general purpose bit flag
  u1 *header_ptr = q;
  put_u2le(q, COMPRESSION_METHOD_STORED);  // compression method = placeholder
  put_u2le(q, 0);                          // last_mod_file_time
  put_u2le(q, 0);                          // last_mod_file_date
  put_u4le(q, 0);                          // crc32 (jar/javac tools don't care)
  put_u4le(q, 0);  // compressed_size = placeholder
  put_u4le(q, 0);  // uncompressed_size = placeholder
  put_u2le(q, entry->file_name_length);
  put_u2le(q, entry->extra_field_length);

  put_n(q, entry->file_name, entry->file_name_length);
  put_n(q, entry->extra_field, entry->extra_field_length);
  entries_.push_back(entry);

  return header_ptr;
}

// Try to compress a file entry in memory using the deflate algorithm.
// It will compress buf (of size length) unless the compressed size is bigger
// than the input size. The result will overwrite the content of buf and the
// final size is returned.
size_t TryDeflate(u1 *buf, size_t length) {
  u1 *outbuf = reinterpret_cast<u1 *>(malloc(length));
  z_stream stream;

  // Initialize the z_stream strcut for reading from buf and wrinting in outbuf.
  stream.zalloc = Z_NULL;
  stream.zfree = Z_NULL;
  stream.opaque = Z_NULL;
  stream.total_in = length;
  stream.avail_in = length;
  stream.total_out = length;
  stream.avail_out = length;
  stream.next_in = buf;
  stream.next_out = outbuf;

  if (deflateInit(&stream, Z_DEFAULT_COMPRESSION) != Z_OK) {
    // Failure to compress => return the buffer uncompressed
    free(outbuf);
    return length;
  }

  if (deflate(&stream, Z_FINISH) == Z_STREAM_END) {
    // Compression successful and fits in outbuf, let's copy the result in buf.
    length = stream.total_out;
    memcpy(buf, outbuf, length);
  }

  deflateEnd(&stream);
  free(outbuf);

  // Return the length of the resulting buffer
  return length;
}

size_t OutputZipFile::WriteFileSizeInLocalFileHeader(u1 *header_ptr,
                                                     size_t out_length,
                                                     bool compress) {
  size_t compressed_size = out_length;
  if (compress) {
    compressed_size = TryDeflate(q, out_length);
  }
  // compression method
  if (compressed_size < out_length) {
    put_u2le(header_ptr, COMPRESSION_METHOD_DEFLATED);
  } else {
    put_u2le(header_ptr, COMPRESSION_METHOD_STORED);
  }
  header_ptr += 8;
  put_u4le(header_ptr, compressed_size);  // compressed_size
  put_u4le(header_ptr, out_length);       // uncompressed_size
  return compressed_size;
}

int OutputZipFile::Finish() {
  if (fd_out > 0) {
    WriteCentralDirectory();
    if (ftruncate(fd_out, GetSize()) < 0) {
      return error("ftruncate(fd_out, GetSize()): %s", strerror(errno));
    }
    if (close(fd_out) < 0) {
      return error("close(fd_out): %s", strerror(errno));
    }
    fd_out = -1;
  }
  return 0;
}

u1* OutputZipFile::NewFile(const char* filename, const u4 attr) {
  header_ptr = WriteLocalFileHeader(filename, attr);
  return q;
}

int OutputZipFile::FinishFile(size_t filelength, bool compress) {
  size_t compressed_size =
      WriteFileSizeInLocalFileHeader(header_ptr, filelength, compress);
  entries_.back()->compressed_length = compressed_size;
  entries_.back()->uncompressed_length = filelength;
  if (compressed_size < filelength) {
    entries_.back()->compression_method = COMPRESSION_METHOD_DEFLATED;
  } else {
    entries_.back()->compression_method = COMPRESSION_METHOD_STORED;
  }
  q += compressed_size;
  return 0;
}

ZipBuilder* ZipBuilder::Create(const char* zip_file, u8 estimated_size) {
  if (estimated_size > kMaximumOutputSize) {
    fprintf(stderr,
            "Uncompressed input jar has size %llu, "
            "which exceeds the maximum supported output size %llu.\n"
            "Assuming that ijar will be smaller and hoping for the best.\n",
            estimated_size, kMaximumOutputSize);
    estimated_size = kMaximumOutputSize;
  }

  int fd_out = open(zip_file, O_CREAT|O_RDWR|O_TRUNC, 0644);
  if (fd_out < 0) {
    return NULL;
  }

  // Create mmap-able sparse file
  if (ftruncate(fd_out, estimated_size) < 0) {
    return NULL;
  }

  // Ensure that any buffer overflow in JarStripper will result in
  // SIGSEGV or SIGBUS by over-allocating beyond the end of the file.
  size_t mmap_length = std::min(estimated_size + sysconf(_SC_PAGESIZE),
                                (u8) std::numeric_limits<size_t>::max());

  void *zipdata_out = mmap(NULL, mmap_length, PROT_WRITE,
                           MAP_SHARED, fd_out, 0);
  if (zipdata_out == MAP_FAILED) {
    fprintf(stderr, "output_length=%llu\n", estimated_size);
    return NULL;
  }

  return new OutputZipFile(fd_out, (u1*) zipdata_out);
}

u8 ZipBuilder::EstimateSize(char **files) {
  struct stat statst;
  // Digital signature field size = 6, End of central directory = 22, Total = 28
  u8 size = 28;
  // Count the size of all the files in the input to estimate the size of the
  // output.
  for (int i = 0; files[i] != NULL; i++) {
    if (stat(files[i], &statst) != 0) {
      fprintf(stderr, "File %s does not seem to exist.", files[i]);
      return 0;
    }
    size += statst.st_size;
    // Add sizes of Zip meta data
    // local file header = 30 bytes
    // data descriptor = 12 bytes
    // central directory descriptor = 46 bytes
    //    Total: 88bytes
    size += 88;
    // The filename is stored twice (once in the central directory
    // and once in the local file header).
    size += strlen(files[i]) * 2;
  }
  return size;
}

}  // namespace devtools_ijar
