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

#include "src/tools/singlejar/combiners.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "src/tools/singlejar/diag.h"

Combiner::~Combiner() {}

Concatenator::~Concatenator() {}

bool Concatenator::Merge(const CDH *cdh, const LH *lh) {
  if (insert_newlines_ && buffer_.get() && buffer_->data_size() &&
      '\n' != buffer_->last_byte()) {
    Append("\n", 1);
  }
  CreateBuffer();
  if (Z_NO_COMPRESSION == lh->compression_method()) {
    buffer_->ReadEntryContents(cdh, lh);
  } else if (Z_DEFLATED == lh->compression_method()) {
    if (!inflater_) {
      inflater_.reset(new Inflater());
    }
    buffer_->DecompressEntryContents(cdh, lh, inflater_.get());
  } else {
    diag_errx(2, "%s is neither stored nor deflated", filename_.c_str());
  }
  return true;
}

void *Concatenator::OutputEntry(bool compress) {
  if (!buffer_) {
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
  bool huge_buffer = ziph::zfield_needs_ext64(buffer_->data_size());
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
  lh->last_mod_file_time(1);                     // 00:00:01
  lh->last_mod_file_date(30 << 9 | 1 << 5 | 1);  // 2010-01-01
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
  uint16_t method;
  if (compress) {
    method = buffer_->CompressOut(lh->data(), &checksum, &compressed_size);
  } else {
    buffer_->CopyOut(lh->data(), &checksum);
    method = Z_NO_COMPRESSION;
    compressed_size = buffer_->data_size();
  }
  lh->crc32(checksum);
  lh->compression_method(method);
  if (huge_buffer) {
    lh->compressed_file_size32(ziph::zfield_needs_ext64(compressed_size)
                                   ? 0xFFFFFFFF
                                   : compressed_size);
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

NullCombiner::~NullCombiner() {}

bool NullCombiner::Merge(const CDH * /*cdh*/, const LH * /*lh*/) {
  return true;
}

void *NullCombiner::OutputEntry(bool /*compress*/) { return nullptr; }

XmlCombiner::~XmlCombiner() {}

bool XmlCombiner::Merge(const CDH *cdh, const LH *lh) {
  if (!concatenator_) {
    concatenator_.reset(new Concatenator(filename_, false));
    concatenator_->Append(start_tag_);
    concatenator_->Append("\n");
  }
  // To ensure xml concatentation is idempotent, read in the entry being added
  // and remove the start and end tags if they are present.
  TransientBytes bytes_;
  if (Z_NO_COMPRESSION == lh->compression_method()) {
    bytes_.ReadEntryContents(cdh, lh);
  } else if (Z_DEFLATED == lh->compression_method()) {
    if (!inflater_) {
      inflater_.reset(new Inflater());
    }
    bytes_.DecompressEntryContents(cdh, lh, inflater_.get());
  } else {
    diag_errx(2, "%s is neither stored nor deflated", filename_.c_str());
  }
  uint32_t checksum;
  char *buf = reinterpret_cast<char *>(malloc(bytes_.data_size()));
  // TODO(b/37631490): optimize this to avoid copying the bytes twice
  bytes_.CopyOut(reinterpret_cast<uint8_t *>(buf), &checksum);
  int start_offset = 0;
  if (strncmp(buf, start_tag_.c_str(), start_tag_.length()) == 0) {
    start_offset = start_tag_.length();
  }
  uint64_t end = bytes_.data_size();
  while (end >= end_tag_.length() && std::isspace(buf[end - 1])) end--;
  if (strncmp(buf + end - end_tag_.length(), end_tag_.c_str(),
              end_tag_.length()) == 0) {
    end -= end_tag_.length();
  } else {
    // Leave trailing whitespace alone if we didn't find a match.
    end = bytes_.data_size();
  }
  concatenator_->Append(buf + start_offset, end - start_offset);
  free(buf);
  return true;
}

void *XmlCombiner::OutputEntry(bool compress) {
  if (!concatenator_) {
    return nullptr;
  }
  concatenator_->Append(end_tag_);
  concatenator_->Append("\n");
  return concatenator_->OutputEntry(compress);
}

PropertyCombiner::~PropertyCombiner() {}

bool PropertyCombiner::Merge(const CDH * /*cdh*/, const LH * /*lh*/) {
  return false;  // This should not be called.
}

ManifestCombiner::~ManifestCombiner() {}

static const char *MULTI_RELEASE = "Multi-Release: true";

static const char *MULTI_RELEASE_PREFIX = "Multi-Release: ";
static const size_t MULTI_RELEASE_PREFIX_LENGTH = strlen(MULTI_RELEASE_PREFIX);

static const char *ADD_EXPORTS_PREFIX = "Add-Exports: ";
static const size_t ADD_EXPORTS_PREFIX_LENGTH = strlen(ADD_EXPORTS_PREFIX);

static const char *ADD_OPENS_PREFIX = "Add-Opens: ";
static const size_t ADD_OPENS_PREFIX_LENGTH = strlen(ADD_OPENS_PREFIX);

void ManifestCombiner::EnableMultiRelease() { multi_release_ = true; }

void ManifestCombiner::AddExports(const std::vector<std::string> &add_exports) {
  add_exports_.insert(std::end(add_exports_), std::begin(add_exports),
                      std::end(add_exports));
}

void ManifestCombiner::AddOpens(const std::vector<std::string> &add_opens) {
  add_opens_.insert(std::end(add_opens_), std::begin(add_opens),
                    std::end(add_opens));
}

bool ManifestCombiner::HandleModuleFlags(std::vector<std::string> &output,
                                         const char *key, size_t key_length,
                                         std::string line) {
  if (line.find(key, 0, key_length) == std::string::npos) {
    return false;
  }
  std::istringstream iss(line.substr(key_length));
  std::copy(std::istream_iterator<std::string>(iss),
            std::istream_iterator<std::string>(), std::back_inserter(output));
  return true;
}

void ManifestCombiner::AppendLine(const std::string &line) {
  if (line.find(MULTI_RELEASE_PREFIX, 0, MULTI_RELEASE_PREFIX_LENGTH) !=
      std::string::npos) {
    if (line.find("true", MULTI_RELEASE_PREFIX_LENGTH) != std::string::npos) {
      multi_release_ = true;
    } else if (line.find("false", MULTI_RELEASE_PREFIX_LENGTH) !=
               std::string::npos) {
      multi_release_ = false;
    }
    return;
  }
  // Handle 'Add-Exports:' and 'Add-Opens:' lines in --deploy_manifest_lines and
  // merge them with the --add_exports= and --add_opens= flags.
  if (HandleModuleFlags(add_exports_, ADD_EXPORTS_PREFIX,
                        ADD_EXPORTS_PREFIX_LENGTH, line)) {
    return;
  }
  if (HandleModuleFlags(add_opens_, ADD_OPENS_PREFIX, ADD_OPENS_PREFIX_LENGTH,
                        line)) {
    return;
  }
  concatenator_->Append(line);
  if (line[line.size() - 1] != '\n') {
    concatenator_->Append("\r\n");
  }
}

bool ManifestCombiner::Merge(const CDH *cdh, const LH *lh) {
  // Ignore Multi-Release attributes in inputs: we write the manifest first,
  // before inputs are processed, so we reply on  deploy_manifest_lines to
  // create Multi-Release jars instead of doing it automatically based on
  // the inputs.
  return true;
}

void ManifestCombiner::OutputModuleFlags(std::vector<std::string> &flags,
                                         const char *key) {
  std::sort(flags.begin(), flags.end());
  flags.erase(std::unique(flags.begin(), flags.end()), flags.end());
  if (!flags.empty()) {
    concatenator_->Append(key);
    bool first = true;
    for (const auto &flag : flags) {
      if (!first) {
        concatenator_->Append("\r\n  ");
      }
      concatenator_->Append(flag);
      first = false;
    }
    concatenator_->Append("\r\n");
  }
}

void *ManifestCombiner::OutputEntry(bool compress) {
  if (multi_release_) {
    concatenator_->Append(MULTI_RELEASE);
    concatenator_->Append("\r\n");
  }
  OutputModuleFlags(add_exports_, ADD_EXPORTS_PREFIX);
  OutputModuleFlags(add_opens_, ADD_OPENS_PREFIX);
  concatenator_->Append("\r\n");
  return concatenator_->OutputEntry(compress);
}
