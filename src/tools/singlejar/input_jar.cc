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

#include "src/tools/singlejar/input_jar.h"

bool InputJar::Open(const std::string &path) {
  if (!path_.empty()) {
    diag_errx(1, "%s:%d: This instance is already handling %s\n", __FILE__,
              __LINE__, path_.c_str());
  }
  if (!mapped_file_.Open(path)) {
    diag_warn("%s:%d: Cannot open input jar %s", __FILE__, __LINE__,
              path.c_str());
    mapped_file_.Close();
    return false;
  }
  if (mapped_file_.size() < sizeof(ECD)) {
    diag_warnx(
        "%s:%d: %s is only 0x%zx"
        " bytes long, should be at least 0x%zx bytes long",
        __FILE__, __LINE__, path.c_str(), mapped_file_.size(), sizeof(ECD));
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
    diag_warnx("%s:%d: Cannot locate  ECD record in %s", __FILE__, __LINE__,
               path.c_str());
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

  // First, consistency check the directory.
  uint32_t cen_position = ecd->cen_offset32();
  if (!ziph::zfield_has_ext64(cen_position)) {
    if (!mapped_file_.mapped(mapped_file_.address(cen_position))) {
      diag_warnx("%s:%d: %s is corrupt: Central Directory location 0x%" PRIx32
                 " is invalid",
                 __FILE__, __LINE__, path.c_str(), cen_position);
      mapped_file_.Close();
      return false;
    }
    if (mapped_file_.offset(ecd) < cen_position) {
      diag_warnx("%s:%d: %s is corrupt: End of Central Directory at 0x%" PRIx64
                 " precedes Central Directory at 0x%" PRIx32,
                 __FILE__, __LINE__, path.c_str(), mapped_file_.offset(ecd),
                 cen_position);
      mapped_file_.Close();
      return false;
    }
  }
  uint32_t cen_size = ecd->cen_size32();
  if (!ziph::zfield_has_ext64(cen_size)) {
    if (cen_size > mapped_file_.offset(ecd)) {
      diag_warnx("%s:%d: %s is corrupt: Central Directory size 0x%" PRIx32
                 " is too large",
                 __FILE__, __LINE__, path.c_str(), cen_size);
      mapped_file_.Close();
      return false;
    }
  }
  if (cen_size == 0) {
    // Empty archive, let cdh_ point to End of Central Directory.
    cdh_ = reinterpret_cast<const CDH *>(ecd);
    preamble_size_ = mapped_file_.offset(cdh_) - cen_position;
  } else {
    auto ecd64loc = reinterpret_cast<const ECD64Locator *>(
        ziph::byte_ptr(ecd) - sizeof(ECD64Locator));
    if (ecd64loc->is()) {
      auto ecd64 = reinterpret_cast<const ECD64 *>(ziph::byte_ptr(ecd64loc) -
                                                   sizeof(ECD64));
      if (!ecd64->is()) {
        diag_warnx(
            "%s:%d: %s is corrupt, expected ECD64 record at offset 0x%" PRIx64
            " is missing",
            __FILE__, __LINE__, path.c_str(), mapped_file_.offset(ecd64));
        mapped_file_.Close();
        return false;
      }
      cdh_ = reinterpret_cast<const CDH *>(ziph::byte_ptr(ecd64) -
                                           ecd64->cen_size());
      preamble_size_ = mapped_file_.offset(cdh_) - ecd64->cen_offset();
      // Find CEN and preamble size.
    } else {
      if (ziph::zfield_has_ext64(cen_size) ||
          ziph::zfield_has_ext64(cen_position)) {
        diag_warnx(
            "%s:%d: %s is corrupt, expected ECD64 locator record at "
            "offset 0x%" PRIx64 " is missing",
            __FILE__, __LINE__, path.c_str(), mapped_file_.offset(ecd64loc));
        return false;
      }
      cdh_ = reinterpret_cast<const CDH *>(ziph::byte_ptr(ecd) - cen_size);
      preamble_size_ = mapped_file_.offset(cdh_) - cen_position;
    }
    if (!cdh_->is()) {
      diag_warnx(
          "%s:%d: In %s, expected central file header signature at "
          "offset0x%" PRIx64,
          __FILE__, __LINE__, path.c_str(), mapped_file_.offset(cdh_));
      mapped_file_.Close();
      return false;
    }
  }
  path_ = path;
  return true;
}

bool InputJar::Close() {
  mapped_file_.Close();
  path_.clear();
  return true;
}
