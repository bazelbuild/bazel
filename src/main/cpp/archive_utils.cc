// Copyright 2014 The Bazel Authors. All rights reserved.
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
#include "src/main/cpp/archive_utils.h"

#include <vector>

#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
#include "src/main/cpp/util/strings.h"
#include "third_party/ijar/zip.h"

namespace blaze {

using std::vector;

// A devtools_ijar::ZipExtractorProcessor that processes the ZIP entries using
// the given PureZipExtractorProcessors.
CompoundZipProcessor::CompoundZipProcessor(
    const vector<PureZipExtractorProcessor *> &processors)
    : processors_(processors) {}

bool CompoundZipProcessor::Accept(const char *filename,
                                  const devtools_ijar::u4 attr) {
  bool should_accept = false;
  for (auto *processor : processors_) {
    if (processor->Accept(filename, attr)) {
      // ZipExtractorProcessor::Accept is allowed to be side-effectful, so
      // we don't want to break out on the first true here.
      should_accept = true;
    }
  }
  return should_accept;
}

void CompoundZipProcessor::Process(const char *filename,
                                   const devtools_ijar::u4 attr,
                                   const devtools_ijar::u1 *data,
                                   const size_t size) {
  for (auto *processor : processors_) {
    if (processor->AcceptPure(filename, attr)) {
      processor->Process(filename, attr, data, size);
    }
  }
}

// A PureZipExtractorProcessor to extract the InstallKeyFile
GetInstallKeyFileProcessor::GetInstallKeyFileProcessor(string *install_base_key)
    : install_base_key_(install_base_key) {}

bool GetInstallKeyFileProcessor::AcceptPure(
    const char *filename, const devtools_ijar::u4 attr) const {
  return strcmp(filename, "install_base_key") == 0;
}

void GetInstallKeyFileProcessor::Process(const char *filename,
                                         const devtools_ijar::u4 attr,
                                         const devtools_ijar::u1 *data,
                                         const size_t size) {
  string str(reinterpret_cast<const char *>(data), size);
  blaze_util::StripWhitespace(&str);
  if (str.size() != 32) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract install_base_key: file size mismatch "
           "(should be 32, is "
        << str.size() << ")";
  }
  *install_base_key_ = str;
}

NoteAllFilesZipProcessor::NoteAllFilesZipProcessor(
    std::vector<std::string> *files)
    : files_(files) {}

bool NoteAllFilesZipProcessor::AcceptPure(const char *filename,
                                          const devtools_ijar::u4 attr) const {
  return false;
}

bool NoteAllFilesZipProcessor::Accept(const char *filename,
                                      const devtools_ijar::u4 attr) {
  files_->push_back(filename);
  return false;
}

void NoteAllFilesZipProcessor::Process(const char *filename,
                                       const devtools_ijar::u4 attr,
                                       const devtools_ijar::u1 *data,
                                       const size_t size) {
  BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
      << "NoteAllFilesZipProcessor::Process shouldn't be called";
}

ExtractBlazeZipProcessor::ExtractBlazeZipProcessor(
    const string &embedded_binaries, blaze::embedded_binaries::Dumper *dumper)
    : embedded_binaries_(embedded_binaries), dumper_(dumper) {}

bool ExtractBlazeZipProcessor::AcceptPure(const char *filename,
                                          const devtools_ijar::u4 attr) const {
  return !devtools_ijar::zipattr_is_dir(attr);
}

void ExtractBlazeZipProcessor::Process(const char *filename,
                                       const devtools_ijar::u4 attr,
                                       const devtools_ijar::u1 *data,
                                       const size_t size) {
  dumper_->Dump(data, size, blaze_util::JoinPath(embedded_binaries_, filename));
}

}  // namespace blaze
