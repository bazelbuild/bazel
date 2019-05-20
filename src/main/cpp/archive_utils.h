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

#ifndef BAZEL_SRC_MAIN_CPP_ARCHIVE_UTILS_H_
#define BAZEL_SRC_MAIN_CPP_ARCHIVE_UTILS_H_

#include <string>
#include <vector>

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/strings.h"
#include "third_party/ijar/zip.h"

namespace blaze {

using std::vector;
using std::string;

// A devtools_ijar::ZipExtractorProcessor that has a pure version of Accept.
class PureZipExtractorProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  virtual ~PureZipExtractorProcessor() {}

  // Like devtools_ijar::ZipExtractorProcessor::Accept, but is guaranteed to not
  // have side-effects.
  virtual bool AcceptPure(const char *filename,
                          const devtools_ijar::u4 attr) const = 0;
};

// A devtools_ijar::ZipExtractorProcessor that processes the ZIP entries using
// the given PureZipExtractorProcessors.
class CompoundZipProcessor : public devtools_ijar::ZipExtractorProcessor {
 public:
  explicit CompoundZipProcessor(
      const vector<PureZipExtractorProcessor *> &processors);

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override;

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override;

 private:
  const vector<PureZipExtractorProcessor *> processors_;
};

// A PureZipExtractorProcessor to extract the InstallKeyFile
class GetInstallKeyFileProcessor : public PureZipExtractorProcessor {
 public:
  explicit GetInstallKeyFileProcessor(string *install_base_key);

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override;

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override;

 private:
  string *install_base_key_;
};

// A PureZipExtractorProcessor that adds the names of all the files ZIP up in
// the Blaze binary to the given vector.
class NoteAllFilesZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit NoteAllFilesZipProcessor(std::vector<std::string> *files);

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override;

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override;

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override;

 private:
  std::vector<std::string> *files_;
};

// A PureZipExtractorProcessor to extract the files from the blaze zip.
class ExtractBlazeZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit ExtractBlazeZipProcessor(const string &embedded_binaries,
                                    blaze::embedded_binaries::Dumper *dumper);

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override;

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override;

 private:
  const string embedded_binaries_;
  blaze::embedded_binaries::Dumper *dumper_;
};

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_ARCHIVE_UTILS_H_
