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

#include "src/main/cpp/blaze_util_platform.h"
#include "src/main/cpp/util/errors.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/exit_code.h"
#include "src/main/cpp/util/logging.h"
#include "src/main/cpp/util/path.h"
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
      const vector<PureZipExtractorProcessor*>& processors)
      : processors_(processors) {}

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
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

  void Process(const char *filename, const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data, const size_t size) override {
    for (auto *processor : processors_) {
      if (processor->AcceptPure(filename, attr)) {
        processor->Process(filename, attr, data, size);
      }
    }
  }

 private:
  const vector<PureZipExtractorProcessor*> processors_;
};

// A PureZipExtractorProcessor to extract the InstallKeyFile
class GetInstallKeyFileProcessor : public PureZipExtractorProcessor {
 public:
  explicit GetInstallKeyFileProcessor(string *install_base_key)
    : install_base_key_(install_base_key) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return strcmp(filename, "install_base_key") == 0;
  }

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  void Process(const char *filename,
               const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data,
               const size_t size) override {
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

 private:
  string *install_base_key_;
};

// A PureZipExtractorProcessor that adds the names of all the files ZIP up in
// the Blaze binary to the given vector.
class NoteAllFilesZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit NoteAllFilesZipProcessor(std::vector<std::string>* files)
      : files_(files) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return false;
  }

  bool Accept(const char *filename,
              const devtools_ijar::u4 attr) override {
    files_->push_back(filename);
    return false;
  }

  void Process(const char *filename,
               const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data,
               const size_t size) override {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "NoteAllFilesZipProcessor::Process shouldn't be called";
  }

 private:
  std::vector<std::string>* files_;
};

// A PureZipExtractorProcessor to extract the files from the blaze zip.
class ExtractBlazeZipProcessor : public PureZipExtractorProcessor {
 public:
  explicit ExtractBlazeZipProcessor(const string &output_dir,
                                    blaze::embedded_binaries::Dumper *dumper)
      : output_dir_(output_dir), dumper_(dumper) {}

  bool AcceptPure(const char *filename,
                  const devtools_ijar::u4 attr) const override {
    return !devtools_ijar::zipattr_is_dir(attr);
  }

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return AcceptPure(filename, attr);
  }

  void Process(const char *filename,
               const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data,
               const size_t size) override {
    dumper_->Dump(data, size, blaze_util::JoinPath(output_dir_, filename));
  }

 private:
  const string output_dir_;
  blaze::embedded_binaries::Dumper *dumper_;
};

// A ZipExtractorProcessor that reads the contents of the build-label.txt file
// from the archive.
class GetBuildLabelFileProcessor
    : public devtools_ijar::ZipExtractorProcessor {
 public:
  explicit GetBuildLabelFileProcessor(string *build_label)
    : build_label_(build_label) {}

  bool Accept(const char *filename, const devtools_ijar::u4 attr) override {
    return strcmp(filename, "build-label.txt") == 0;
  }

  void Process(const char *filename,
               const devtools_ijar::u4 attr,
               const devtools_ijar::u1 *data,
               const size_t size) override {
    string contents(reinterpret_cast<const char *>(data), size);
    blaze_util::StripWhitespace(&contents);
    *build_label_ = contents;
  }

 private:
  string *build_label_;
};

static void RunZipProcessorOrDie(
    const string &archive_path,
    const string &product_name,
    devtools_ijar::ZipExtractorProcessor *processor) {
  std::unique_ptr<devtools_ijar::ZipExtractor> extractor(
      devtools_ijar::ZipExtractor::Create(archive_path.c_str(), processor));

  if (extractor == NULL) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to open " << product_name
        << " as a zip file: " << blaze_util::GetLastErrorString();
  }

  if (extractor->ProcessAll() < 0) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract install_base_key: " << extractor->GetError();
  }
}

void DetermineArchiveContents(
    const string &archive_path,
    const string &product_name,
    std::vector<std::string>* files,
    string *install_md5) {
  NoteAllFilesZipProcessor note_all_files_processor(files);
  GetInstallKeyFileProcessor install_key_processor(install_md5);
  CompoundZipProcessor processor({&note_all_files_processor,
                                  &install_key_processor});

  RunZipProcessorOrDie(archive_path, product_name, &processor);

  if (install_md5->empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to find install_base_key's in zip file";
  }
}

void ExtractArchiveOrDie(const string &archive_path,
                         const string &product_name,
                         const string &expected_install_md5,
                         const string &output_dir) {
  std::string install_md5;
  GetInstallKeyFileProcessor install_key_processor(&install_md5);

  std::string error;
  std::unique_ptr<blaze::embedded_binaries::Dumper> dumper(
      blaze::embedded_binaries::Create(&error));
  if (dumper == nullptr) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR) << error;
  }
  ExtractBlazeZipProcessor extract_blaze_processor(output_dir,
                                                   dumper.get());

  CompoundZipProcessor processor({&extract_blaze_processor,
                                  &install_key_processor});
  if (!blaze_util::MakeDirectories(output_dir, 0777)) {
    BAZEL_DIE(blaze_exit_code::INTERNAL_ERROR)
        << "couldn't create '" << output_dir
        << "': " << blaze_util::GetLastErrorString();
  }

  BAZEL_LOG(USER) << "Extracting " << product_name
                  << " installation...";

  RunZipProcessorOrDie(archive_path, product_name, &processor);

  if (!dumper->Finish(&error)) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Failed to extract embedded binaries: " << error;
  }

  if (install_md5 != expected_install_md5) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "The " << product_name << " binary at " << archive_path
        << " was replaced during the client's self-extraction (old md5: "
        << expected_install_md5 << " new md5: " << install_md5
        << "). If you expected this then you should simply re-run "
        << product_name
        << " in order to pick up the different version. If you didn't expect "
           "this then you should investigate what happened.";
  }
}

void ExtractBuildLabel(const string &archive_path,
                       const string &product_name,
                       string *build_label) {
  GetBuildLabelFileProcessor processor(build_label);
  RunZipProcessorOrDie(archive_path, product_name, &processor);

  // We expect the build label file to exist and be non-empty, if neither is the
  // case then something unexpected is going on.
  if (build_label->empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Couldn't determine build label from archive";
  }
}

std::string GetServerJarPath(
    const std::vector<std::string> &archive_contents) {
  if (archive_contents.empty()) {
    BAZEL_DIE(blaze_exit_code::LOCAL_ENVIRONMENTAL_ERROR)
        << "Couldn't find server jar in archive";
  }
  return archive_contents[0];
}

}  // namespace blaze
