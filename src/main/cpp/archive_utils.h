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

namespace blaze {

// Determines the contents of the archive, storing the names of the contained
// files into `files` and the install md5 key into `install_md5`.
void DetermineArchiveContents(const std::string &archive_path,
                              const std::string &product_name,
                              std::vector<std::string> *files,
                              std::string *install_md5);

// Extracts the embedded data files in `archive_path` into `output_dir`.
// Fails if `expected_install_md5` doesn't match that contained in the archive,
// as this could indicate that the contents has unexpectedly changed.
void ExtractArchiveOrDie(const std::string &archive_path,
                         const std::string &product_name,
                         const std::string &expected_install_md5,
                         const std::string &output_dir);

// Retrieves the build label (version string) from `archive_path` into
// `build_label`.
void ExtractBuildLabel(const std::string &archive_path,
                       const std::string &product_name,
                       std::string *build_label);

// Returns the server jar path from the archive contents.
std::string GetServerJarPath(const std::vector<std::string> &archive_contents);

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_ARCHIVE_UTILS_H_
