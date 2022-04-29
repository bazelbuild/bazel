// Copyright 2022 The Bazel Authors. All rights reserved.
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

#ifndef BAZEL_TOOLS_CPP_BAZEL_BUILD_INFO_TRANSLATION_HELPER_H_
#define BAZEL_TOOLS_CPP_BAZEL_BUILD_INFO_TRANSLATION_HELPER_H_

#include <fstream>

#include "tools/cpp/build_info_translation_helper.h"

namespace bazel {
namespace tools {
namespace cpp {

class BazelBuildInfoTranslationHelper
    : public bazel::tools::cpp::BuildInfoTranslationHelper {
 public:
  BazelBuildInfoTranslationHelper(const std::string& info_file_path,
                                  const std::string& version_file_path)
      : bazel::tools::cpp::BuildInfoTranslationHelper(info_file_path,
                                                      version_file_path) {}

  BuildInfoTranslationHelper::KeyMap getStableKeys() override;

  BuildInfoTranslationHelper::KeyMap getVolatileKeys() override;
};
}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // BAZEL_TOOLS_CPP_BAZEL_BUILD_INFO_TRANSLATION_HELPER_H_
