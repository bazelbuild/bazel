// Copyright 2025 The Bazel Authors.
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

#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, lang, "",
          "Comma-separated list of languages to stamp (e.g., cc,go).");
ABSL_FLAG(std::string, input, "",
          "Comma-separated list of input files (e.g., "
          "build-info.txt,build-changelist.txt).");
ABSL_FLAG(bool, h, false, "Recalculate the build ID.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  if (absl::GetFlag(FLAGS_lang).empty()) {
    std::cerr << "Error: --lang is a required flag." << std::endl;
    return 1;
  }
  if (absl::GetFlag(FLAGS_input).empty()) {
    std::cerr << "Error: --input is a required flag." << std::endl;
    return 1;
  }

  // TODO: b/452239395 - Implement the core stamping logic.
  std::cout << "lang: " << absl::GetFlag(FLAGS_lang) << std::endl;
  std::cout << "input: " << absl::GetFlag(FLAGS_input) << std::endl;
  std::cout << "h: " << (absl::GetFlag(FLAGS_h) ? "true" : "false")
            << std::endl;

  return 0;
}
