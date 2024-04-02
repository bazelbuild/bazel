// Copyright 2024 The Bazel Authors. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/die_if_null.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "src/tools/one_version/allowlist.h"
#include "src/tools/one_version/duplicate_class_collector.h"
#include "src/tools/one_version/one_version.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/token_stream.h"
#include "src/tools/singlejar/zip_headers.h"

// Scans a classpath and reports one version violations.
//
// usage: --output <file to touch>
//        --inputs <jar1,label1 jar2,label2 ... jarN,labelN>
int main(int argc, char *argv[]) {
  std::string output_file;
  bool succeed_on_found_violations = false;
  std::vector<std::string> inputs;
  ArgTokenStream tokens(argc - 1, argv + 1);
  while (!tokens.AtEnd()) {
    if (tokens.MatchAndSet("--output", &output_file) ||
        tokens.MatchAndSet("--succeed_on_found_violations",
                           &succeed_on_found_violations) ||
        tokens.MatchAndSet("--inputs", &inputs)) {
    } else {
      std::cerr << "error: bad command line argument " << tokens.token()
                << std::endl;
      return 1;
    }
  }

  // TODO(cushon): support customizing the allowlist
  one_version::OneVersion one_version(
      std::make_unique<one_version::MapAllowlist>(
          absl::flat_hash_map<std::string,
                              absl::flat_hash_set<std::string>>()));

  for (const std::string &input : inputs) {
    std::vector<std::string> pieces = absl::StrSplit(input, ',');
    if (pieces.size() != 2) {
      std::cerr << "error: expected <jar path>,<label>, got: " << input
                << std::endl;
      return 1;
    }
    std::string jar = pieces[0];
    std::string label = pieces[1];
    InputJar input_jar;
    if (!input_jar.Open(jar)) {
      std::cerr << "error: unable to open: " << jar << std::endl;
      return 1;
    }
    const CDH *dir_entry;
    const LH *local_header;
    while ((dir_entry = input_jar.NextEntry(&local_header))) {
      absl::string_view file_name(ABSL_DIE_IF_NULL(local_header)->file_name(),
                                  local_header->file_name_length());
      one_version.Add(file_name, dir_entry,
                      one_version::Label(label, jar, /*allowlisted=*/false));
    }
    input_jar.Close();
  }

  std::vector<one_version::Violation> violations = one_version.Report();

  // Touch the output file, which exists because blaze actions are required to
  // have outputs but is always empty.
  std::ofstream o(output_file.c_str());
  if (!violations.empty()) {
    // TODO(cushon): support other classpaths (runtime, android resources)
    std::cerr << "Found one definition violations on the runtime classpath:\n"
              << one_version::DuplicateClassCollector::Report(violations);
    return succeed_on_found_violations ? 0 : 1;
  }

  return 0;
}
