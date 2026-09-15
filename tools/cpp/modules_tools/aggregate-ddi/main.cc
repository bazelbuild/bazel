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

#include "tools/cpp/modules_tools/aggregate-ddi/aggregate-ddi.h"

// Main function
int main(int argc, char *argv[]) {
  std::vector<std::string> cpp20modules_info;
  std::vector<std::string> ddi;
  std::vector<std::string> module_file;
  std::string output;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-m" && i + 1 < argc) {
      cpp20modules_info.emplace_back(argv[++i]);
    } else if (arg == "-d" && i + 1 < argc) {
      ddi.emplace_back(argv[++i]);
    } else if (arg == "-f" && i + 1 < argc) {
      module_file.emplace_back(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output = argv[++i];
    } else {
      std::cerr << "ERROR: Unknown or incomplete argument: " << arg
                << std::endl;
      std::exit(1);
    }
  }
  if (ddi.size() != module_file.size()) {
    std::cerr << "ERROR: The number of -d and -f arguments must match."
              << std::endl;
    std::exit(1);
  }
  if (output.empty()) {
    std::cerr << "ERROR: output not specified" << std::endl;
    std::exit(1);
  }

  Cpp20ModulesInfo full_info{};

  // Process cpp20modules_info files
  for (const auto &info_filename : cpp20modules_info) {
    std::ifstream info_stream(info_filename);
    auto info = parse_info(info_stream);
    full_info.merge(info);
  }

  // Process ddi files
  for (std::size_t i = 0; i < ddi.size(); i++) {
    auto ddi_filename = ddi[i];
    auto pcm_path = module_file[i];
    std::ifstream ddi_stream(ddi_filename);
    auto dep = parse_ddi(ddi_stream);
    if (dep.gen_bmi) {
      full_info.modules[dep.name] = pcm_path;
      full_info.usages[dep.name] = dep.require_list;
    }
  }

  // Write final output to file
  std::ofstream of(output);
  if (!of.is_open()) {
    std::cerr << "ERROR: Failed to open the file " << output << "\n";
    std::exit(1);
  }
  write_output(of, full_info);

  return 0;
}
