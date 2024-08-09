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

#include "generate-modmap.h"
#include <fstream>
int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: generate-modmap <ddi-file> <cpp20modules-info-file> "
                 "<output> <compiler>"
              << std::endl;
    std::exit(1);
  }

  // Retrieve the values of the flags
  std::string ddi_filename = argv[1];
  std::string info_filename = argv[2];
  std::string output = argv[3];
  std::string compiler = argv[4];

  std::ifstream info_stream(info_filename);
  if (!info_stream.is_open()) {
    std::cerr << "ERROR: Failed to open the file " << info_filename
              << std::endl;
    std::exit(1);
  }
  std::ifstream ddi_stream(ddi_filename);
  if (!ddi_stream.is_open()) {
    std::cerr << "ERROR: Failed to open the file " << ddi_filename << std::endl;
    std::exit(1);
  }
  auto dep = parse_ddi(ddi_stream);
  auto info = parse_info(info_stream);
  auto modmap = process(dep, info);

  std::string modmap_filename = output;
  std::string modmap_dot_input_filename = modmap_filename + ".input";
  std::ofstream modmap_file_stream(modmap_filename);
  std::ofstream modmap_file_dot_input_stream(modmap_dot_input_filename);
  if (!modmap_file_stream.is_open()) {
    std::cerr << "ERROR: Failed to open the file " << modmap_filename
              << std::endl;
    std::exit(1);
  }
  if (!modmap_file_dot_input_stream.is_open()) {
    std::cerr << "ERROR: Failed to open the file " << modmap_dot_input_filename
              << std::endl;
    std::exit(1);
  }
  std::optional<ModmapItem> generated;
  if (dep.gen_bmi) {
    ModmapItem item;
    item.name = dep.name;
    item.path = info.modules[dep.name];
    generated = item;
  }
  write_modmap(modmap_file_stream, modmap_file_dot_input_stream, modmap,
               compiler, generated);
  modmap_file_stream.close();
  modmap_file_dot_input_stream.close();

  return 0;
}
