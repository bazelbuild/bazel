// Copyright 2026 The Bazel Authors. All rights reserved.
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
#include <string>

/**
 * This tool concatenates two input files into a single output file.
 * It provides the same basic functionality as the system 'cat' command
 * but is contained entirely within the Bazel repository to avoid
 * relying on the host environment's 'cat' utility.
 *
 * Usage: simple_catter <input_file1> <input_file2> <output_file>
 */
int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input_file1> <input_file2> <output_file>\n";
    return 1;
  }

  std::string input1_path = argv[1];
  std::string input2_path = argv[2];
  std::string output_path = argv[3];

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    std::cerr << "Error opening output file: " << output_path << "\n";
    return 1;
  }

  // Copy input 1
  std::ifstream in(input1_path, std::ios::binary);
  if (!in) {
    std::cerr << "Error opening input file: " << input1_path << "\n";
    return 1;
  }
  out << in.rdbuf();
  if (!out) {
    std::cerr << "Error writing to output file while copying: " << input1_path
              << "\n";
    return 1;
  }
  in.close();

  // Copy input 2
  in.open(input2_path, std::ios::binary);
  if (!in) {
    std::cerr << "Error opening input file: " << input2_path << "\n";
    return 1;
  }
  out << in.rdbuf();
  if (!out) {
    std::cerr << "Error writing to output file while copying: " << input2_path
              << "\n";
    return 1;
  }

  in.close();
  out.close();

  return 0;
}
