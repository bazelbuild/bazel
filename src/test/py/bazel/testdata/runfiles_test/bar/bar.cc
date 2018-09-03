// Copyright 2018 The Bazel Authors. All rights reserved.
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

// Mock C++ binary, only used in tests.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "tools/cpp/runfiles/runfiles.h"

namespace {

using bazel::tools::cpp::runfiles::Runfiles;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unique_ptr;

bool is_file(const string& path) {
  if (path.empty()) {
    return false;
  }
  return ifstream(path).is_open();
}

int _main(int argc, char** argv) {
  cout << "Hello C++ Bar!" << endl;
  string error;
  unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
  if (runfiles == nullptr) {
    cerr << "ERROR[" << __FILE__ << "]: " << error << endl;
    return 1;
  }
  string path = runfiles->Rlocation("foo_ws/bar/bar-cc-data.txt");
  if (!is_file(path)) {
    return 1;
  }
  cout << "rloc=" << path << endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return _main(argc, argv); }
