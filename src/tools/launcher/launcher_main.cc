// Copyright 2017 The Bazel Authors. All rights reserved.
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

#include <memory>

#include "src/tools/launcher/bash_launcher.h"
#include "src/tools/launcher/java_launcher.h"
#include "src/tools/launcher/launcher.h"
#include "src/tools/launcher/python_launcher.h"
#include "src/tools/launcher/util/data_parser.h"
#include "src/tools/launcher/util/launcher_util.h"

static constexpr const char* BINARY_TYPE = "binary_type";

using bazel::launcher::BashBinaryLauncher;
using bazel::launcher::BinaryLauncherBase;
using bazel::launcher::GetBinaryPathWithExtension;
using bazel::launcher::JavaBinaryLauncher;
using bazel::launcher::LaunchDataParser;
using bazel::launcher::PythonBinaryLauncher;
using bazel::launcher::die;
using std::make_unique;
using std::unique_ptr;

int wmain(int argc, wchar_t* argv[]) {
  LaunchDataParser::LaunchInfo launch_info;

  if (!LaunchDataParser::GetLaunchInfo(GetBinaryPathWithExtension(argv[0]),
                                       &launch_info)) {
    die(L"Failed to parse launch info.");
  }

  auto result = launch_info.find(BINARY_TYPE);
  if (result == launch_info.end()) {
    die(L"Cannot find key \"%hs\" from launch data.", BINARY_TYPE);
  }

  unique_ptr<BinaryLauncherBase> binary_launcher;

  if (result->second == L"Python") {
    binary_launcher =
        make_unique<PythonBinaryLauncher>(launch_info, argc, argv);
  } else if (result->second == L"Bash") {
    binary_launcher = make_unique<BashBinaryLauncher>(launch_info, argc, argv);
  } else if (result->second == L"Java") {
    binary_launcher = make_unique<JavaBinaryLauncher>(launch_info, argc, argv);
  } else {
    die(L"Unknown binary type, cannot launch anything.");
  }

  return binary_launcher->Launch();
}
