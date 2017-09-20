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
package com.google.devtools.build.lib.packages.util;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;

/** Mocking support for platforms and toolchains. */
public class MockPlatformSupport {

  /** Adds mocks for basic host and target platform. */
  public static void setup(MockToolsConfig mockToolsConfig, String platformsPath)
      throws IOException {
    mockToolsConfig.create(
        platformsPath + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "platform(",
        "   name = 'target_platform',",
        "   target_platform = True,",
        ")",
        "platform(",
        "   name = 'host_platform',",
        "   host_platform = True,",
        ")");
  }

  /** Adds a mock piii platform. */
  public static void addMockPiiiPlatform(MockToolsConfig mockToolsConfig, Label crosstoolLabel)
      throws Exception {
    mockToolsConfig.create(
        "mock_platform/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "constraint_setting(name = 'mock_setting')",
        "constraint_value(name = 'mock_value', constraint_setting = ':mock_setting')",
        "platform(",
        "   name = 'mock-piii-platform',",
        "   constraint_values = [':mock_value'],",
        ")",
        "toolchain(",
        "   name = 'toolchain_cc-compiler-piii',",
        "   toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/cpp:toolchain_category',",
        "   toolchain = '" + crosstoolLabel.getRelative("cc-compiler-piii") + "',",
        "   target_compatible_with = [':mock_value'],",
        ")");
  }
}
