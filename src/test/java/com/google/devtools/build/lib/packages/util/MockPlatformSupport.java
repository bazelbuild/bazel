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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import javax.annotation.Nullable;

/** Mocking support for platforms and toolchains. */
public class MockPlatformSupport {

  /** Adds mocks for basic host and target platform. */
  public static void setup(MockToolsConfig mockToolsConfig, String bazelToolsPlatformsPath)
      throws IOException {
    setup(mockToolsConfig, bazelToolsPlatformsPath, null);
  }

  /** Adds mocks for basic host and target platform. */
  public static void setup(
      MockToolsConfig mockToolsConfig,
      String bazelToolsPlatformsPath,
      @Nullable String localConfigPlatformPath)
      throws IOException {
    mockToolsConfig.create(
        bazelToolsPlatformsPath + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "constraint_setting(name = 'cpu')",
        "constraint_value(",
        "    name = 'x86_32',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'x86_64',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'ppc',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'arm',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'aarch64',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 's390x',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_setting(name = 'os')",
        "constraint_value(",
        "    name = 'osx',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'ios',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'android',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'linux',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'windows',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'freebsd',",
        "    constraint_setting = ':os',",
        ")",
        "platform(",
        "   name = 'target_platform',",
        "   target_platform = True,",
        "    cpu_constraints = [",
        "        ':x86_32',",
        "        ':x86_64',",
        "        ':ppc',",
        "        ':arm',",
        "    ],",
        "    os_constraints = [",
        "        ':osx',",
        "        ':linux',",
        "        ':windows',",
        "    ],",
        ")",
        "platform(",
        "   name = 'host_platform',",
        "   host_platform = True,",
        "    cpu_constraints = [",
        "        ':x86_32',",
        "        ':x86_64',",
        "        ':ppc',",
        "        ':arm',",
        "    ],",
        "    os_constraints = [",
        "        ':osx',",
        "        ':linux',",
        "        ':windows',",
        "    ],",
        ")");
    if (localConfigPlatformPath != null) {
      // Only create these if the local config workspace exists.
      mockToolsConfig.create(
          localConfigPlatformPath + "/WORKSPACE", "workspace(name = 'local_config_platform')");
      mockToolsConfig.create(
          localConfigPlatformPath + "/BUILD",
          "package(default_visibility=['//visibility:public'])",
          "platform(name = 'host')");
    }
  }

  /** Adds a mock K8 platform. */
  public static void addMockK8Platform(MockToolsConfig mockToolsConfig, Label crosstoolLabel)
      throws Exception {
    mockToolsConfig.create(
        "mock_platform/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "constraint_setting(name = 'mock_setting')",
        "constraint_value(name = 'mock_value', constraint_setting = ':mock_setting')",
        "platform(",
        "   name = 'mock-k8-platform',",
        "   constraint_values = [':mock_value'],",
        ")",
        "toolchain(",
        "   name = 'toolchain_cc-compiler-k8',",
        "   toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "   toolchain = '"
            + crosstoolLabel.getRelativeWithRemapping("cc-compiler-k8-compiler", ImmutableMap.of())
            + "',",
        "   target_compatible_with = [':mock_value'],",
        ")");
  }
}
