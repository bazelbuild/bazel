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

  /** Adds mocks for basic exec and target platform. */
  public static void setup(MockToolsConfig mockToolsConfig) throws IOException {
    setup(
        mockToolsConfig,
        TestConstants.PLATFORMS_PATH,
        TestConstants.CONSTRAINTS_PACKAGE_ROOT,
        TestConstants.CONSTRAINTS_PATH,
        TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT,
        TestConstants.LOCAL_CONFIG_PLATFORM_PATH);
  }

  /** Adds mocks for basic exec and target platform. */
  public static void setup(
      MockToolsConfig mockToolsConfig,
      String platformsPath,
      String constraintsPackageRoot,
      String constraintsPath,
      String localConfigPlatformPackageRoot,
      String localConfigPlatformPath)
      throws IOException {
    mockToolsConfig.create(
        constraintsPath + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])");
    mockToolsConfig.create(
        constraintsPath + "/cpu/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "constraint_setting(name = 'cpu')",
        "constraint_value(",
        "    name = 'all',",
        "    constraint_setting = ':cpu',",
        ")",
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
        "    name = 'arm64',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'arm64_32',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'armv7',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'armv7k',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'aarch64',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'riscv32',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 'riscv64',",
        "    constraint_setting = ':cpu',",
        ")",
        "constraint_value(",
        "    name = 's390x',",
        "    constraint_setting = ':cpu',",
        ")");
    mockToolsConfig.create(
        constraintsPath + "/os/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "constraint_setting(name = 'os')",
        "constraint_value(",
        "    name = 'osx',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'macos',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'ios',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'tvos',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'visionos',",
        "    constraint_setting = ':os',",
        ")",
        "constraint_value(",
        "    name = 'watchos',",
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
        "constraint_value(",
        "    name = 'openbsd',",
        "    constraint_setting = ':os',",
        ")");
    mockToolsConfig.create(
        platformsPath + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "platform(",
        "    name = 'default_target',",
        "    constraint_values = [",
        // Regardless of the actual machine the tests are run on, hardcode everything to a single
        // default value for simplicity.
        "        '" + constraintsPackageRoot + "cpu:x86_64',",
        "        '" + constraintsPackageRoot + "os:linux',",
        "    ],",
        ")",
        "platform(",
        "    name = 'default_host',",
        "    constraint_values = [",
        "        '" + constraintsPackageRoot + "cpu:x86_64',",
        "        '" + constraintsPackageRoot + "os:linux',",
        "    ],",
        ")");
    mockToolsConfig.create(
        localConfigPlatformPath + "/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "licenses(['notice'])",
        "platform(",
        "    name = 'host',",
        "    constraint_values = [",
        // Regardless of the actual machine the tests are run on, hardcode everything to a single
        // default value for simplicity.
        "        '" + constraintsPackageRoot + "cpu:x86_64',",
        "        '" + constraintsPackageRoot + "os:linux',",
        "    ],",
        ")",
        "platform(",
        "     name = 'piii',",
        "     constraint_values = [",
        "        '" + constraintsPackageRoot + "cpu:x86_32',",
        "        '" + constraintsPackageRoot + "os:linux',",
        "    ],",
        ")");

    mockToolsConfig.create(
        "third_party/bazel_platforms/android/BUILD",
        "licenses(['notice'])",
        "package(default_visibility=['//visibility:public'])",
        "platform(",
        "  name = 'armeabi-v7a',",
        "  parents = ['" + TestConstants.LOCAL_CONFIG_PLATFORM_PACKAGE_ROOT + ":host'],",
        "  constraint_values = [",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:android',",
        "    '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:armv7',",
        "  ],",
        ")");
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
        "   toolchain = '" + crosstoolLabel.getSamePackageLabel("cc-compiler-k8-compiler") + "',",
        "   target_compatible_with = [':mock_value'],",
        ")");
  }

  /** Adds a mock PPC platform. */
  public static void addMockPPCPlatform(MockToolsConfig mockToolsConfig, Label crosstoolLabel)
      throws Exception {
    mockToolsConfig.create(
        "mock_platform/BUILD",
        "package(default_visibility=['//visibility:public'])",
        "constraint_setting(name = 'mock_setting')",
        "constraint_value(name = 'mock_value', constraint_setting = ':mock_setting')",
        "platform(",
        "   name = 'mock-ppc-platform',",
        "   constraint_values = [':mock_value'],",
        ")",
        "toolchain(",
        "   name = 'toolchain_cc-compiler-ppc',",
        "   toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        "   toolchain = '" + crosstoolLabel.getSamePackageLabel("cc-compiler-ppc-compiler") + "',",
        "   target_compatible_with = [':mock_value'],",
        ")");
  }
}
