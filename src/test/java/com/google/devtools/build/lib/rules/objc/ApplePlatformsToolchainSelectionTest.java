// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.util.MockObjcSupport;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for the use of Apple platforms and the macOS crosstool. */
@RunWith(JUnit4.class)
public class ApplePlatformsToolchainSelectionTest extends ObjcRuleTestCase {

  @Override
  protected boolean platformBasedToolchains() {
    return true;
  }

  private static final ConstraintSettingInfo CPU_CONSTRAINT =
      ConstraintSettingInfo.create(
          Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:cpu"));

  private static final ConstraintSettingInfo OS_CONSTRAINT =
      ConstraintSettingInfo.create(
          Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:os"));

  private static final ToolchainTypeInfo CPP_TOOLCHAIN_TYPE =
      ToolchainTypeInfo.create(
          Label.parseCanonicalUnchecked(
              TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type"));

  private static final String EXTRA_SDK_TOOLCHAINS_FLAG =
      "--extra_toolchains=//tools/build_defs/apple/toolchains:all";

  @Test
  public void testMacOsToolchainSetup() throws Exception {
    // Verify the macOS toolchain and its associated cpp toolchain.
    ConfiguredTarget darwinToolchain =
        getConfiguredTarget("//tools/build_defs/apple/toolchains:darwin_x86_64_any");
    assertThat(darwinToolchain).isNotNull();
    DeclaredToolchainInfo darwinToolchainInfo =
        PlatformProviderUtils.declaredToolchainInfo(darwinToolchain);
    assertThat(darwinToolchainInfo).isNotNull();
    assertThat(darwinToolchainInfo.toolchainLabel())
        .isEqualTo(
            Label.parseCanonicalUnchecked(
                "//" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL_DIR + ":cc-compiler-darwin_x86_64"));
    assertThat(darwinToolchainInfo.toolchainType()).isEqualTo(CPP_TOOLCHAIN_TYPE);

    // Verify the macOS platform.
    ConfiguredTarget darwinPlatform =
        getConfiguredTarget(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:darwin_x86_64");
    PlatformInfo darwinPlatformInfo = PlatformProviderUtils.platform(darwinPlatform);
    assertThat(darwinPlatformInfo).isNotNull();
  }

  @Test
  public void testIosDeviceToolchainSetup() throws Exception {
    // Verify the iOS 64 bit device toolchain and its associated cpp toolchain.
    ConfiguredTarget iosDeviceToolchain =
        getConfiguredTarget("//tools/build_defs/apple/toolchains:ios_arm64_any");
    assertThat(iosDeviceToolchain).isNotNull();
    DeclaredToolchainInfo iosDeviceToolchainInfo =
        PlatformProviderUtils.declaredToolchainInfo(iosDeviceToolchain);
    assertThat(iosDeviceToolchainInfo).isNotNull();
    assertThat(iosDeviceToolchainInfo.toolchainLabel())
        .isEqualTo(
            Label.parseCanonicalUnchecked(
                "//" + MockObjcSupport.DEFAULT_OSX_CROSSTOOL_DIR + ":cc-compiler-ios_arm64"));
    assertThat(iosDeviceToolchainInfo.toolchainType()).isEqualTo(CPP_TOOLCHAIN_TYPE);

    // Verify the iOS 64 bit device platform.
    ConfiguredTarget iosDevicePlatform =
        getConfiguredTarget(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:ios_arm64");
    PlatformInfo iosDevicePlatformInfo = PlatformProviderUtils.platform(iosDevicePlatform);
    assertThat(iosDevicePlatformInfo).isNotNull();
  }

  @Test
  public void testToolchainSelectionMacOs() throws Exception {
    // TODO(b/210057756): Modify AppleConfiguration such that apple_platform_type is not needed for
    // platforms.
    useConfiguration(
        EXTRA_SDK_TOOLCHAINS_FLAG,
        "--apple_platform_type=macos",
        "--apple_platforms=" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:darwin_x86_64",
        "--platforms=" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:darwin_x86_64");
    createLibraryTargetWriter("//a:lib").write();

    BuildConfigurationValue config = getAppleCrosstoolConfiguration();
    BuildOptions crosstoolBuildOptions = config.getOptions();
    assertThat(crosstoolBuildOptions.contains(PlatformOptions.class)).isNotNull();
    List<Label> platforms = crosstoolBuildOptions.get(PlatformOptions.class).platforms;
    assertThat(platforms)
        .containsExactly(
            Label.parseCanonicalUnchecked(
                TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:darwin_x86_64"));

    ConfiguredTarget darwinPlatform = getConfiguredTarget(platforms.get(0).toString());
    PlatformInfo darwinPlatformInfo = PlatformProviderUtils.platform(darwinPlatform);

    // Verify the OS and CPU constraints.
    ConstraintValueInfo expectedCpuConstraint =
        ConstraintValueInfo.create(
            CPU_CONSTRAINT,
            Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:x86_64"));
    assertThat(darwinPlatformInfo.constraints().has(CPU_CONSTRAINT)).isTrue();
    assertThat(darwinPlatformInfo.constraints().get(CPU_CONSTRAINT))
        .isEqualTo(expectedCpuConstraint);

    ConstraintValueInfo expectedOsConstraint =
        ConstraintValueInfo.create(
            OS_CONSTRAINT,
            Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx"));
    assertThat(darwinPlatformInfo.constraints().has(OS_CONSTRAINT)).isTrue();
    assertThat(darwinPlatformInfo.constraints().get(OS_CONSTRAINT))
        .isEqualTo(expectedOsConstraint);
  }

  @Test
  public void testToolchainSelectionIosDevice() throws Exception {
    // TODO(b/210057756): Modify AppleConfiguration such that apple_platform_type is not needed for
    // platforms.
    useConfiguration(
        EXTRA_SDK_TOOLCHAINS_FLAG,
        "--apple_platform_type=ios",
        "--apple_platforms=" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:ios_arm64",
        "--platforms=" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:ios_arm64");
    createLibraryTargetWriter("//a:lib").write();

    BuildConfigurationValue config = getAppleCrosstoolConfiguration();
    BuildOptions crosstoolBuildOptions = config.getOptions();
    assertThat(crosstoolBuildOptions.contains(PlatformOptions.class)).isNotNull();
    List<Label> platforms = crosstoolBuildOptions.get(PlatformOptions.class).platforms;
    assertThat(platforms)
        .containsExactly(
            Label.parseCanonicalUnchecked(
                TestConstants.CONSTRAINTS_PACKAGE_ROOT + "apple:ios_arm64"));

    ConfiguredTarget iosPlatform = getConfiguredTarget(platforms.get(0).toString());
    PlatformInfo iosPlatformInfo = PlatformProviderUtils.platform(iosPlatform);

    // Verify the OS and CPU constraints.
    ConstraintValueInfo expectedCpuConstraint =
        ConstraintValueInfo.create(
            CPU_CONSTRAINT,
            Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:arm64"));
    assertThat(iosPlatformInfo.constraints().has(CPU_CONSTRAINT)).isTrue();
    assertThat(iosPlatformInfo.constraints().get(CPU_CONSTRAINT))
        .isEqualTo(expectedCpuConstraint);

    ConstraintValueInfo expectedOsConstraint =
        ConstraintValueInfo.create(
            OS_CONSTRAINT,
            Label.parseCanonicalUnchecked(TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:ios"));
    assertThat(iosPlatformInfo.constraints().has(OS_CONSTRAINT)).isTrue();
    assertThat(iosPlatformInfo.constraints().get(OS_CONSTRAINT))
        .isEqualTo(expectedOsConstraint);
  }
}
