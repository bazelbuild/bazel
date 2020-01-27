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

package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the {@code available_xcodes} rule. */
@RunWith(JUnit4.class)
public final class AvailableXcodesTest extends BuildViewTestCase {
  @Test
  public void testXcodeVersionCanBeReadFromNative() throws Exception {
    scratch.file(
        "examples/apple/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "available_xcodes(",
        "    name = 'my_xcodes',",
        "    default = ':xcode_8',",
        "    versions = [':xcode_8', ':xcode_9'],",
        ")",
        "xcode_version(",
        "    name = 'xcode_8',",
        "    version = '8',",
        "    default_ios_sdk_version = '9.0',",
        "    default_watchos_sdk_version = '9.1',",
        "    default_tvos_sdk_version = '9.2',",
        "    default_macos_sdk_version = '9.3',",
        ")",
        "xcode_version(",
        "    name = 'xcode_9',",
        "    version = '9',",
        "    default_ios_sdk_version = '10.0',",
        "    default_watchos_sdk_version = '10.1',",
        "    default_tvos_sdk_version = '10.2',",
        "    default_macos_sdk_version = '10.3',",
        ")");

    ConfiguredTarget nativeTarget = getConfiguredTarget("//examples/apple:my_xcodes");
    AvailableXcodesInfo availableXcodesInfo = nativeTarget.get(AvailableXcodesInfo.PROVIDER);
    ConfiguredTarget version8 = getConfiguredTarget("//examples/apple:xcode_8");
    XcodeVersionProperties version8properties =
        version8.get(XcodeVersionProperties.SKYLARK_CONSTRUCTOR);
    ConfiguredTarget version9 = getConfiguredTarget("//examples/apple:xcode_9");
    XcodeVersionProperties version9properties =
        version9.get(XcodeVersionProperties.SKYLARK_CONSTRUCTOR);
    assertThat(availableXcodesInfo.getAvailableVersions()).hasSize(2);
    assertThat(
            Iterables.get(availableXcodesInfo.getAvailableVersions(), 0)
                .getXcodeVersionProperties())
        .isEqualTo(version8properties);
    assertThat(
            Iterables.get(availableXcodesInfo.getAvailableVersions(), 1)
                .getXcodeVersionProperties())
        .isEqualTo(version9properties);
    assertThat(availableXcodesInfo.getDefaultVersion().getVersion().toString()).isEqualTo("8");
    assertThat(availableXcodesInfo.getDefaultVersion().getXcodeVersionProperties())
        .isEqualTo(version8properties);
  }

  @Test
  public void testXcodeVersionRequiresDefault() throws Exception {
    scratch.file(
        "examples/apple/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "available_xcodes(",
        "    name = 'my_xcodes',",
        "    versions = [':my_xcode'],",
        ")",
        "xcode_version(",
        "    name = 'my_xcode',",
        "    version = '8',",
        "    default_ios_sdk_version = '9.0',",
        "    default_watchos_sdk_version = '9.1',",
        "    default_tvos_sdk_version = '9.2',",
        "    default_macos_sdk_version = '9.3',",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//examples/apple:my_xcodes");
    assertContainsEvent(
        "missing value for mandatory attribute 'default' in 'available_xcodes' rule");
  }
}
