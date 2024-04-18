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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.Sequence;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the {@code available_xcodes} rule. */
@RunWith(JUnit4.class)
public final class AvailableXcodesTest extends BuildViewTestCase {
  private static final Provider.Key AVAILABLE_XCODES_PROVIDER_KEY =
      new StarlarkProvider.Key(
          Label.parseCanonicalUnchecked("@_builtins//:common/xcode/providers.bzl"),
          "AvailableXcodesInfo");

  private static final Provider.Key XCODE_VERSION_PROPERTIES_PROVIDER_KEY =
      new StarlarkProvider.Key(
          Label.parseCanonicalUnchecked("@_builtins//:common/xcode/providers.bzl"),
          "XcodeVersionPropertiesInfo");

  @Test
  public void testXcodeVersionCanBeReadFromNative() throws Exception {
    scratch.file(
        "examples/apple/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        available_xcodes(
            name = "my_xcodes",
            default = ":xcode_8",
            versions = [
                ":xcode_8",
                ":xcode_9",
            ],
        )

        xcode_version(
            name = "xcode_8",
            default_ios_sdk_version = "9.0",
            default_macos_sdk_version = "9.3",
            default_tvos_sdk_version = "9.2",
            default_watchos_sdk_version = "9.1",
            version = "8",
        )

        xcode_version(
            name = "xcode_9",
            default_ios_sdk_version = "10.0",
            default_macos_sdk_version = "10.3",
            default_tvos_sdk_version = "10.2",
            default_watchos_sdk_version = "10.1",
            version = "9",
        )
        """);

    ConfiguredTarget nativeTarget = getConfiguredTarget("//examples/apple:my_xcodes");
    StructImpl availableXcodesInfo = (StructImpl) nativeTarget.get(AVAILABLE_XCODES_PROVIDER_KEY);
    ConfiguredTarget version8 = getConfiguredTarget("//examples/apple:xcode_8");
    StructImpl version8properties =
        (StructImpl) version8.get(XCODE_VERSION_PROPERTIES_PROVIDER_KEY);
    ConfiguredTarget version9 = getConfiguredTarget("//examples/apple:xcode_9");
    StructImpl version9properties =
        (StructImpl) version9.get(XCODE_VERSION_PROPERTIES_PROVIDER_KEY);
    Sequence<StructImpl> availableVersions =
        Sequence.cast(
            availableXcodesInfo.getValue("available_versions"),
            StructImpl.class,
            "available_versions");
    assertThat(availableVersions).hasSize(2);
    assertThat(availableVersions.get(0).getValue("xcode_version_properties"))
        .isEqualTo(version8properties);
    assertThat(availableVersions.get(1).getValue("xcode_version_properties"))
        .isEqualTo(version9properties);
    StructImpl defaultVersion = availableXcodesInfo.getValue("default_version", StructImpl.class);
    assertThat(defaultVersion.getValue("xcode_version_properties")).isEqualTo(version8properties);
  }

  @Test
  public void testXcodeVersionRequiresDefault() throws Exception {
    scratch.file(
        "examples/apple/BUILD",
        """
        package(default_visibility = ["//visibility:public"])

        available_xcodes(
            name = "my_xcodes",
            versions = [":my_xcode"],
        )

        xcode_version(
            name = "my_xcode",
            default_ios_sdk_version = "9.0",
            default_macos_sdk_version = "9.3",
            default_tvos_sdk_version = "9.2",
            default_watchos_sdk_version = "9.1",
            version = "8",
        )
        """);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//examples/apple:my_xcodes");
    assertContainsEvent(
        "missing value for mandatory attribute 'default' in 'available_xcodes' rule");
  }
}
