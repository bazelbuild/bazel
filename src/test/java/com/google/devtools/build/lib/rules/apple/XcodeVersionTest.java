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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code xcode_version} rule.
 */
@RunWith(JUnit4.class)
public final class XcodeVersionTest extends BuildViewTestCase {

  @Test
  public void testXcodeVersionCanBeReadFromStarlark() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "MyInfo = provider()",
        "def my_rule_impl(ctx):",
        "   xcode_properties = ctx.attr.xcode[apple_common.XcodeProperties]",
        "   xcode_version = xcode_properties.xcode_version",
        "   ios_version = xcode_properties.default_ios_sdk_version",
        "   watchos_version = xcode_properties.default_watchos_sdk_version",
        "   tvos_version = xcode_properties.default_tvos_sdk_version",
        "   macos_version = xcode_properties.default_macos_sdk_version",
        "   return MyInfo(",
        "       xcode_version=xcode_version,",
        "       ios_version=ios_version,",
        "       watchos_version=watchos_version,",
        "       tvos_version=tvos_version,",
        "       macos_version=macos_version,",
        "   )",
        "my_rule = rule(implementation = my_rule_impl,",
        "   attrs = {",
        "     'xcode': attr.label(),",
        "   },",
        ")");
    scratch.file(
        "examples/apple_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'my_rule')",
        "my_rule(",
        "    name = 'my_target',",
        "    xcode = ':my_xcode',",
        ")",
        "xcode_version(",
        "    name = 'my_xcode',",
        "    version = '8',",
        "    default_ios_sdk_version = '9.0',",
        "    default_watchos_sdk_version = '9.1',",
        "    default_tvos_sdk_version = '9.2',",
        "    default_macos_sdk_version = '9.3',",
        ")");

    RuleConfiguredTarget starlarkTarget =
        (RuleConfiguredTarget) getConfiguredTarget("//examples/apple_starlark:my_target");
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//examples/rule:apple_rules.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl myInfo = (StructImpl) starlarkTarget.get(key);
    assertThat((String) myInfo.getValue("xcode_version")).isEqualTo("8");
    assertThat((String) myInfo.getValue("ios_version")).isEqualTo("9.0");
    assertThat((String) myInfo.getValue("watchos_version")).isEqualTo("9.1");
    assertThat((String) myInfo.getValue("tvos_version")).isEqualTo("9.2");
    assertThat((String) myInfo.getValue("macos_version")).isEqualTo("9.3");
  }

  @Test
  public void testXcodeVersionCanBeReadFromNative() throws Exception {
    scratch.file(
        "examples/apple/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "xcode_version(",
        "    name = 'my_xcode',",
        "    version = '8',",
        "    default_ios_sdk_version = '9.0',",
        "    default_watchos_sdk_version = '9.1',",
        "    default_tvos_sdk_version = '9.2',",
        "    default_macos_sdk_version = '9.3',",
        ")");

    ConfiguredTarget nativeTarget = getConfiguredTarget("//examples/apple:my_xcode");
    XcodeVersionProperties xcodeProperties =
        nativeTarget.get(XcodeVersionProperties.STARLARK_CONSTRUCTOR);
    assertThat(xcodeProperties.getXcodeVersion().get().toString()).isEqualTo("8");
    assertThat(xcodeProperties.getDefaultIosSdkVersion().toString()).isEqualTo("9.0");
    assertThat(xcodeProperties.getDefaultWatchosSdkVersion().toString()).isEqualTo("9.1");
    assertThat(xcodeProperties.getDefaultTvosSdkVersion().toString()).isEqualTo("9.2");
    assertThat(xcodeProperties.getDefaultMacosSdkVersion().toString()).isEqualTo("9.3");
  }
}
