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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code ios_device} rule.
 */
@RunWith(JUnit4.class)
public class IosDeviceTest extends BuildViewTestCase {

  @Test
  public void testIosVersion_specified() throws Exception {
    scratch.file("test/BUILD",
        "ios_device(name = 'foo', ios_version = '42.0', type = 'IPHONE_6',)");

    assertIosVersion("//test:foo", "42.0");
  }

  @Test
  public void testIosVersion_default() throws Exception {
    scratch.file("test/BUILD",
        "ios_device(name = 'foo', type = 'IPHONE_6',)");

    assertIosVersion("//test:foo", AppleCommandLineOptions.DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testIosVersion_flagValue() throws Exception {
    scratch.file("test/BUILD",
        "ios_device(name = 'foo', type = 'IPHONE_6',)");
    useConfiguration("--ios_sdk_version=42.3");

    assertIosVersion("//test:foo", "42.3");
  }

  /**
   * Tests that if {@code ios_device} specifies an xcode version that does not specify a {@code
   * default_ios_sdk_version}, the ios sdk version of the device defaults to the default value of
   * {@code default_ios_sdk_version} instead of the build configuration value. This is a confusing
   * (perhaps convoluted) corner case.
   */
  @Test
  public void testXcodeVersion_noIosVersion() throws Exception {
    scratch.file("test/BUILD",
        "xcode_version(name = 'my_xcode', version = '15.2')",
        "ios_device(name = 'foo', type = 'IPHONE_6', xcode = ':my_xcode')");
    useConfiguration("--xcode_version=7.3", "--ios_sdk_version=42.3");

    assertXcodeVersion("//test:foo", "15.2");
    assertIosVersion("//test:foo", XcodeVersionProperties.DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testXcodeVersion_withIosVersion() throws Exception {
    scratch.file("test/BUILD",
        "xcode_version(name = 'my_xcode', version = '15.2', default_ios_sdk_version='17.8')",
        "ios_device(name = 'foo', type = 'IPHONE_6', xcode = ':my_xcode')");
    useConfiguration("--xcode_version=7.3", "--ios_sdk_version=42.3");

    assertXcodeVersion("//test:foo", "15.2");
    assertIosVersion("//test:foo", "17.8");
  }

  @Test
  public void testXcodeVersion_iosVersionOverride() throws Exception {
    scratch.file("test/BUILD",
        "xcode_version(name = 'my_xcode', version = '15.2', default_ios_sdk_version='17.8')",
        "ios_device(name = 'foo', type = 'IPHONE_6', ios_version='98.7', xcode = ':my_xcode')");
    useConfiguration("--xcode_version=7.3", "--ios_sdk_version=42.3");

    assertXcodeVersion("//test:foo", "15.2");
    assertIosVersion("//test:foo", "98.7");
  }

  @Test
  public void testType() throws Exception {
    scratch.file("test/BUILD",
        "ios_device(name = 'foo', type = 'IPHONE_6',)");

    assertThat(view.hasErrors(getConfiguredTarget("//test:foo"))).isFalse();

    ConfiguredTarget target = getConfiguredTarget("//test:foo");
    IosDeviceProvider provider = target.get(IosDeviceProvider.SKYLARK_CONSTRUCTOR);
    assertThat(provider.getType()).isEqualTo("IPHONE_6");
  }

  @Test
  public void testIosDeviceAttributesCanBeReadFromSkylark() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "def my_rule_impl(ctx):",
        "   ios_device_attr = ctx.attr.ios_device[apple_common.IosDevice]",
        "   return struct(",
        "      xcode_version=ios_device_attr.xcode_version,",
        "      ios_version=ios_device_attr.ios_version,",
        "      type=ios_device_attr.type",
        "   )",
        "my_rule = rule(implementation = my_rule_impl,",
        "   attrs = {",
        "     'ios_device': attr.label(),",
        "   },",
        ")");
    scratch.file(
        "examples/apple_skylark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('/examples/rule/apple_rules', 'my_rule')",
        "my_rule(",
        "    name = 'my_target',",
        "    ios_device = ':my_device',",
        ")",
        "ios_device(",
        "   name = 'my_device',",
        "   type = 'IPHONE_8',",
        "   xcode = ':my_xcode',",
        "   ios_version='98.7'",
        ")",
        "xcode_version(",
        "   name = 'my_xcode',",
        "   version = '15.2'",
        ")");

    RuleConfiguredTarget skylarkTarget =
        (RuleConfiguredTarget) getConfiguredTarget("//examples/apple_skylark:my_target");
    assertThat((String) skylarkTarget.get("xcode_version")).isEqualTo("15.2");
    assertThat((String) skylarkTarget.get("type")).isEqualTo("IPHONE_8");
    assertThat((String) skylarkTarget.get("ios_version")).isEqualTo("98.7");
  }

  private void assertXcodeVersion(String label, String version) throws Exception {
    assertThat(view.hasErrors(getConfiguredTarget(label))).isFalse();

    ConfiguredTarget target = getConfiguredTarget(label);
    IosDeviceProvider provider = target.get(IosDeviceProvider.SKYLARK_CONSTRUCTOR);
    assertThat(provider.getXcodeVersion()).isEqualTo(DottedVersion.fromString(version));
  }

  private void assertIosVersion(String label, String version) throws Exception {
    assertThat(view.hasErrors(getConfiguredTarget(label))).isFalse();

    ConfiguredTarget target = getConfiguredTarget(label);
    IosDeviceProvider provider = target.get(IosDeviceProvider.SKYLARK_CONSTRUCTOR);
    assertThat(provider.getIosVersion()).isEqualTo(DottedVersion.fromString(version));
  }
}
