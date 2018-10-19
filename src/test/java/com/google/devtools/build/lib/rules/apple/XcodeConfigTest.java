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
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.syntax.Type;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the {@code xcode_config} rule.
 */
@RunWith(JUnit4.class)
public class XcodeConfigTest extends BuildViewTestCase {

  @Test
  public void testEmptyConfig_noVersionFlag() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(name = 'foo',)");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertIosSdkVersion(AppleCommandLineOptions.DEFAULT_IOS_SDK_VERSION);
  }

  @Test
  public void testDefaultVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testConfigAlias_configSetting() throws Exception {
    scratch.file("skylark/BUILD");
    scratch.file("skylark/version_retriever.bzl",
        "def _version_retriever_impl(ctx):",
        "  xcode_properties = ctx.attr.dep[apple_common.XcodeProperties]",
        "  version = xcode_properties.xcode_version",
        "  return struct(providers = [config_common.FeatureFlagInfo(value=version)])",
        "",
        "version_retriever = rule(",
        "  implementation = _version_retriever_impl,",
        "  attrs = {'dep': attr.label()},",
        ")");

    scratch.file("xcode/BUILD",
        "load('//skylark:version_retriever.bzl', 'version_retriever')",
        "version_retriever(",
        "    name = 'flag_propagator',",
        "    dep = ':alias',",
        ")",
        "",
        "xcode_config(",
        "    name = 'config',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64', ':version12'],",
        ")",
        "",
        "xcode_config_alias(",
        "    name = 'alias'",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'six', '6'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version12',",
        "    version = '12',",
        ")",
        "config_setting(name = 'xcode_5_1_2',",
        "    flag_values = {':flag_propagator': '5.1.2'})",
        "config_setting(name = 'xcode_6_4',",
        "    flag_values = {':flag_propagator': '6.4'})",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = select({",
        "       ':xcode_5_1_2': '5.1.2',",
        "       ':xcode_6_4': '6.4',",
        "       '//conditions:default': 'none'",
        "    }))");

    useConfiguration("--xcode_version_config=//xcode:config");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("5.1.2");

    useConfiguration("--xcode_version_config=//xcode:config", "--xcode_version=6.4");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("6.4");

    useConfiguration("--xcode_version_config=//xcode:config", "--xcode_version=6");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("6.4");

    useConfiguration("--xcode_version_config=//xcode:config", "--xcode_version=12");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("none");
  }

  @Test
  public void testDefaultVersion_configSetting() throws Exception {
    scratch.file("skylark/BUILD");
    scratch.file("skylark/version_retriever.bzl",
        "def _version_retriever_impl(ctx):",
        "  xcode_properties = ctx.attr.dep[apple_common.XcodeProperties]",
        "  version = xcode_properties.xcode_version",
        "  return struct(providers = [config_common.FeatureFlagInfo(value=version)])",
        "",
        "version_retriever = rule(",
        "  implementation = _version_retriever_impl,",
        "  attrs = {'dep': attr.label()},",
        ")");

    scratch.file("xcode/BUILD",
        "load('//skylark:version_retriever.bzl', 'version_retriever')",
        "version_retriever(",
        "    name = 'flag_propagator',",
        "    dep = ':alias',",
        ")",
        "xcode_config_alias(",
        "    name = 'alias'",
        ")",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        ")",
        "config_setting(name = 'xcode_5_1_2',",
        "    flag_values = {':flag_propagator': '5.1.2'})",
        "config_setting(name = 'xcode_6_4',",
        "    flag_values = {':flag_propagator': '6.4'})",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = select({",
        "       ':xcode_5_1_2': '5.1.2',",
        "       ':xcode_6_4': '6.4',",
        "       '//conditions:default': 'none'",
        "    }))");

    useConfiguration("--xcode_version_config=//xcode:foo");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("5.1.2");

    useConfiguration("--xcode_version_config=//xcode:foo", "--xcode_version=6.4");
    assertThat(getMapper("//xcode:gen").get("cmd", Type.STRING)).isEqualTo("6.4");
  }

  @Test
  public void testRequiresDefined_validVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    require_defined_version = 1,",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")");
    useConfiguration("--xcode_version=5.1.2", "--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testRequiresDefined_validAlias_dottedVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    require_defined_version = 1,",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")");
    useConfiguration("--xcode_version=5", "--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testRequiresDefined_validAlias_nonNumerical() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    require_defined_version = 1,",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['valid_version'],",
        ")");
    useConfiguration("--xcode_version=valid_version", "--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testRequiresDefined_validDefault() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testInvalidXcodeSpecified() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    require_defined_version = 1,",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "xcode_version(",
        "    name = 'version84',",
        "    version = '8.4',",
        ")");
    useConfiguration("--xcode_version=6");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("--xcode_version=6 specified, but '6' is not an available Xcode version. "
        + "available versions: [5.1.2, 8.4]. If you believe you have '6' installed");
  }

  @Test
  public void testRequiresDefault() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    require_defined_version = 1,",
        "    versions = [':version512'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("default version must be specified");
  }

  @Test
  public void testDuplicateAliases_definedVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    versions = [':version512', ':version5'],",
        "    default = ':version512'",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version5',",
        "    version = '5',",
        "    aliases = ['5', '5.0', 'foo'],",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent(
        "'5' is registered to multiple labels (//xcode:version512, //xcode:version5)");
  }

  @Test
  public void testVersionAliasedToItself() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1', '5.1.2'],",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
  }

  @Test
  public void testDuplicateVersionNumbers() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    versions = [':version512', ':version5'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version5',",
        "    version = '5.1.2',",
        "    aliases = ['foo'],",
        ")");
    useConfiguration("--xcode_version=5");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent(
        "'5.1.2' is registered to multiple labels (//xcode:version512, //xcode:version5)");
  }

  @Test
  public void testVersionConflictsWithAlias() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    versions = [':version512', ':version5'],",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5',",
        "    aliases = ['5.1'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version5',",
        "    version = '5.1.3',",
        "    aliases = ['5'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent(
        "'5' is registered to multiple labels (//xcode:version512, //xcode:version5)");
  }

  @Test
  public void testDefaultIosSdkVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        "    default_ios_sdk_version = '7.1'",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        "    default_ios_sdk_version = '43.0'",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
    assertIosSdkVersion("7.1");
  }

  @Test
  public void testDefaultSdkVersions() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        "    default_ios_sdk_version = '101',",
        "    default_watchos_sdk_version = '102',",
        "    default_tvos_sdk_version = '103',",
        "    default_macos_sdk_version = '104',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        "    default_ios_sdk_version = '43.0'",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");

    assertXcodeVersion("5.1.2");
    ImmutableMap<ApplePlatform, String> platformToVersion =
        ImmutableMap.<ApplePlatform, String>builder()
            .put(ApplePlatform.IOS_SIMULATOR, "101")
            .put(ApplePlatform.WATCHOS_SIMULATOR, "102")
            .put(ApplePlatform.TVOS_SIMULATOR, "103")
            .put(ApplePlatform.MACOS, "104")
            .build();
    for (ApplePlatform platform : platformToVersion.keySet()) {
      DottedVersion version = DottedVersion.fromString(platformToVersion.get(platform));
      assertThat(getSdkVersionForPlatform(platform)).isEqualTo(version);
      assertThat(getMinimumOsVersionForPlatform(platform)).isEqualTo(version);
    }
  }

  @Test
  public void testDefaultSdkVersions_selectedXcode() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        "    default_ios_sdk_version = '7.1'",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        "    default_ios_sdk_version = '43',",
        "    default_watchos_sdk_version = '44',",
        "    default_tvos_sdk_version = '45',",
        "    default_macos_sdk_version = '46',",
        ")");
    useConfiguration("--xcode_version=6", "--xcode_version_config=//xcode:foo");

    assertXcodeVersion("6.4");
    ImmutableMap<ApplePlatform, String> platformToVersion =
        ImmutableMap.<ApplePlatform, String>builder()
            .put(ApplePlatform.IOS_SIMULATOR, "43")
            .put(ApplePlatform.WATCHOS_SIMULATOR, "44")
            .put(ApplePlatform.TVOS_SIMULATOR, "45")
            .put(ApplePlatform.MACOS, "46")
            .build();
    for (ApplePlatform platform : platformToVersion.keySet()) {
      DottedVersion version = DottedVersion.fromString(platformToVersion.get(platform));
      assertThat(getSdkVersionForPlatform(platform)).isEqualTo(version);
      assertThat(getMinimumOsVersionForPlatform(platform)).isEqualTo(version);
    }
  }

  @Test
  public void testOverrideDefaultSdkVersions() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512', ':version64'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        "    default_ios_sdk_version = '7.1'",
        ")",
        "",
        "xcode_version(",
        "    name = 'version64',",
        "    version = '6.4',",
        "    aliases = ['6.0', 'foo', '6'],",
        "    default_ios_sdk_version = '101',",
        "    default_watchos_sdk_version = '102',",
        "    default_tvos_sdk_version = '103',",
        "    default_macos_sdk_version = '104',",
        ")");
    useConfiguration("--xcode_version=6", "--xcode_version_config=//xcode:foo",
        "--ios_sdk_version=15.3", "--watchos_sdk_version=15.4",
        "--tvos_sdk_version=15.5", "--macos_sdk_version=15.6");

    assertXcodeVersion("6.4");
    ImmutableMap<ApplePlatform, String> platformToVersion =
        ImmutableMap.<ApplePlatform, String>builder()
            .put(ApplePlatform.IOS_SIMULATOR, "15.3")
            .put(ApplePlatform.WATCHOS_SIMULATOR, "15.4")
            .put(ApplePlatform.TVOS_SIMULATOR, "15.5")
            .put(ApplePlatform.MACOS, "15.6")
            .build();
    for (ApplePlatform platform : platformToVersion.keySet()) {
      DottedVersion version = DottedVersion.fromString(platformToVersion.get(platform));
      assertThat(getSdkVersionForPlatform(platform)).isEqualTo(version);
      assertThat(getMinimumOsVersionForPlatform(platform)).isEqualTo(version);
    }
  }

  @Test
  public void testXcodeVersionFromSkylarkByAlias() throws Exception {
    scratch.file("x/BUILD",
        "load('//x:r.bzl', 'r')",
        "xcode_config_alias(name='a')",
        "xcode_config(name='c', default=':v', versions=[':v'])",
        "xcode_version(",
        "    name = 'v',",
        "    version = '0.0',",
        "    default_ios_sdk_version = '1.0',",
        "    default_tvos_sdk_version = '2.0',",
        "    default_macos_sdk_version = '3.0',",
        "    default_watchos_sdk_version = '4.0',",
        ")",
        "r(name='r')");
    scratch.file("x/r.bzl",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode[apple_common.XcodeVersionConfig]",
        "  ios = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)",
        "  tvos = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.tvos)",
        "  return struct(",
        "    xcode = conf.xcode_version(),",
        "    ios_sdk = conf.sdk_version_for_platform(ios),",
        "    tvos_sdk = conf.sdk_version_for_platform(tvos),",
        "    macos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.macos),",
        "    watchos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.watchos),",
        "  )",
        "r = rule(implementation = _impl,",
        "    attrs = { '_xcode': attr.label(default = Label('//x:a'))},",
        "    fragments = ['apple'],",
        ")");

    useConfiguration(
        "--xcode_version_config=//x:c",
        "--tvos_sdk_version=2.5",
        "--watchos_minimum_os=4.5");
    ConfiguredTarget r = getConfiguredTarget("//x:r");
    assertThat(r.get("xcode").toString()).isEqualTo("0.0");
    assertThat(r.get("ios_sdk").toString()).isEqualTo("1.0");
    assertThat(r.get("tvos_sdk").toString()).isEqualTo("2.5");
    assertThat(r.get("macos_min").toString()).isEqualTo("3.0");
    assertThat(r.get("watchos_min").toString()).isEqualTo("4.5");
  }

  @Test
  public void testDefaultWithoutVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1', '5.1.2'],",
        ")");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent(
        "default label '//xcode:version512' must be contained in versions attribute");
  }

  @Test
  public void testVersionDoesNotContainDefault() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version6'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1', '5.1.2'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version6',",
        "    version = '6.0',",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("must be contained in versions attribute");
  }

  @Test
  public void testInvalidBitcodeVersion() throws Exception {
    scratch.file("xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    versions = [':version512'],",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        ")");

    useConfiguration("--apple_bitcode=embedded", "--apple_split_cpu=arm64");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("apple_bitcode mode 'embedded' is unsupported");
  }

  // Verifies that the --xcode_version_config configuration value can be accessed via the
  // configuration_field() skylark method and used in a skylark rule.
  @Test
  public void testConfigurationFieldForRule() throws Exception {
    scratch.file("x/provider_grabber.bzl",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode_dep[apple_common.XcodeVersionConfig]",
        "  return struct(",
        "    providers = [conf],",
        "  )",

        "provider_grabber = rule(implementation = _impl,",
        "    attrs = { '_xcode_dep': attr.label(",
        "        default = configuration_field(",
        "            fragment = 'apple', name = 'xcode_config_label')),",
        "    },",
        "    fragments = ['apple'],",
        ")");

    scratch.file("x/BUILD",
        "load('//x:provider_grabber.bzl', 'provider_grabber')",
        "xcode_config(name='config1', default=':version1', versions=[':version1'])",
        "xcode_config(name='config2', default=':version2', versions=[':version2'])",
        "xcode_version(name = 'version1', version = '1.0')",
        "xcode_version(name = 'version2', version = '2.0')",
        "provider_grabber(name='provider_grabber')");

    useConfiguration("--xcode_version_config=//x:config1");
    assertXcodeVersion("1.0", "//x:provider_grabber");

    useConfiguration("--xcode_version_config=//x:config2");
    assertXcodeVersion("2.0", "//x:provider_grabber");
  }

  // Verifies that the --xcode_version_config configuration value can be accessed via the
  // configuration_field() skylark method and used in a skylark aspect.
  @Test
  public void testConfigurationFieldForAspect() throws Exception {
    scratch.file("x/provider_grabber.bzl",
        "def _aspect_impl(target, ctx):",
        "  conf = ctx.attr._xcode_dep[apple_common.XcodeVersionConfig]",
        "  return struct(",
        "    providers = [conf],",
        "  )",
        "",
        "MyAspect = aspect(implementation = _aspect_impl,",
        "    attrs = { '_xcode_dep': attr.label(",
        "        default = configuration_field(",
        "            fragment = 'apple', name = 'xcode_config_label')),",
        "    },",
        "    fragments = ['apple'],",
        ")",
        "",
        "def _rule_impl(ctx):",
        "  conf = ctx.attr.dep[0][apple_common.XcodeVersionConfig]",
        "  return struct(",
        "    providers = [conf],",
        "  )",
        "",
        "provider_grabber = rule(implementation = _rule_impl,",
        "    attrs = { 'dep' : ",
        "             attr.label_list(mandatory=True, allow_files=True, aspects = [MyAspect]) },",
        ")");

    scratch.file("x/BUILD",
        "load('//x:provider_grabber.bzl', 'provider_grabber')",
        "xcode_config(name='config1', default=':version1', versions=[':version1'])",
        "xcode_config(name='config2', default=':version2', versions=[':version2'])",
        "xcode_version(name = 'version1', version = '1.0')",
        "xcode_version(name = 'version2', version = '2.0')",
        "java_library(",
        "     name = 'fake_lib',",
        ")",
        "provider_grabber(",
        "     name = 'provider_grabber',",
        "     dep = [':fake_lib'],",
        ")");

    useConfiguration("--xcode_version_config=//x:config1");
    assertXcodeVersion("1.0", "//x:provider_grabber");

    useConfiguration("--xcode_version_config=//x:config2");
    assertXcodeVersion("2.0", "//x:provider_grabber");
  }

  private DottedVersion getSdkVersionForPlatform(ApplePlatform platform) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget("//xcode:foo");
    XcodeConfigProvider provider = xcodeConfig.get(XcodeConfigProvider.PROVIDER);
    return provider.getSdkVersionForPlatform(platform);
  }

  private DottedVersion getMinimumOsVersionForPlatform(ApplePlatform platform) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget("//xcode:foo");
    XcodeConfigProvider provider = xcodeConfig.get(XcodeConfigProvider.PROVIDER);
    return provider.getMinimumOsForPlatformType(platform.getType());
  }

  private void assertXcodeVersion(String version) throws Exception {
    assertXcodeVersion(version, "//xcode:foo");
  }

  private void assertXcodeVersion(String version, String providerTargetLabel) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget(providerTargetLabel);
    XcodeConfigProvider provider = xcodeConfig.get(XcodeConfigProvider.PROVIDER);
    assertThat(provider.getXcodeVersion()).isEqualTo(DottedVersion.fromString(version));
  }

  private void assertIosSdkVersion(String version) throws Exception {
    assertThat(getSdkVersionForPlatform(ApplePlatform.IOS_SIMULATOR))
        .isEqualTo(DottedVersion.fromString(version));
  }

  /**
   * Returns a ConfiguredAttributeMapper bound to the given rule with the target configuration.
   */
  private ConfiguredAttributeMapper getMapper(String label) throws Exception {
    ConfiguredTargetAndData ctad = getConfiguredTargetAndData(label);
    return getMapperFromConfiguredTargetAndTarget(ctad);
  }
}
