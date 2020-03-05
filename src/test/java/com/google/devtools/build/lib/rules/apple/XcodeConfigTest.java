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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Map;
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
  }

  @Test
  public void testMutualAndExplicitXcodesThrows() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512',],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("'versions' may not be set if '[local,remote]_versions' is set");
  }

  @Test
  public void testMutualAndDefaultThrows() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    default = ':version512',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512',],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("'default' may not be set if '[local,remote]_versions' is set.");
  }

  @Test
  public void testNoLocalXcodesThrows() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        ")",
        "",
        "xcode_version(",
        "    name = 'version512',",
        "    version = '5.1.2',",
        "    aliases = ['5', '5.1'],",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512',],",
        "    default = ':version512',",
        ")");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("if 'remote_versions' are set, you must also set 'local_versions'");
  }

  @Test
  public void testAcceptFlagForMutuallyAvailable() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version=8.4", "--xcode_version_config=//xcode:foo");
    assertXcodeVersion("8.4");
    assertAvailability(XcodeConfigInfo.Availability.BOTH);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertNoEvents();
  }

  @Test
  public void testPreferFlagOverMutuallyAvailable() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version=5", "--xcode_version_config=//xcode:foo");
    assertXcodeVersion("5.1.2");
    assertAvailability(XcodeConfigInfo.Availability.REMOTE);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN,
            ExecutionRequirements.NO_LOCAL,
            ExecutionRequirements.REQUIREMENTS_SET));

    assertContainsEvent(
        "--xcode_version=5 specified, but it is not available locally. Your build"
            + " will fail if any actions require a local Xcode.");
  }

  @Test
  public void testWarnWithExplicitLocalOnlyVersion() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version=8.4", "--xcode_version_config=//xcode:foo");
    assertXcodeVersion("8.4");
    assertAvailability(XcodeConfigInfo.Availability.LOCAL);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN,
            ExecutionRequirements.NO_REMOTE,
            ExecutionRequirements.REQUIREMENTS_SET));

    assertContainsEvent(
        "--xcode_version=8.4 specified, but it is not available remotely. Actions"
            + " requiring Xcode will be run locally, which could make your build"
            + " slower.");
  }

  @Test
  public void testPreferLocalDefaultIfNoMutualNoFlag() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512',],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");
    assertXcodeVersion("8.4");
    assertAvailability(XcodeConfigInfo.Availability.LOCAL);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN,
            ExecutionRequirements.NO_REMOTE,
            ExecutionRequirements.REQUIREMENTS_SET));

    assertContainsEvent(
        "Using a local Xcode version, '8.4', since there are no"
            + " remotely available Xcodes on this machine. Consider downloading one of the"
            + " remotely available Xcode versions (5.1.2)");
  }

  @Test
  public void testChooseNewestMutualXcode() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "xcode_version(",
        "    name = 'version92',",
        "    version = '9.2',",
        ")",
        "xcode_version(",
        "    name = 'version9',",
        "    version = '9',",
        ")",
        "xcode_version(",
        "    name = 'version10',",
        "    version = '10',",
        "    aliases = [ '10.0'],",
        ")",
        "xcode_version(",
        "    name = 'other_version10',",
        "    version = '10.0',",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version84', ':version92', ':version9', ':version10'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84', ':other_version10', ':version92', ':version9'],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version_config=//xcode:foo");
    assertXcodeVersion("10");
    assertAvailability(XcodeConfigInfo.Availability.BOTH);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertNoEvents();
  }

  @Test
  public void testPreferMutualXcodeFalseOverridesMutual() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "xcode_version(",
        "    name = 'version92',",
        "    version = '9.2',",
        ")",
        "xcode_version(",
        "    name = 'version9',",
        "    version = '9',",
        ")",
        "xcode_version(",
        "    name = 'version10',",
        "    version = '10',",
        "    aliases = [ '10.0'],",
        ")",
        "xcode_version(",
        "    name = 'other_version10',",
        "    version = '10.0',",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version92', ':version9', ':version10'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84', ':other_version10', ':version92', ':version9'],",
        "    default = ':version84',",
        ")");
    useConfiguration(
        "--xcode_version_config=//xcode:foo", "--experimental_prefer_mutual_xcode=false");
    assertXcodeVersion("8.4");
    assertAvailability(XcodeConfigInfo.Availability.LOCAL);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertContainsEvent("You passed --experimental_prefer_mutual_xcode=false");
    assertContainsEvent("consider using --experimental_prefer_mutual_xcode=true");
  }

  @Test
  public void testLocalDefaultCanBeMutuallyAvailable() throws Exception {
    // Passing "--experimental_prefer_mutual_xcode=false" allows toggling between Xcode versions
    // using xcode-select. This test ensures that if the version from xcode-select is available
    // remotely, both local and remote execution are enabled.
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "xcode_version(",
        "    name = 'version92',",
        "    version = '9.2',",
        ")",
        "xcode_version(",
        "    name = 'version9',",
        "    version = '9',",
        ")",
        "xcode_version(",
        "    name = 'version10',",
        "    version = '10',",
        "    aliases = ['10.0.0.10E1001'],",
        ")",
        "xcode_version(",
        "    name = 'other_version10',",
        "    version = '10.0.0.10E1001',",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version92', ':version9', ':version10'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84', ':other_version10', ':version92', ':version9'],",
        "    default = ':other_version10',",
        ")");
    useConfiguration(
        "--xcode_version_config=//xcode:foo", "--experimental_prefer_mutual_xcode=false");
    assertXcodeVersion("10");
    assertAvailability(XcodeConfigInfo.Availability.BOTH);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertNoEvents();
  }

  @Test
  public void testLocalDefaultMatchesOnLocalAlias() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "xcode_version(",
        "    name = 'version92',",
        "    version = '9.2',",
        ")",
        "xcode_version(",
        "    name = 'version9',",
        "    version = '9',",
        ")",
        "xcode_version(",
        "    name = 'version10',",
        "    version = '10',",
        "    aliases = ['10.0'],",
        ")",
        "xcode_version(",
        "    name = 'other_version10',",
        "    version = '10.0.0.10E1001',",
        "    aliases = ['10.0'],",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version92', ':version9', ':version10'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84', ':other_version10', ':version92', ':version9'],",
        "    default = ':other_version10',",
        ")");
    useConfiguration(
        "--xcode_version_config=//xcode:foo", "--experimental_prefer_mutual_xcode=false");
    assertXcodeVersion("10");
    assertAvailability(XcodeConfigInfo.Availability.BOTH);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertNoEvents();
  }

  @Test
  public void testMatchLocalAliasToRemoteName() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "xcode_version(",
        "    name = 'version92',",
        "    version = '9.2',",
        ")",
        "xcode_version(",
        "    name = 'version9',",
        "    version = '9',",
        ")",
        "xcode_version(",
        "    name = 'version10',",
        "    version = '10',",
        "    aliases = ['10.0'],",
        ")",
        "xcode_version(",
        "    name = 'other_version10',",
        "    version = '10.0.0.10E1001',",
        "    aliases = ['10'],",
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version92', ':version9', ':version10'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84', ':other_version10', ':version92', ':version9'],",
        "    default = ':other_version10',",
        ")");
    useConfiguration(
        "--xcode_version_config=//xcode:foo", "--experimental_prefer_mutual_xcode=false");
    assertXcodeVersion("10");
    assertAvailability(XcodeConfigInfo.Availability.BOTH);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

    assertNoEvents();
  }

  @Test
  public void testInvalidXcodeFromMutualThrows() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84'],",
        "    default = ':version84',",
        ")");
    useConfiguration("--xcode_version=6");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent(
        "--xcode_version=6 specified, but '6' is not an available Xcode version."
            + " localy available versions: [8.4]. remotely available versions:"
            + " [5.1.2, 8.4].");
  }

  @Test
  public void xcodeVersionConfig_isFunction() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "result = provider()",
        "def _impl(ctx):",
        "  return [result(xcode_version ="
            + " apple_common.XcodeVersionConfig("
            + " iosSdkVersion='1.1',"
            + " iosMinimumOsVersion='1.2',"
            + " watchosSdkVersion='1.3',"
            + " watchosMinimumOsVersion='1.4',"
            + " tvosSdkVersion='1.5',"
            + " tvosMinimumOsVersion='1.6',"
            + " macosSdkVersion='1.7',"
            + " macosMinimumOsVersion='1.8',"
            + " xcodeVersion='1.9'))]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file("foo/BUILD", "load(':extension.bzl', 'my_rule')", "my_rule(name='test')");
    assertNoEvents();
    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:test");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new SkylarkKey(
                    Label.parseAbsolute("//foo:extension.bzl", ImmutableMap.of()), "result"));
    assertThat(info.getValue("xcode_version"))
        .isEqualTo(
            new XcodeConfigInfo(
                DottedVersion.fromStringUnchecked("1.1"),
                DottedVersion.fromStringUnchecked("1.2"),
                DottedVersion.fromStringUnchecked("1.3"),
                DottedVersion.fromStringUnchecked("1.4"),
                DottedVersion.fromStringUnchecked("1.5"),
                DottedVersion.fromStringUnchecked("1.6"),
                DottedVersion.fromStringUnchecked("1.7"),
                DottedVersion.fromStringUnchecked("1.8"),
                DottedVersion.fromStringUnchecked("1.9"),
                XcodeConfigInfo.Availability.UNKNOWN));
  }

  @Test
  public void xcodeVersionConfig_throwsOnBadInput() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "result = provider()",
        "def _impl(ctx):",
        "  return [result(xcode_version ="
            + " apple_common.XcodeVersionConfig("
            + " iosSdkVersion='not a valid dotted version',"
            + " iosMinimumOsVersion='1.2',"
            + " watchosSdkVersion='1.3',"
            + " watchosMinimumOsVersion='1.4',"
            + " tvosSdkVersion='1.5',"
            + " tvosMinimumOsVersion='1.6',"
            + " macosSdkVersion='1.7',"
            + " macosMinimumOsVersion='1.8',"
            + " xcodeVersion='1.9'))]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() })");
    scratch.file("foo/BUILD", "load(':extension.bzl', 'my_rule')", "my_rule(name='test')");
    assertNoEvents();
    assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:test"));
    assertContainsEvent("Dotted version components must all be of the form");
    assertContainsEvent("got 'not a valid dotted version'");
  }

  @Test
  public void xcodeVersionConfig_exposesExpectedAttributes() throws Exception {
    scratch.file(
        "foo/extension.bzl",
        "result = provider()",
        "def _impl(ctx):",
        "  xcode_version ="
            + " apple_common.XcodeVersionConfig("
            + " iosSdkVersion='1.1',"
            + " iosMinimumOsVersion='1.2',"
            + " watchosSdkVersion='1.3',"
            + " watchosMinimumOsVersion='1.4',"
            + " tvosSdkVersion='1.5',"
            + " tvosMinimumOsVersion='1.6',"
            + " macosSdkVersion='1.7',"
            + " macosMinimumOsVersion='1.8',"
            + " xcodeVersion='1.9')",
        "  return [result(xcode_version=xcode_version.xcode_version(),"
            + " min_os=xcode_version.minimum_os_for_platform_type(ctx.fragments.apple.single_arch_platform.platform_type)),]",
        "my_rule = rule(_impl, attrs = { 'dep' : attr.label() },  fragments = ['apple'])");
    scratch.file("foo/BUILD", "load(':extension.bzl', 'my_rule')", "my_rule(name='test')");
    assertNoEvents();
    ConfiguredTarget myRuleTarget = getConfiguredTarget("//foo:test");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new SkylarkKey(
                    Label.parseAbsolute("//foo:extension.bzl", ImmutableMap.of()), "result"));
    assertThat(info.getValue("xcode_version").toString()).isEqualTo("1.9");
    assertThat(info.getValue("min_os").toString()).isEqualTo("1.8");
  }

  @Test
  public void testConfigAlias_configSetting() throws Exception {
    scratch.file("skylark/BUILD");
    scratch.file(
        "skylark/version_retriever.bzl",
        "def _version_retriever_impl(ctx):",
        "  xcode_properties = ctx.attr.dep[apple_common.XcodeProperties]",
        "  version = xcode_properties.xcode_version",
        "  return [config_common.FeatureFlagInfo(value=version)]",
        "",
        "version_retriever = rule(",
        "  implementation = _version_retriever_impl,",
        "  attrs = {'dep': attr.label()},",
        ")");

    scratch.file(
        "xcode/BUILD",
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
    scratch.file(
        "skylark/version_retriever.bzl",
        "def _version_retriever_impl(ctx):",
        "  xcode_properties = ctx.attr.dep[apple_common.XcodeProperties]",
        "  version = xcode_properties.xcode_version",
        "  return [config_common.FeatureFlagInfo(value=version)]",
        "",
        "version_retriever = rule(",
        "  implementation = _version_retriever_impl,",
        "  attrs = {'dep': attr.label()},",
        ")");

    scratch.file(
        "xcode/BUILD",
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
  public void testValidVersion() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
  }

  @Test
  public void testValidAlias_dottedVersion() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
  }

  @Test
  public void testValidAlias_nonNumerical() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
  }

  @Test
  public void testInvalidXcodeSpecified() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
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
    assertContainsEvent(
        "--xcode_version=6 specified, but '6' is not an available Xcode version. "
            + "available versions: [5.1.2, 8.4]. If you believe you have '6' installed");
  }

  @Test
  public void testRequiresDefault() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
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
    scratch.file(
        "xcode/BUILD",
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
  public void testDuplicateAliases_withinAvailableXcodes() throws Exception {
    scratch.file(
        "xcode/BUILD",
        "xcode_config(",
        "    name = 'foo',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "   name = 'remote',",
        "   default = ':version512',",
        "   versions = [':version512', ':version5'],",
        ")",
        "available_xcodes(",
        "   name = 'local',",
        "   default = ':version512',",
        "   versions = [':version512'],",
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));
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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
    assertHasRequirements(
        ImmutableList.of(
            ExecutionRequirements.REQUIRES_DARWIN, ExecutionRequirements.REQUIREMENTS_SET));

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
    assertAvailability(XcodeConfigInfo.Availability.UNKNOWN);
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
    scratch.file(
        "x/r.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode[apple_common.XcodeVersionConfig]",
        "  ios = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)",
        "  tvos = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.tvos)",
        "  return MyInfo(",
        "    xcode = conf.xcode_version(),",
        "    ios_sdk = conf.sdk_version_for_platform(ios),",
        "    tvos_sdk = conf.sdk_version_for_platform(tvos),",
        "    macos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.macos),",
        "    watchos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.watchos),",
        "    availability = conf.availability(),",
        "    execution_info = conf.execution_info(),",
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
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.parseAbsolute("//x:r.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl info = (StructImpl) r.get(key);

    assertThat(info.getValue("xcode").toString()).isEqualTo("0.0");
    assertThat(info.getValue("ios_sdk").toString()).isEqualTo("1.0");
    assertThat(info.getValue("tvos_sdk").toString()).isEqualTo("2.5");
    assertThat(info.getValue("macos_min").toString()).isEqualTo("3.0");
    assertThat(info.getValue("watchos_min").toString()).isEqualTo("4.5");
    assertThat(info.getValue("availability").toString()).isEqualTo("unknown");
    assertThat((Map<?, ?>) info.getValue("execution_info"))
        .containsKey(ExecutionRequirements.REQUIRES_DARWIN);
    assertThat((Map<?, ?>) info.getValue("execution_info"))
        .containsKey(ExecutionRequirements.REQUIREMENTS_SET);
  }

  @Test
  public void testMutualXcodeFromSkylarkByAlias() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//x:r.bzl', 'r')",
        "xcode_config_alias(name='a')",
        "xcode_config(name='c',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512', ':version84'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")",
        "r(name='r')");
    scratch.file(
        "x/r.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode[apple_common.XcodeVersionConfig]",
        "  ios = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)",
        "  tvos = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.tvos)",
        "  return MyInfo(",
        "    xcode = conf.xcode_version(),",
        "    ios_sdk = conf.sdk_version_for_platform(ios),",
        "    tvos_sdk = conf.sdk_version_for_platform(tvos),",
        "    macos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.macos),",
        "    watchos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.watchos),",
        "    availability = conf.availability(),",
        "    execution_info = conf.execution_info(),",
        "  )",
        "r = rule(implementation = _impl,",
        "    attrs = { '_xcode': attr.label(default = Label('//x:a'))},",
        "    fragments = ['apple'],",
        ")");

    useConfiguration("--xcode_version_config=//x:c");
    ConfiguredTarget r = getConfiguredTarget("//x:r");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.parseAbsolute("//x:r.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl info = (StructImpl) r.get(key);
    assertThat((Map<?, ?>) info.getValue("execution_info"))
        .containsKey(ExecutionRequirements.REQUIRES_DARWIN);
    assertThat((Map<?, ?>) info.getValue("execution_info"))
        .containsKey(ExecutionRequirements.REQUIREMENTS_SET);
  }

  @Test
  public void testLocalXcodeFromSkylarkByAlias() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//x:r.bzl', 'r')",
        "xcode_config_alias(name='a')",
        "xcode_config(name='c',",
        "    remote_versions = ':remote',",
        "    local_versions = ':local',",
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
        ")",
        "available_xcodes(",
        "    name = 'remote',",
        "    versions = [':version512'],",
        "    default = ':version512',",
        ")",
        "available_xcodes(",
        "    name = 'local',",
        "    versions = [':version84',],",
        "    default = ':version84',",
        ")",
        "r(name='r')");
    scratch.file(
        "x/r.bzl",
        "MyInfo = provider()",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode[apple_common.XcodeVersionConfig]",
        "  ios = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.ios)",
        "  tvos = ctx.fragments.apple.multi_arch_platform(apple_common.platform_type.tvos)",
        "  return MyInfo(",
        "    xcode = conf.xcode_version(),",
        "    ios_sdk = conf.sdk_version_for_platform(ios),",
        "    tvos_sdk = conf.sdk_version_for_platform(tvos),",
        "    macos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.macos),",
        "    watchos_min = conf.minimum_os_for_platform_type(apple_common.platform_type.watchos),",
        "    availability = conf.availability(),",
        "  )",
        "r = rule(implementation = _impl,",
        "    attrs = { '_xcode': attr.label(default = Label('//x:a'))},",
        "    fragments = ['apple'],",
        ")");

    useConfiguration("--xcode_version_config=//x:c");
    ConfiguredTarget r = getConfiguredTarget("//x:r");
    Provider.Key key =
        new SkylarkProvider.SkylarkKey(
            Label.parseAbsolute("//x:r.bzl", ImmutableMap.of()), "MyInfo");
    StructImpl info = (StructImpl) r.get(key);

    assertThat(info.getValue("xcode").toString()).isEqualTo("8.4");
    assertThat(info.getValue("availability").toString()).isEqualTo("local");
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

    useConfiguration(
        "--apple_platform_type=ios", "--apple_bitcode=embedded", "--apple_split_cpu=arm64");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//xcode:foo");
    assertContainsEvent("apple_bitcode mode 'embedded' is unsupported");
  }

  // Verifies that the --xcode_version_config configuration value can be accessed via the
  // configuration_field() skylark method and used in a skylark rule.
  @Test
  public void testConfigurationFieldForRule() throws Exception {
    scratch.file(
        "x/provider_grabber.bzl",
        "def _impl(ctx):",
        "  conf = ctx.attr._xcode_dep[apple_common.XcodeVersionConfig]",
        "  return [conf]",
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
    scratch.file(
        "x/provider_grabber.bzl",
        "def _aspect_impl(target, ctx):",
        "  conf = ctx.attr._xcode_dep[apple_common.XcodeVersionConfig]",
        "  return [conf]",
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
        "  return [conf]",
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
    XcodeConfigInfo provider = xcodeConfig.get(XcodeConfigInfo.PROVIDER);
    return provider.getSdkVersionForPlatform(platform);
  }

  private DottedVersion getMinimumOsVersionForPlatform(ApplePlatform platform) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget("//xcode:foo");
    XcodeConfigInfo provider = xcodeConfig.get(XcodeConfigInfo.PROVIDER);
    return provider.getMinimumOsForPlatformType(platform.getType());
  }

  private void assertXcodeVersion(String version) throws Exception {
    assertXcodeVersion(version, "//xcode:foo");
  }

  private void assertXcodeVersion(String version, String providerTargetLabel) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget(providerTargetLabel);
    XcodeConfigInfo provider = xcodeConfig.get(XcodeConfigInfo.PROVIDER);
    assertThat(provider.getXcodeVersion()).isEqualTo(DottedVersion.fromString(version));
  }

  private void assertAvailability(XcodeConfigInfo.Availability availabilty) throws Exception {
    assertAvailability(availabilty, "//xcode:foo");
  }

  private void assertAvailability(
      XcodeConfigInfo.Availability availabilty, String providerTargetLabel) throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget(providerTargetLabel);
    XcodeConfigInfo provider = xcodeConfig.get(XcodeConfigInfo.PROVIDER);
    assertThat(provider.getAvailability()).isEqualTo(availabilty);
  }

  private void assertHasRequirements(List<String> executionRequirements) throws Exception {
    assertHasRequirements(executionRequirements, "//xcode:foo");
  }

  private void assertHasRequirements(List<String> executionRequirements, String providerTargetLabel)
      throws Exception {
    ConfiguredTarget xcodeConfig = getConfiguredTarget(providerTargetLabel);
    XcodeConfigInfo provider = xcodeConfig.get(XcodeConfigInfo.PROVIDER);
    for (String requirement : executionRequirements) {
      assertThat(requirement).isIn(provider.getExecutionRequirements().keySet());
    }
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
