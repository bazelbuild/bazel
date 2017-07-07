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

package com.google.devtools.build.lib.analysis.featurecontrol;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the FeaturePolicyLoader. */
@RunWith(JUnit4.class)
public final class FeaturePolicyLoaderTest extends AnalysisTestCase {

  private FeaturePolicyConfiguration getFragmentWithFeatures(
      Iterable<String> allowedFeatures, Collection<String> args) throws Exception {
    ConfigurationEnvironment env =
        new ConfigurationEnvironment.TargetProviderEnvironment(
            skyframeExecutor.getPackageManager(), reporter, directories);
    BuildOptions options =
        BuildOptions.of(
            ImmutableList.<Class<? extends FragmentOptions>>of(FeaturePolicyOptions.class),
            args.toArray(new String[0]));
    return (FeaturePolicyConfiguration)
        new FeaturePolicyLoader(allowedFeatures).create(env, options);
  }

  @Test
  public void specifiedFeaturesGetListedAccessPolicy() throws Exception {
    scratch.file(
        "policy/BUILD",
        "package_group(",
        "    name='featured',",
        "    packages=[",
        "        '//direct',",
        "        '//recursive/...',",
        "    ])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//policy:featured"));
    assertThat(fragment.getPolicyForFeature("policiedFeature")).isEqualTo("//policy:featured");
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//:rule")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//policy:featured")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//direct:allow")))
        .isTrue();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//direct/child:allow")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//recursive:allow")))
        .isTrue();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//recursive/child:allow")))
        .isTrue();
  }

  @Test
  public void resolvesIncludedPackageGroups() throws Exception {
    scratch.file(
        "policy/BUILD",
        "package_group(",
        "    name='main',",
        "    packages=['//a'],",
        "    includes=[':include'])",
        "package_group(",
        "    name='include',",
        "    packages=['//b'])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//policy:main"));
    assertThat(fragment.getPolicyForFeature("policiedFeature")).isEqualTo("//policy:main");
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isFalse();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//a:a")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//b:b")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//c:c")))
        .isFalse();
  }

  @Test
  public void resolvesAliasToPolicy() throws Exception {
    scratch.file(
        "policy/BUILD",
        "alias(",
        "    name='aliased',",
        "    actual=':featured')",
        "package_group(",
        "    name='featured',",
        "    packages=[",
        "        '//direct',",
        "        '//recursive/...',",
        "    ])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//policy:aliased"));
    assertThat(fragment.getPolicyForFeature("policiedFeature")).isEqualTo("//policy:aliased");
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//policy:aliased")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//policy:featured")))
        .isFalse();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//direct:allow")))
        .isTrue();
  }

  @Test
  public void resolvesAliasToIncludesInPackageGroups() throws Exception {
    scratch.file(
        "policy/BUILD",
        "package_group(",
        "    name='main',",
        "    packages=['//a'],",
        "    includes=[':aliased'])",
        "alias(",
        "    name='aliased',",
        "    actual=':include')",
        "package_group(",
        "    name='include',",
        "    packages=['//b'])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//policy:main"));
    assertThat(fragment.getPolicyForFeature("policiedFeature")).isEqualTo("//policy:main");
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isFalse();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//a:a")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//b:b")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//c:c")))
        .isFalse();
  }

  @Test
  public void allowsCyclesInPackageGroupsAndCoversAllMembersOfCycle() throws Exception {
    scratch.file(
        "policy/BUILD",
        "package_group(",
        "    name='cycle',",
        "    packages=['//a'],",
        "    includes=[':elcyc'])",
        "package_group(",
        "    name='elcyc',",
        "    packages=['//b'],",
        "    includes=[':cycle'])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//policy:cycle"));
    assertThat(fragment.getPolicyForFeature("policiedFeature")).isEqualTo("//policy:cycle");
    assertThat(
            fragment.isFeatureEnabledForRule(
                "policiedFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isFalse();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//a:a")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//b:b")))
        .isTrue();
    assertThat(fragment.isFeatureEnabledForRule("policiedFeature", Label.parseAbsolute("//c:c")))
        .isFalse();
  }

  @Test
  public void unspecifiedFeaturesGetUniversalAccessPolicy() throws Exception {
    scratch.file("null/BUILD", "package_group(name='null', packages=[])");
    FeaturePolicyConfiguration fragment =
        getFragmentWithFeatures(
            ImmutableSet.of("defaultFeature", "policiedFeature"),
            ImmutableList.of("--feature_control_policy=policiedFeature=//null:null"));
    assertThat(fragment.getPolicyForFeature("defaultFeature")).isEqualTo("//...");
    assertThat(fragment.isFeatureEnabledForRule("defaultFeature", Label.parseAbsolute("//:rule")))
        .isTrue();
    assertThat(
            fragment.isFeatureEnabledForRule(
                "defaultFeature", Label.parseAbsolute("//arbitrary:rule")))
        .isTrue();
  }

  @Test
  public void throwsForFeatureWithMultiplePolicyDefinitions() throws Exception {
    scratch.file(
        "null/BUILD",
        "package_group(name='null', packages=[])",
        "package_group(name='empty', packages=[])");
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("duplicateFeature"),
          ImmutableList.of(
              "--feature_control_policy=duplicateFeature=//null:null",
              "--feature_control_policy=duplicateFeature=//null:empty"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected).hasMessageThat().contains("Multiple definitions");
      assertThat(expected).hasMessageThat().contains("duplicateFeature");
    }
  }

  @Test
  public void throwsForFeatureNotSpecifiedInLoader() throws Exception {
    scratch.file("null/BUILD", "package_group(name='null', packages=[])");
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("otherFeature"),
          ImmutableList.of("--feature_control_policy=missingFeature=//null:null"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected).hasMessageThat().contains("No such feature");
      assertThat(expected).hasMessageThat().contains("missingFeature");
    }
  }

  @Test
  public void throwsForFeatureWithNonexistentPolicy() throws Exception {
    scratch.file("null/BUILD", "package_group(name='null', packages=[])");
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("brokenFeature"),
          ImmutableList.of("--feature_control_policy=brokenFeature=//null:missing"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected).hasMessageThat().contains("no such target '//null:missing'");
    }
  }

  @Test
  public void throwsForFeatureWithPolicyInNonexistentPackage() throws Exception {
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("brokenFeature"),
          ImmutableList.of("--feature_control_policy=brokenFeature=//missing:missing"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected).hasMessageThat().contains("no such package 'missing'");
    }
  }

  @Test
  public void throwsForFeatureWithNonPackageGroupPolicy() throws Exception {
    scratch.file("policy/BUILD", "filegroup(name='non_package_group')");
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("brokenFeature"),
          ImmutableList.of("--feature_control_policy=brokenFeature=//policy:non_package_group"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains(
              "//policy:non_package_group is not a package_group in brokenFeature feature policy");
    }
  }

  @Test
  public void throwsForFeatureWithNonRulePolicy() throws Exception {
    scratch.file("policy/BUILD", "exports_files(['not_even_a_rule'])");
    try {
      getFragmentWithFeatures(
          ImmutableSet.of("brokenFeature"),
          ImmutableList.of("--feature_control_policy=brokenFeature=//policy:not_even_a_rule"));
      fail("Expected an exception");
    } catch (InvalidConfigurationException expected) {
      assertThat(expected)
          .hasMessageThat()
          .contains(
              "//policy:not_even_a_rule is not a package_group in brokenFeature feature policy");
    }
  }
}
