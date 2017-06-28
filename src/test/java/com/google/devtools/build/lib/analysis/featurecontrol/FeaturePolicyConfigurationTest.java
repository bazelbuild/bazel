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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.PackageSpecification;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the FeaturePolicyConfiguration. */
@RunWith(JUnit4.class)
public final class FeaturePolicyConfigurationTest {

  @Test
  public void isFeatureEnabledForRule_FalseIfAbsentFromFeatureMap() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.<String, PackageSpecification>of(),
            ImmutableMap.of("newFeature", "//package_group:empty"));

    assertThat(config.isFeatureEnabledForRule("newFeature", Label.parseAbsolute("//:rule")))
        .isFalse();
  }

  @Test
  public void isFeatureEnabledForRule_TrueIfMappedToEverything() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of("newFeature", PackageSpecification.everything()),
            ImmutableMap.of("newFeature", "//..."));

    assertThat(config.isFeatureEnabledForRule("newFeature", Label.parseAbsolute("//:rule")))
        .isTrue();
  }

  @Test
  public void isFeatureEnabledForRule_TrueIfInPackageSpecification() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of(
                "newFeature",
                PackageSpecification.fromString(RepositoryName.MAIN, "//allowed/...")),
            ImmutableMap.of("newFeature", "//allowed/..."));

    assertThat(config.isFeatureEnabledForRule("newFeature", Label.parseAbsolute("//allowed:rule")))
        .isTrue();
  }

  @Test
  public void isFeatureEnabledForRule_FalseIfNotInPackageSpecification() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of(
                "newFeature",
                PackageSpecification.fromString(RepositoryName.MAIN, "//allowed/...")),
            ImmutableMap.of("newFeature", "//allowed/..."));

    assertThat(
            config.isFeatureEnabledForRule("newFeature", Label.parseAbsolute("//forbidden:rule")))
        .isFalse();
  }

  @Test
  public void isFeatureEnabledForRule_FailsIfNotPresentInPolicyList() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of("newFeature", PackageSpecification.everything()),
            ImmutableMap.<String, String>of());

    try {
      config.isFeatureEnabledForRule("newFeature", Label.parseAbsolute("//:rule"));
      fail("Expected an exception.");
    } catch (IllegalArgumentException expected) {
      assertThat(expected).hasMessageThat().contains("No such feature: newFeature");
    }
  }

  @Test
  public void getPolicyForFeature_ReturnsValueFromPolicy() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of("newFeature", PackageSpecification.everything()),
            ImmutableMap.<String, String>of("newFeature", "my policy"));

    assertThat(config.getPolicyForFeature("newFeature")).isEqualTo("my policy");
  }

  @Test
  public void getPolicyForFeature_FailsIfNotPresentInPolicyList() throws Exception {
    FeaturePolicyConfiguration config =
        new FeaturePolicyConfiguration(
            ImmutableSetMultimap.of("newFeature", PackageSpecification.everything()),
            ImmutableMap.<String, String>of());

    try {
      config.getPolicyForFeature("newFeature");
      fail("Expected an exception.");
    } catch (IllegalArgumentException expected) {
      assertThat(expected).hasMessageThat().contains("No such feature: newFeature");
    }
  }
}
