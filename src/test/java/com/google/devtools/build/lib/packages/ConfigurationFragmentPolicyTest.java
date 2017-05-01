// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the ConfigurationFragmentPolicy builder and methods.
 */
@RunWith(JUnit4.class)
public final class ConfigurationFragmentPolicyTest {

  @SkylarkModule(name = "test_fragment", doc = "first fragment")
  private static final class TestFragment {}

  @SkylarkModule(name = "other_fragment", doc = "second fragment")
  private static final class OtherFragment {}

  @SkylarkModule(name = "unknown_fragment", doc = "useless waste of permgen")
  private static final class UnknownFragment {}

  @Test
  public void testMissingFragmentPolicy() throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(MissingFragmentPolicy.IGNORE)
            .build();

    assertThat(policy.getMissingFragmentPolicy()).isEqualTo(MissingFragmentPolicy.IGNORE);

    ConfigurationFragmentPolicy otherPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(MissingFragmentPolicy.CREATE_FAIL_ACTIONS)
            .build();

    assertThat(otherPolicy.getMissingFragmentPolicy())
        .isEqualTo(MissingFragmentPolicy.CREATE_FAIL_ACTIONS);
  }

  @Test
  public void testRequiresConfigurationFragments_AddsToRequiredSet() throws Exception {
    // Although these aren't configuration fragments, there are no requirements as to what the class
    // has to be, so...
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class, String.class))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(String.class, Long.class))
            .build();

    assertThat(policy.getRequiredConfigurationFragments())
        .containsExactly(Integer.class, String.class, Long.class);
  }

  @Test
  public void testRequiresConfigurationFragments_RequiredAndLegalForSpecifiedConfiguration()
      throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class))
            .requiresHostConfigurationFragments(ImmutableSet.<Class<?>>of(Long.class))
            .build();

    assertThat(policy.getRequiredConfigurationFragments()).containsAllOf(Integer.class, Long.class);

    assertThat(policy.isLegalConfigurationFragment(Integer.class)).isTrue();
    assertThat(policy.isLegalConfigurationFragment(Integer.class, ConfigurationTransition.NONE))
        .isTrue();
    // TODO(mstaib): .isFalse() when dynamic configurations care which configuration a fragment was
    // specified for
    assertThat(policy.isLegalConfigurationFragment(Integer.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(Long.class)).isTrue();
    // TODO(mstaib): .isFalse() when dynamic configurations care which configuration a fragment was
    // specified for
    assertThat(policy.isLegalConfigurationFragment(Long.class, ConfigurationTransition.NONE))
        .isTrue();
    assertThat(policy.isLegalConfigurationFragment(Long.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(String.class)).isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, ConfigurationTransition.HOST))
        .isFalse();
  }

  @Test
  public void testRequiresConfigurationFragments_MapSetsLegalityBySkylarkModuleName_NoRequires()
      throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("test_fragment"))
            .requiresHostConfigurationFragmentsBySkylarkModuleName(
                ImmutableSet.of("other_fragment"))
            .build();

    assertThat(policy.getRequiredConfigurationFragments()).isEmpty();

    assertThat(policy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.NONE))
        .isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.HOST))
        .isFalse();

    assertThat(policy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, ConfigurationTransition.HOST))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(UnknownFragment.class)).isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(
                UnknownFragment.class, ConfigurationTransition.NONE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(
                UnknownFragment.class, ConfigurationTransition.HOST))
        .isFalse();
  }

  @Test
  public void testIncludeConfigurationFragmentsFrom_MergesWithExistingFragmentSet()
      throws Exception {
    ConfigurationFragmentPolicy basePolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("test_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class, Double.class))
            .build();
    ConfigurationFragmentPolicy addedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableSet.of("other_fragment"))
            .requiresHostConfigurationFragmentsBySkylarkModuleName(
                ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Boolean.class))
            .requiresHostConfigurationFragments(ImmutableSet.<Class<?>>of(Character.class))
            .build();
    ConfigurationFragmentPolicy combinedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .includeConfigurationFragmentsFrom(basePolicy)
            .includeConfigurationFragmentsFrom(addedPolicy)
            .build();

    assertThat(combinedPolicy.getRequiredConfigurationFragments())
        .containsExactly(Integer.class, Double.class, Boolean.class, Character.class);
    assertThat(combinedPolicy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(combinedPolicy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
  }
}
