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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the ConfigurationFragmentPolicy builder and methods.
 */
@RunWith(JUnit4.class)
public final class ConfigurationFragmentPolicyTest {

  @StarlarkBuiltin(name = "test_fragment", doc = "first fragment")
  private static final class TestFragment implements StarlarkValue {}

  @StarlarkBuiltin(name = "other_fragment", doc = "second fragment")
  private static final class OtherFragment implements StarlarkValue {}

  @StarlarkBuiltin(name = "unknown_fragment", doc = "useless waste of permgen")
  private static final class UnknownFragment implements StarlarkValue {}

  @Test
  public void testMissingFragmentPolicy() throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(Integer.class, MissingFragmentPolicy.IGNORE)
            .build();

    assertThat(policy.getMissingFragmentPolicy(Integer.class))
        .isEqualTo(MissingFragmentPolicy.IGNORE);

    ConfigurationFragmentPolicy otherPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(String.class, MissingFragmentPolicy.CREATE_FAIL_ACTIONS)
            .build();

    assertThat(otherPolicy.getMissingFragmentPolicy(String.class))
        .isEqualTo(MissingFragmentPolicy.CREATE_FAIL_ACTIONS);
  }

  @Test
  public void testRequiresConfigurationFragments_addsToRequiredSet() throws Exception {
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

  private static final ConfigurationTransition TEST_HOST_TRANSITION =
      new ConfigurationTransition() {
        @Override
        public ImmutableMap<String, BuildOptions> apply(
            BuildOptionsView buildOptions, EventHandler eventHandler) {
          return ImmutableMap.of("", buildOptions.underlying());
        }

        @Override
        public String reasonForOverride() {
          return null;
        }

        @Override
        public boolean isHostTransition() {
          return true;
        }
      };

  @Test
  public void testRequiresConfigurationFragments_requiredAndLegalForSpecifiedConfiguration()
      throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class))
            .requiresConfigurationFragments(TEST_HOST_TRANSITION,
                ImmutableSet.<Class<?>>of(Long.class))
            .build();

    assertThat(policy.getRequiredConfigurationFragments())
        .containsAtLeast(Integer.class, Long.class);

    assertThat(policy.isLegalConfigurationFragment(Integer.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(Integer.class, NoTransition.INSTANCE))
        .isTrue();
    // TODO(b/140641941): .isFalse() when dynamic configurations care which configuration a fragment
    // was specified for
    assertThat(policy.isLegalConfigurationFragment(Integer.class, TEST_HOST_TRANSITION)).isTrue();

    assertThat(policy.isLegalConfigurationFragment(Long.class)).isTrue();
    // TODO(b/140641941): .isFalse() when dynamic configurations care which configuration a fragment
    // was specified for
    assertThat(policy.isLegalConfigurationFragment(Long.class, NoTransition.INSTANCE)).isTrue();
    assertThat(policy.isLegalConfigurationFragment(Long.class, TEST_HOST_TRANSITION))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(String.class)).isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, NoTransition.INSTANCE))
        .isFalse();
    assertThat(policy.isLegalConfigurationFragment(String.class, TEST_HOST_TRANSITION))
        .isFalse();
  }

  @Test
  public void testRequiresConfigurationFragments_mapSetsLegalityByStarlarkModuleName_noRequires()
      throws Exception {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableSet.of("test_fragment"))
            .requiresConfigurationFragmentsByStarlarkBuiltinName(
                TEST_HOST_TRANSITION, ImmutableSet.of("other_fragment"))
            .build();

    assertThat(policy.getRequiredConfigurationFragments()).isEmpty();

    assertThat(policy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, NoTransition.INSTANCE))
        .isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(TestFragment.class, TEST_HOST_TRANSITION))
        .isFalse();

    assertThat(policy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, NoTransition.INSTANCE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(OtherFragment.class, TEST_HOST_TRANSITION))
        .isTrue();

    assertThat(policy.isLegalConfigurationFragment(UnknownFragment.class)).isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(UnknownFragment.class, NoTransition.INSTANCE))
        .isFalse();
    assertThat(
            policy.isLegalConfigurationFragment(
                UnknownFragment.class, TEST_HOST_TRANSITION))
        .isFalse();
  }

  @Test
  public void testIncludeConfigurationFragmentsFrom_mergesWithExistingFragmentSet()
      throws Exception {
    ConfigurationFragmentPolicy basePolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableSet.of("test_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Integer.class, Double.class))
            .build();
    ConfigurationFragmentPolicy addedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragmentsByStarlarkBuiltinName(
                TEST_HOST_TRANSITION, ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragments(ImmutableSet.<Class<?>>of(Boolean.class))
            .requiresConfigurationFragments(
                TEST_HOST_TRANSITION, ImmutableSet.<Class<?>>of(Character.class))
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
