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
import com.google.devtools.build.lib.analysis.config.Fragment;
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

  private static final class FragmentA extends Fragment {}

  private static final class FragmentB extends Fragment {}

  private static final class FragmentC extends Fragment {}

  private static final class FragmentD extends Fragment {}

  @Test
  public void testMissingFragmentPolicy() {
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(FragmentA.class, MissingFragmentPolicy.IGNORE)
            .build();

    assertThat(policy.getMissingFragmentPolicy(FragmentA.class))
        .isEqualTo(MissingFragmentPolicy.IGNORE);

    ConfigurationFragmentPolicy otherPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .setMissingFragmentPolicy(FragmentB.class, MissingFragmentPolicy.CREATE_FAIL_ACTIONS)
            .build();

    assertThat(otherPolicy.getMissingFragmentPolicy(FragmentB.class))
        .isEqualTo(MissingFragmentPolicy.CREATE_FAIL_ACTIONS);
  }

  @Test
  public void testRequiresConfigurationFragments_addsToRequiredSet() throws Exception {
    // Although these aren't configuration fragments, there are no requirements as to what the class
    // has to be, so...
    ConfigurationFragmentPolicy policy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragments(ImmutableSet.of(FragmentA.class, FragmentB.class))
            .requiresConfigurationFragments(ImmutableSet.of(FragmentB.class, FragmentC.class))
            .build();

    assertThat(policy.getRequiredConfigurationFragments())
        .containsExactly(FragmentA.class, FragmentB.class, FragmentC.class);
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
            .requiresConfigurationFragments(ImmutableSet.of(FragmentA.class))
            .requiresConfigurationFragments(TEST_HOST_TRANSITION, ImmutableSet.of(FragmentB.class))
            .build();

    assertThat(policy.getRequiredConfigurationFragments())
        .containsAtLeast(FragmentA.class, FragmentB.class);

    assertThat(policy.isLegalConfigurationFragment(FragmentA.class)).isTrue();
    assertThat(policy.isLegalConfigurationFragment(FragmentA.class, NoTransition.INSTANCE))
        .isTrue();
    // TODO(b/140641941): .isFalse() when dynamic configurations care which configuration a fragment
    // was specified for
    assertThat(policy.isLegalConfigurationFragment(FragmentA.class, TEST_HOST_TRANSITION)).isTrue();

    assertThat(policy.isLegalConfigurationFragment(FragmentB.class)).isTrue();
    // TODO(b/140641941): .isFalse() when dynamic configurations care which configuration a fragment
    // was specified for
    assertThat(policy.isLegalConfigurationFragment(FragmentB.class, NoTransition.INSTANCE))
        .isTrue();
    assertThat(policy.isLegalConfigurationFragment(FragmentB.class, TEST_HOST_TRANSITION)).isTrue();

    assertThat(policy.isLegalConfigurationFragment(FragmentC.class)).isFalse();
    assertThat(policy.isLegalConfigurationFragment(FragmentC.class, NoTransition.INSTANCE))
        .isFalse();
    assertThat(policy.isLegalConfigurationFragment(FragmentC.class, TEST_HOST_TRANSITION))
        .isFalse();
  }

  @Test
  public void testRequiresConfigurationFragments_mapSetsLegalityByStarlarkModuleName_noRequires() {
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
            .requiresConfigurationFragments(ImmutableSet.of(FragmentA.class, FragmentB.class))
            .build();
    ConfigurationFragmentPolicy addedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragmentsByStarlarkBuiltinName(
                TEST_HOST_TRANSITION, ImmutableSet.of("other_fragment"))
            .requiresConfigurationFragments(ImmutableSet.of(FragmentC.class))
            .requiresConfigurationFragments(TEST_HOST_TRANSITION, ImmutableSet.of(FragmentD.class))
            .build();
    ConfigurationFragmentPolicy combinedPolicy =
        new ConfigurationFragmentPolicy.Builder()
            .includeConfigurationFragmentsFrom(basePolicy)
            .includeConfigurationFragmentsFrom(addedPolicy)
            .build();

    assertThat(combinedPolicy.getRequiredConfigurationFragments())
        .containsExactly(FragmentA.class, FragmentB.class, FragmentC.class, FragmentD.class);
    assertThat(combinedPolicy.isLegalConfigurationFragment(TestFragment.class)).isTrue();
    assertThat(combinedPolicy.isLegalConfigurationFragment(OtherFragment.class)).isTrue();
  }
}
