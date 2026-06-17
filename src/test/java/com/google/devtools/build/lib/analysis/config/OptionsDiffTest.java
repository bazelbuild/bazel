// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.Scope.ScopeType;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of @link OptionsDiff}. */
@RunWith(JUnit4.class)
public class OptionsDiffTest {
  private static final ImmutableList<Class<? extends FragmentOptions>> BUILD_CONFIG_OPTIONS =
      ImmutableList.of(CoreOptions.class);

  @Test
  public void diff() throws Exception {
    BuildOptions one = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--compilation_mode=dbg", "cpu=k8");
    BuildOptions three = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--compilation_mode=dbg", "cpu=k8");

    OptionsDiff diffOneTwo = OptionsDiff.diff(one, two);
    OptionsDiff diffTwoThree = OptionsDiff.diff(two, three);

    assertThat(diffOneTwo.areSame()).isFalse();
    assertThat(diffOneTwo.getFirst().keySet()).isEqualTo(diffOneTwo.getSecond().keySet());
    assertThat(diffOneTwo.prettyPrint()).contains("opt");
    assertThat(diffOneTwo.prettyPrint()).contains("dbg");

    assertThat(diffTwoThree.areSame()).isTrue();
  }

  @Test
  public void diff_differentFragments() throws Exception {
    BuildOptions one = BuildOptions.of(ImmutableList.of(CppOptions.class));
    BuildOptions two = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    OptionsDiff diff = OptionsDiff.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getExtraFirstFragmentClassesForTesting()).containsExactly(CppOptions.class);
    assertThat(
            diff.getExtraSecondFragmentsForTesting().stream().map(FragmentOptions::getOptionsClass))
        .containsExactlyElementsIn(BUILD_CONFIG_OPTIONS);
  }

  @Test
  public void biff_nullOptionsThrow() throws Exception {
    BuildOptions options =
        BuildOptions.of(BUILD_CONFIG_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    assertThrows(NullPointerException.class, () -> OptionsDiff.diff(options, null));
    assertThrows(NullPointerException.class, () -> OptionsDiff.diff(null, options));
  }

  @Test
  public void diff_sameStarlarkOptions() {
    Label flagName = Label.parseCanonicalUnchecked("//foo/flag");
    String flagValue = "value";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValue));

    assertThat(OptionsDiff.diff(one, two).areSame()).isTrue();
  }

  @Test
  public void diff_differentStarlarkOptions() {
    Label flagName = Label.parseCanonicalUnchecked("//bar/flag");
    String flagValueOne = "valueOne";
    String flagValueTwo = "valueTwo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValueOne));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValueTwo));

    OptionsDiff diff = OptionsDiff.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getStarlarkFirstForTesting().keySet())
        .isEqualTo(diff.getStarlarkSecondForTesting().keySet());
    assertThat(diff.getStarlarkFirstForTesting().keySet()).containsExactly(flagName);
    assertThat(diff.getStarlarkFirstForTesting().values()).containsExactly(flagValueOne);
    assertThat(diff.getStarlarkSecondForTesting().values()).containsExactly(flagValueTwo);
  }

  @Test
  public void diff_extraStarlarkOptions() {
    Label flagNameOne = Label.parseCanonicalUnchecked("//extra/flag/one");
    Label flagNameTwo = Label.parseCanonicalUnchecked("//extra/flag/two");
    String flagValue = "foo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagNameOne, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagNameTwo, flagValue));

    OptionsDiff diff = OptionsDiff.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getExtraStarlarkOptionsFirstForTesting()).containsExactly(flagNameOne);
    assertThat(diff.getExtraStarlarkOptionsSecondForTesting().entrySet())
        .containsExactly(Maps.immutableEntry(flagNameTwo, flagValue));
  }

  @Test
  public void diff_starlarkMetadataChanged() {
    Label flagName = Label.parseCanonicalUnchecked("//metadata/flag");
    ScopeType scope1 = new ScopeType(ScopeType.DEFAULT);
    ScopeType scope2 = new ScopeType(ScopeType.UNIVERSAL);

    BuildOptions one =
        BuildOptions.builder()
            .addStarlarkOption(flagName, "value")
            .addScopeType(flagName, scope1)
            .addOnLeaveScopeValue(flagName, "onLeave1")
            .build();

    BuildOptions two =
        BuildOptions.builder()
            .addStarlarkOption(flagName, "value")
            .addScopeType(flagName, scope2)
            .addOnLeaveScopeValue(flagName, "onLeave2")
            .build();

    OptionsDiff diff = OptionsDiff.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getStarlarkFirstScopes()).containsExactly(flagName, scope1);
    assertThat(diff.getStarlarkSecondScopes()).containsExactly(flagName, scope2);
    assertThat(diff.getStarlarkFirstOnLeaveValues()).containsExactly(flagName, "onLeave1");
    assertThat(diff.getStarlarkSecondOnLeaveValues()).containsExactly(flagName, "onLeave2");
  }

  @Test
  public void diff_extraStarlarkMetadata() {
    Label flagName = Label.parseCanonicalUnchecked("//metadata/flag");
    ScopeType scope = new ScopeType(ScopeType.DEFAULT);

    BuildOptions one = BuildOptions.builder().build();

    BuildOptions two =
        BuildOptions.builder()
            .addStarlarkOption(flagName, "value")
            .addScopeType(flagName, scope)
            .addOnLeaveScopeValue(flagName, "onLeaveValue")
            .build();

    OptionsDiff diff = OptionsDiff.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getExtraStarlarkOptionsSecondScopes()).containsExactly(flagName, scope);
    assertThat(diff.getExtraStarlarkOptionsSecondOnLeaveValues())
        .containsExactly(flagName, "onLeaveValue");
  }

  @Test
  public void testApplyDiff_recreatesIdenticalObject() throws Exception {
    BuildOptions one =
        BuildOptions.of(ImmutableList.of(CoreOptions.class), "--compilation_mode=opt", "cpu=k8");

    Label flagName = Label.parseCanonicalUnchecked("//metadata/flag");
    ScopeType scope = new ScopeType(ScopeType.UNIVERSAL);

    BuildOptions two =
        BuildOptions.of(ImmutableList.of(CoreOptions.class)).toBuilder()
            .addStarlarkOption(flagName, "value")
            .addScopeType(flagName, scope)
            .addOnLeaveScopeValue(flagName, "onLeaveValue")
            .build();

    // Diff one - two
    OptionsDiff diff = OptionsDiff.diff(one, two);

    // Apply diff on one to reconstruct two
    BuildOptions reconstructedTwo = OptionsDiff.applyDiff(one, diff);

    // Verify equality of two and reconstructed two
    assertThat(reconstructedTwo).isEqualTo(two);
    assertThat(reconstructedTwo.getScopeTypeMap()).isEqualTo(two.getScopeTypeMap());
    assertThat(reconstructedTwo.getOnLeaveScopeValues()).isEqualTo(two.getOnLeaveScopeValues());
  }
}
