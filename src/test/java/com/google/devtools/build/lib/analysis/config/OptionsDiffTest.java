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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
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
            diff.getExtraSecondFragmentsForTesting().stream()
                .map(Object::getClass)
                .collect(toImmutableList()))
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
}
