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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag options data. */
@RunWith(JUnit4.class)
public final class FeatureFlagValueTest extends BuildViewTestCase {

  private BuildOptions emptyBuildOptions() throws Exception {
    return BuildOptions.of(ImmutableList.of(ConfigFeatureFlagOptions.class));
  }

  private Set<Label> getKnownDefaultFlags(BuildOptions options) {
    return options.getStarlarkOptions().entrySet().stream()
        .filter((entry) -> FeatureFlagValue.DefaultValue.INSTANCE.equals(entry.getValue()))
        .map(Map.Entry::getKey)
        .collect(toImmutableSet());
  }

  @Test
  public void replaceFlagValues_reflectedInGetFlagValues() throws Exception {
    BuildOptions options =
        FeatureFlagValue.replaceFlagValues(
            emptyBuildOptions(),
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"), "value",
                Label.parseCanonicalUnchecked("//label:b"), "otherValue"));
    assertThat(options.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"),
            FeatureFlagValue.SetValue.of("value"),
            Label.parseCanonicalUnchecked("//label:b"),
            FeatureFlagValue.SetValue.of("otherValue"));
  }

  @Test
  public void replaceFlagValues_totallyReplacesFlagValuesMap() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"), "value",
                Label.parseCanonicalUnchecked("//label:b"), "otherValue"));
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"), "differentValue",
                Label.parseCanonicalUnchecked("//label:c"), "differentFlag"));
    assertThat(options.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"),
            FeatureFlagValue.SetValue.of("differentValue"),
            Label.parseCanonicalUnchecked("//label:c"),
            FeatureFlagValue.SetValue.of("differentFlag"));
  }

  @Test
  public void replaceFlagValues_emptiesKnownDefaultFlagsAndUnknownFlags() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseCanonicalUnchecked("//label:a"), "value",
            Label.parseCanonicalUnchecked("//label:b"), "otherValue");
    BuildOptions options = emptyBuildOptions();
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSortedSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:c"),
                Label.parseCanonicalUnchecked("//label:d")));
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent).isTrue();
  }

  @Test
  public void replaceFlagValues_leavesNonFeatureFlagValuesAlone() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseCanonicalUnchecked("//label:a"), "value",
            Label.parseCanonicalUnchecked("//label:b"), "otherValue");
    Map<Label, String> newMap =
        ImmutableMap.of(
            Label.parseCanonicalUnchecked("//label:a"), "differentValue",
            Label.parseCanonicalUnchecked("//label:c"), "differentFlag");
    BuildOptions options =
        emptyBuildOptions().toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true)
            .build();
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    options = FeatureFlagValue.replaceFlagValues(options, newMap);
    assertThat(options.getStarlarkOptions())
        .containsEntry(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true);
  }

  @Test
  public void trimFlagValues_defaults_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(options.getStarlarkOptions()).isEmpty();
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options)).isEmpty();
  }

  @Test
  public void trimFlagValues_defaults_toPopulatedSetPopulatesKnownDefaultFlags() throws Exception {
    BuildOptions options = emptyBuildOptions();

    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b"),
                Label.parseCanonicalUnchecked("//label:c")));

    assertThat(options.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"), FeatureFlagValue.DefaultValue.INSTANCE,
            Label.parseCanonicalUnchecked("//label:b"), FeatureFlagValue.DefaultValue.INSTANCE,
            Label.parseCanonicalUnchecked("//label:c"), FeatureFlagValue.DefaultValue.INSTANCE);
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options))
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"),
            Label.parseCanonicalUnchecked("//label:b"),
            Label.parseCanonicalUnchecked("//label:c"));
  }

  @Test
  public void trimFlagValues_withFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(options.getStarlarkOptions()).isEmpty();
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options)).isEmpty();
  }

  @Test
  public void trimFlagValues_withFlagsSet_toPopulatedSetPopulatesFlagValuesAndKnownDefaultFlags()
      throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));

    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b"),
                Label.parseCanonicalUnchecked("//label:c")));

    assertThat(options.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"), FeatureFlagValue.SetValue.of("value"),
            Label.parseCanonicalUnchecked("//label:b"), FeatureFlagValue.DefaultValue.INSTANCE,
            Label.parseCanonicalUnchecked("//label:c"), FeatureFlagValue.DefaultValue.INSTANCE);
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options))
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:b"), Label.parseCanonicalUnchecked("//label:c"));
  }

  @Test
  public void trimFlagValues_withTrimmedFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b")));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(options.getStarlarkOptions()).isEmpty();
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options)).isEmpty();
  }

  @Test
  public void trimFlagValues_withTrimmedFlagsSet_toPopulatedSetPopulatesFlagState()
      throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b")));

    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b"),
                Label.parseCanonicalUnchecked("//label:c")));

    assertThat(options.getStarlarkOptions())
        .containsExactly(
            Label.parseCanonicalUnchecked("//label:a"), FeatureFlagValue.SetValue.of("value"),
            Label.parseCanonicalUnchecked("//label:b"), FeatureFlagValue.DefaultValue.INSTANCE,
            Label.parseCanonicalUnchecked("//label:c"), FeatureFlagValue.UnknownValue.INSTANCE);
  }

  @Test
  public void trimFlagValues_leavesNonFeatureFlagValuesAlone() throws Exception {
    BuildOptions options =
        emptyBuildOptions().toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b")));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(options.getStarlarkOptions())
        .containsEntry(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true);
  }

  @Test
  public void trimFlagValues_overwritesRequestedNonFeatureFlagValueWithDefaultIfUntrimmed()
      throws Exception {
    BuildOptions options =
        emptyBuildOptions().toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b"),
                Label.parseCanonicalUnchecked("//unrelated/starlark:option")));

    assertThat(options.getStarlarkOptions())
        .containsEntry(
            Label.parseCanonicalUnchecked("//unrelated/starlark:option"),
            FeatureFlagValue.DefaultValue.INSTANCE);
  }

  @Test
  public void trimFlagValues_overwritesRequestedNonFeatureFlagValueWithUnknownIfTrimmed()
      throws Exception {
    BuildOptions options =
        emptyBuildOptions().toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseCanonicalUnchecked("//label:a"),
                "value",
                Label.parseCanonicalUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b")));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseCanonicalUnchecked("//label:a"),
                Label.parseCanonicalUnchecked("//label:b"),
                Label.parseCanonicalUnchecked("//unrelated/starlark:option")));

    assertThat(options.getStarlarkOptions())
        .containsEntry(
            Label.parseCanonicalUnchecked("//unrelated/starlark:option"),
            FeatureFlagValue.UnknownValue.INSTANCE);
  }
}
