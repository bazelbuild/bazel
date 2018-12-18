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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag options data. */
@RunWith(JUnit4.class)
public final class FeatureFlagValueTest {

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
  public void flagState_startsEmpty() throws Exception {
    assertThat(FeatureFlagValue.getFlagValues(emptyBuildOptions())).isEmpty();
  }

  @Test
  public void flagState_ignoresNonFeatureFlagValues() throws Exception {
    BuildOptions options =
        emptyBuildOptions()
            .toBuilder()
            .addStarlarkOption(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true)
            .build();
    assertThat(FeatureFlagValue.getFlagValues(options)).isEmpty();
  }

  @Test
  public void replaceFlagValues_reflectedInGetFlagValues() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    BuildOptions options = FeatureFlagValue.replaceFlagValues(emptyBuildOptions(), originalMap);
    assertThat(FeatureFlagValue.getFlagValues(options)).containsExactlyEntriesIn(originalMap);
  }

  @Test
  public void replaceFlagValues_totallyReplacesFlagValuesMap() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    Map<Label, String> newMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "differentValue",
            Label.parseAbsoluteUnchecked("//label:c"), "differentFlag");
    BuildOptions options = emptyBuildOptions();
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    options = FeatureFlagValue.replaceFlagValues(options, newMap);
    assertThat(FeatureFlagValue.getFlagValues(options)).containsExactlyEntriesIn(newMap);
  }

  @Test
  public void replaceFlagValues_emptiesKnownDefaultFlagsAndUnknownFlags() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    BuildOptions options = emptyBuildOptions();
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSortedSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:c"),
                Label.parseAbsoluteUnchecked("//label:d")));
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent).isTrue();
    // testing that this does not throw UnknownValueException
    FeatureFlagValue.getFlagValues(options);
  }

  @Test
  public void replaceFlagValues_leavesNonFeatureFlagValuesAlone() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    Map<Label, String> newMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "differentValue",
            Label.parseAbsoluteUnchecked("//label:c"), "differentFlag");
    BuildOptions options =
        emptyBuildOptions()
            .toBuilder()
            .addStarlarkOption(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true)
            .build();
    options = FeatureFlagValue.replaceFlagValues(options, originalMap);
    options = FeatureFlagValue.replaceFlagValues(options, newMap);
    assertThat(options.getStarlarkOptions())
        .containsEntry(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true);
  }

  @Test
  public void trimFlagValues_defaults_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(FeatureFlagValue.getFlagValues(options)).isEmpty();
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
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//label:c")));

    assertThat(FeatureFlagValue.getFlagValues(options)).isEmpty();
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c"));
  }

  @Test
  public void trimFlagValues_withFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(FeatureFlagValue.getFlagValues(options)).isEmpty();
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
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));

    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//label:c")));

    assertThat(FeatureFlagValue.getFlagValues(options))
        .containsExactly(Label.parseAbsoluteUnchecked("//label:a"), "value");
    assertThat(options.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
        .isFalse();
    assertThat(getKnownDefaultFlags(options))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//label:b"), Label.parseAbsoluteUnchecked("//label:c"));
  }

  @Test
  public void trimFlagValues_withTrimmedFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b")));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(FeatureFlagValue.getFlagValues(options)).isEmpty();
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
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b")));

    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//label:c")));

    final BuildOptions testedOptions = options;

    FeatureFlagValue.UnknownValueException unknownValueException =
        MoreAsserts.assertThrows(
            FeatureFlagValue.UnknownValueException.class,
            () -> FeatureFlagValue.getFlagValues(testedOptions));
    assertThat(unknownValueException.getUnknownFlags())
        .containsExactly(Label.parseAbsoluteUnchecked("//label:c"));
  }

  @Test
  public void trimFlagValues_leavesNonFeatureFlagValuesAlone() throws Exception {
    BuildOptions options =
        emptyBuildOptions()
            .toBuilder()
            .addStarlarkOption(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b")));

    options = FeatureFlagValue.trimFlagValues(options, ImmutableSet.of());

    assertThat(options.getStarlarkOptions())
        .containsEntry(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true);
  }

  @Test
  public void trimFlagValues_overwritesRequestedNonFeatureFlagValueWithDefaultIfUntrimmed()
      throws Exception {
    BuildOptions options =
        emptyBuildOptions()
            .toBuilder()
            .addStarlarkOption(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//unrelated/starlark:option")));

    assertThat(options.getStarlarkOptions())
        .containsEntry(
            Label.parseAbsoluteUnchecked("//unrelated/starlark:option"),
            FeatureFlagValue.DefaultValue.INSTANCE);
  }

  @Test
  public void trimFlagValues_overwritesRequestedNonFeatureFlagValueWithUnknownIfTrimmed()
      throws Exception {
    BuildOptions options =
        emptyBuildOptions()
            .toBuilder()
            .addStarlarkOption(Label.parseAbsoluteUnchecked("//unrelated/starlark:option"), true)
            .build();
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b")));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//unrelated/starlark:option")));

    assertThat(options.getStarlarkOptions())
        .containsEntry(
            Label.parseAbsoluteUnchecked("//unrelated/starlark:option"),
            FeatureFlagValue.UnknownValue.INSTANCE);
  }

  @Test
  public void hostMode_disablesTrimmingButIsOtherwiseEquivalent() throws Exception {
    BuildOptions options = emptyBuildOptions();
    options.get(ConfigFeatureFlagOptions.class).enforceTransitiveConfigsForConfigFeatureFlag = true;
    options =
        FeatureFlagValue.replaceFlagValues(
            options,
            ImmutableMap.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                "value",
                Label.parseAbsoluteUnchecked("//label:d"),
                "otherValue"));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b")));
    options =
        FeatureFlagValue.trimFlagValues(
            options,
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//label:a"),
                Label.parseAbsoluteUnchecked("//label:b"),
                Label.parseAbsoluteUnchecked("//label:c")));

    BuildOptions hostOptions = options.createHostOptions();
    assertThat(hostOptions).isNotEqualTo(options);
    BuildOptions withTransitiveConfigsDisabled = options.clone();
    withTransitiveConfigsDisabled.get(ConfigFeatureFlagOptions.class)
            .enforceTransitiveConfigsForConfigFeatureFlag =
        false;
    assertThat(hostOptions).isEqualTo(withTransitiveConfigsDisabled);
  }
}
