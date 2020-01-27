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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag configuration fragments. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagConfigurationTest {
  @Test
  public void getFeatureFlagValue_returnsValueOfFlagWhenRequestingSetFlag() throws Exception {
    Label ruleLabel = Label.parseAbsoluteUnchecked("//a:a");
    Optional<String> flagValue =
        getConfigurationWithFlags(ImmutableMap.of(ruleLabel, "valued"))
            .getFeatureFlagValue(new LabelArtifactOwner(ruleLabel));
    assertThat(flagValue).isPresent();
    assertThat(flagValue).hasValue("valued");
  }

  @Test
  public void getFeatureFlagValue_returnsEmptyOptionalWhenRequestingFlagNotInInput()
      throws Exception {
    Optional<String> flagValue =
        getConfigurationWithFlags(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "valued"))
            .getFeatureFlagValue(new LabelArtifactOwner(Label.parseAbsoluteUnchecked("//b:b")));
    assertThat(flagValue).isEmpty();
  }

  @Test
  public void getOutputDirectoryName_returnsNullWhenFlagMapIsEmpty() throws Exception {
    assertThat(getConfigurationWithFlags(ImmutableMap.<Label, String>of()).getOutputDirectoryName())
        .isNull();
  }

  @Test
  public void getOutputDirectoryName_returnsNonNullWhenFlagMapIsNonEmpty() throws Exception {
    assertThat(
            getConfigurationWithFlags(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "ok"))
                .getOutputDirectoryName())
        .isNotNull();
  }

  @Test
  public void getOutputDirectoryName_returnsSameValueForTwoMapsWithSamePairsRegardlessOfOrder()
      throws Exception {
    Map<Label, String> firstOrder =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//b:b"), "first",
            Label.parseAbsoluteUnchecked("//a:a"), "second");
    Map<Label, String> reverseOrder =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "second",
            Label.parseAbsoluteUnchecked("//b:b"), "first");
    assertThat(getConfigurationWithFlags(reverseOrder).getOutputDirectoryName())
        .isEqualTo(getConfigurationWithFlags(firstOrder).getOutputDirectoryName());
  }

  @Test
  public void getOutputDirectoryName_returnsDifferentValueForDifferentFlags() throws Exception {
    Map<Label, String> someFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "first",
            Label.parseAbsoluteUnchecked("//b:b"), "second");
    Map<Label, String> otherFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//c:c"), "first",
            Label.parseAbsoluteUnchecked("//d:d"), "second");
    assertThat(getConfigurationWithFlags(otherFlags).getOutputDirectoryName())
        .isNotEqualTo(getConfigurationWithFlags(someFlags).getOutputDirectoryName());
  }

  @Test
  public void getOutputDirectoryName_returnsDifferentValueForDifferentValues() throws Exception {
    Map<Label, String> someFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "first",
            Label.parseAbsoluteUnchecked("//b:b"), "second");
    Map<Label, String> otherFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "worst",
            Label.parseAbsoluteUnchecked("//b:b"), "heckin");
    assertThat(getConfigurationWithFlags(otherFlags).getOutputDirectoryName())
        .isNotEqualTo(getConfigurationWithFlags(someFlags).getOutputDirectoryName());
  }

  @Test
  public void getOutputDirectoryName_differentiatesLabelAndValue() throws Exception {
    Map<Label, String> someFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "firestarter",
            Label.parseAbsoluteUnchecked("//b:b"), "second");
    Map<Label, String> otherFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:afire"), "starter",
            Label.parseAbsoluteUnchecked("//b:b"), "second");
    assertThat(getConfigurationWithFlags(otherFlags).getOutputDirectoryName())
        .isNotEqualTo(getConfigurationWithFlags(someFlags).getOutputDirectoryName());
  }

  @Test
  public void getOutputDirectoryName_returnsDifferentValueForSubsetOfFlagValuePairs()
      throws Exception {
    Map<Label, String> someFlags = ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "first");
    Map<Label, String> moreFlags =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//a:a"), "first",
            Label.parseAbsoluteUnchecked("//b:b"), "second");
    assertThat(getConfigurationWithFlags(moreFlags).getOutputDirectoryName())
        .isNotEqualTo(getConfigurationWithFlags(someFlags).getOutputDirectoryName());
  }

  /** Generates a configuration fragment with the given set of flag-value pairs. */
  private static ConfigFeatureFlagConfiguration getConfigurationWithFlags(
      Map<Label, String> flags) {
    return new ConfigFeatureFlagConfiguration(ImmutableSortedMap.copyOf(flags));
  }
}
