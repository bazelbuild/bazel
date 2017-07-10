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
import static org.junit.Assert.fail;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag configuration fragments. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagConfigurationTest {

  @Test
  public void options_getFlagValues_startsEmpty() throws Exception {
    assertThat(new ConfigFeatureFlagConfiguration.Options().getFlagValues()).isEmpty();
  }

  @Test
  public void options_replaceFlagValues_reflectedInGetFlagValues() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    ConfigFeatureFlagConfiguration.Options options = new ConfigFeatureFlagConfiguration.Options();
    options.replaceFlagValues(originalMap);
    assertThat(options.getFlagValues()).containsExactlyEntriesIn(originalMap);
  }

  @Test
  public void options_replaceFlagValues_totallyReplacesFlagValuesMap() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    Map<Label, String> newMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "differentValue",
            Label.parseAbsoluteUnchecked("//label:c"), "differentFlag");
    ConfigFeatureFlagConfiguration.Options options = new ConfigFeatureFlagConfiguration.Options();
    options.replaceFlagValues(originalMap);
    options.replaceFlagValues(newMap);
    assertThat(options.getFlagValues()).containsExactlyEntriesIn(newMap);
  }

  @Test
  public void options_getDefault_isEmpty() throws Exception {
    assertThat(
            ((ConfigFeatureFlagConfiguration.Options)
                    new ConfigFeatureFlagConfiguration.Options().getDefault())
                .getFlagValues())
        .isEmpty();
  }

  @Test
  public void options_getHost_isEmpty() throws Exception {
    assertThat(
            ((ConfigFeatureFlagConfiguration.Options)
                    new ConfigFeatureFlagConfiguration.Options().getHost(false))
                .getFlagValues())
        .isEmpty();
    assertThat(
            ((ConfigFeatureFlagConfiguration.Options)
                    new ConfigFeatureFlagConfiguration.Options().getHost(true))
                .getFlagValues())
        .isEmpty();
  }

  @Test
  public void options_equals_forEquivalentMaps() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.<Label, String>of()),
            new ConfigFeatureFlagConfiguration.Options(),
            new ConfigFeatureFlagConfiguration.Options().getDefault(),
            new ConfigFeatureFlagConfiguration.Options().getHost(false),
            new ConfigFeatureFlagConfiguration.Options().getHost(true))
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "a")),
            getOptionsWith(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "a")))
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.of(Label.parseAbsoluteUnchecked("//b:b"), "a")))
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "b")))
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.of(Label.parseAbsoluteUnchecked("//b:b"), "b")))
        .addEqualityGroup(
            getOptionsWith(
                ImmutableMap.of(
                    Label.parseAbsoluteUnchecked("//a:a"), "b",
                    Label.parseAbsoluteUnchecked("//b:b"), "a")),
            getOptionsWith(
                ImmutableMap.of(
                    Label.parseAbsoluteUnchecked("//b:b"), "a",
                    Label.parseAbsoluteUnchecked("//a:a"), "b")))
        .testEquals();
  }

  @Test
  public void options_doesNotAllowFlagValuesToBeParsed() throws Exception {
    OptionsParser parser =
        OptionsParser.newOptionsParser(ConfigFeatureFlagConfiguration.Options.class);
    try {
      parser.parse(
          "--"
              + Iterables.getOnlyElement(
                  new ConfigFeatureFlagConfiguration.Options().asMap().keySet())
              + "={}");
      fail("Flags successfully parsed despite passing a private flag.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("Unrecognized option:");
    }
  }

  private ConfigFeatureFlagConfiguration.Options getOptionsWith(Map<Label, String> values) {
    ConfigFeatureFlagConfiguration.Options result = new ConfigFeatureFlagConfiguration.Options();
    result.replaceFlagValues(values);
    return result;
  }

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
  public void getFeatureFlagValue_returnsAbsentOptionalWhenRequestingUnsetFlag() throws Exception {
    Optional<String> flagValue =
        getConfigurationWithFlags(ImmutableMap.of(Label.parseAbsoluteUnchecked("//a:a"), "valued"))
            .getFeatureFlagValue(new LabelArtifactOwner(Label.parseAbsoluteUnchecked("//b:b")));
    assertThat(flagValue).isAbsent();
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
    ConfigFeatureFlagConfiguration.Options options = new ConfigFeatureFlagConfiguration.Options();
    options.replaceFlagValues(flags);
    return new ConfigFeatureFlagConfiguration(options);
  }
}
