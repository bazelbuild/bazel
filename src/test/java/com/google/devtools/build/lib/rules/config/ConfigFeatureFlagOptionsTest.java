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

import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag option fragments. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagOptionsTest {

  @Test
  public void flagState_startsEmpty() throws Exception {
    assertThat(new ConfigFeatureFlagOptions().getFlagValues()).isEmpty();
    assertThat(new ConfigFeatureFlagOptions().getKnownDefaultFlags()).isEmpty();
    assertThat(new ConfigFeatureFlagOptions().getUnknownFlags()).isEmpty();
  }

  @Test
  public void replaceFlagValues_reflectedInGetFlagValues() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(originalMap);
    assertThat(options.getFlagValues()).containsExactlyEntriesIn(originalMap);
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
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(originalMap);
    options.replaceFlagValues(newMap);
    assertThat(options.getFlagValues()).containsExactlyEntriesIn(newMap);
  }

  @Test
  public void replaceFlagValues_emptiesKnownDefaultFlagsAndUnknownFlags() throws Exception {
    Map<Label, String> originalMap =
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"), "value",
            Label.parseAbsoluteUnchecked("//label:b"), "otherValue");
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(originalMap);
    options.trimFlagValues(
        ImmutableSortedSet.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:c"),
            Label.parseAbsoluteUnchecked("//label:d")));
    options.replaceFlagValues(originalMap);
    assertThat(options.getKnownDefaultFlags()).isEmpty();
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_defaults_toEmptySetProducesEmptyOptions() throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();

    options.trimFlagValues(ImmutableSet.of());

    assertThat(options.getFlagValues()).isEmpty();
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get()).isEmpty();
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_defaults_toPopulatedSetPopulatesKnownDefaultFlags() throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();

    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c")));

    assertThat(options.getFlagValues()).isEmpty();
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get())
        .containsExactly(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c"));
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_withFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            "value",
            Label.parseAbsoluteUnchecked("//label:d"),
            "otherValue"));

    options.trimFlagValues(ImmutableSet.of());

    assertThat(options.getFlagValues()).isEmpty();
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get()).isEmpty();
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_withFlagsSet_toPopulatedSetPopulatesFlagValuesAndKnownDefaultFlags()
      throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            "value",
            Label.parseAbsoluteUnchecked("//label:d"),
            "otherValue"));

    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c")));

    assertThat(options.getFlagValues())
        .containsExactly(Label.parseAbsoluteUnchecked("//label:a"), "value");
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get())
        .containsExactly(
            Label.parseAbsoluteUnchecked("//label:b"), Label.parseAbsoluteUnchecked("//label:c"));
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_withTrimmedFlagsSet_toEmptySetProducesEmptyOptions() throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            "value",
            Label.parseAbsoluteUnchecked("//label:d"),
            "otherValue"));
    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"), Label.parseAbsoluteUnchecked("//label:b")));

    options.trimFlagValues(ImmutableSet.of());

    assertThat(options.getFlagValues()).isEmpty();
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get()).isEmpty();
    assertThat(options.getUnknownFlags()).isEmpty();
  }

  @Test
  public void trimFlagValues_withTrimmedFlagsSet_toPopulatedSetPopulatesFlagState()
      throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.replaceFlagValues(
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            "value",
            Label.parseAbsoluteUnchecked("//label:d"),
            "otherValue"));
    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"), Label.parseAbsoluteUnchecked("//label:b")));

    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c")));

    assertThat(options.getFlagValues())
        .containsExactly(Label.parseAbsoluteUnchecked("//label:a"), "value");
    assertThat(options.getKnownDefaultFlags()).isPresent();
    assertThat(options.getKnownDefaultFlags().get())
        .containsExactly(Label.parseAbsoluteUnchecked("//label:b"));
    assertThat(options.getUnknownFlags())
        .containsExactly(Label.parseAbsoluteUnchecked("//label:c"));
  }

  @Test
  public void hostMode_clearsFlagState() throws Exception {
    ConfigFeatureFlagOptions options = new ConfigFeatureFlagOptions();
    options.enforceTransitiveConfigsForConfigFeatureFlag = true;
    options.replaceFlagValues(
        ImmutableMap.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            "value",
            Label.parseAbsoluteUnchecked("//label:d"),
            "otherValue"));
    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"), Label.parseAbsoluteUnchecked("//label:b")));
    options.trimFlagValues(
        ImmutableSet.of(
            Label.parseAbsoluteUnchecked("//label:a"),
            Label.parseAbsoluteUnchecked("//label:b"),
            Label.parseAbsoluteUnchecked("//label:c")));

    ConfigFeatureFlagOptions hostOptions = (ConfigFeatureFlagOptions) options.getHost();
    assertThat(hostOptions.enforceTransitiveConfigsForConfigFeatureFlag).isFalse();
    assertThat(hostOptions.getFlagValues()).isEmpty();
    assertThat(hostOptions.getKnownDefaultFlags()).isEmpty();
    assertThat(hostOptions.getUnknownFlags()).isEmpty();
  }

  @Test
  public void equals_forEquivalentMaps() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            // Empty with all flags set to default values
            getOptionsWith(ImmutableMap.<Label, String>of()),
            new ConfigFeatureFlagOptions(),
            new ConfigFeatureFlagOptions().getDefault(),
            new ConfigFeatureFlagOptions().getHost(),
            new ConfigFeatureFlagOptions().getHost())
        .addEqualityGroup(
            // Empty with all flags trimmed
            getOptionsAndTrim(ImmutableMap.of(), ImmutableSet.of()),
            getOptionsAndTrim(
                ImmutableMap.of(Label.parseAbsolute("//a:a"), "a"), ImmutableSet.of()),
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//a:a"), "a", Label.parseAbsolute("//b:b"), "b"),
                ImmutableSet.of()))
        .addEqualityGroup(
            // Only //a:a => a, others default
            getOptionsWith(ImmutableMap.of(Label.parseAbsolute("//a:a"), "a")),
            getOptionsWith(ImmutableMap.of(Label.parseAbsolute("//a:a"), "a")))
        .addEqualityGroup(
            // Error: //a:a is absent
            getOptionsAndTrim(ImmutableMap.of(), ImmutableSet.of(Label.parseAbsolute("//a:a"))),
            getOptionsAndTrim(
                ImmutableMap.of(Label.parseAbsolute("//b:b"), "b"),
                ImmutableSet.of(Label.parseAbsolute("//a:a"))))
        .addEqualityGroup(
            // Only //a:a => a, others trimmed
            getOptionsAndTrim(
                ImmutableMap.of(Label.parseAbsolute("//a:a"), "a"),
                ImmutableSet.of(Label.parseAbsolute("//a:a"))),
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//a:a"), "a", Label.parseAbsolute("//b:b"), "b"),
                ImmutableSet.of(Label.parseAbsolute("//a:a"))))
        .addEqualityGroup(
            // Only //b:b => a, others default
            getOptionsWith(ImmutableMap.of(Label.parseAbsolute("//b:b"), "a")))
        .addEqualityGroup(
            // Only //a:a => b, others default
            getOptionsWith(ImmutableMap.of(Label.parseAbsolute("//a:a"), "b")))
        .addEqualityGroup(
            // Only //b:b => b, others default
            getOptionsWith(ImmutableMap.of(Label.parseAbsolute("//b:b"), "b")))
        .addEqualityGroup(
            // //a:a => b and //b:b => a, others default (order doesn't matter)
            getOptionsWith(
                ImmutableMap.of(
                    Label.parseAbsolute("//a:a"), "b",
                    Label.parseAbsolute("//b:b"), "a")),
            getOptionsWith(
                ImmutableMap.of(
                    Label.parseAbsolute("//b:b"), "a",
                    Label.parseAbsolute("//a:a"), "b")))
        .addEqualityGroup(
            // //a:a => b and //b:b => a, others trimmed (order doesn't matter)
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//a:a"), "b",
                    Label.parseAbsolute("//b:b"), "a"),
                ImmutableSet.of(Label.parseAbsolute("//a:a"), Label.parseAbsolute("//b:b"))),
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//a:a"), "b",
                    Label.parseAbsolute("//b:b"), "a"),
                ImmutableSet.of(Label.parseAbsolute("//b:b"), Label.parseAbsolute("//a:a"))),
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//b:b"), "a",
                    Label.parseAbsolute("//a:a"), "b"),
                ImmutableSet.of(Label.parseAbsolute("//a:a"), Label.parseAbsolute("//b:b"))),
            getOptionsAndTrim(
                ImmutableMap.of(
                    Label.parseAbsolute("//b:b"), "a",
                    Label.parseAbsolute("//a:a"), "b"),
                ImmutableSet.of(Label.parseAbsolute("//b:b"), Label.parseAbsolute("//a:a"))))
        .testEquals();
  }

  private ConfigFeatureFlagOptions getOptionsWith(Map<Label, String> values) {
    ConfigFeatureFlagOptions result = new ConfigFeatureFlagOptions();
    result.replaceFlagValues(values);
    return result;
  }

  private ConfigFeatureFlagOptions getOptionsAndTrim(
      Map<Label, String> values, Set<Label> trimming) {
    ConfigFeatureFlagOptions result = getOptionsWith(values);
    result.trimFlagValues(trimming);
    return result;
  }

  @Test
  public void parser_doesNotAllowFlagValuesToBeParsed() throws Exception {
    ConfigFeatureFlagOptions options = Options.getDefaults(ConfigFeatureFlagOptions.class);
    ImmutableSortedMap<Label, String> testValue =
        ImmutableSortedMap.of(Label.parseAbsolute("//what:heck"), "something");
    options.flagValues = testValue;
    String flagValuesOption =
        options
            .asMap()
            .entrySet()
            .stream()
            .filter((entry) -> testValue.equals(entry.getValue()))
            .map(Map.Entry::getKey)
            .collect(onlyElement());
    OptionsParser parser = OptionsParser.newOptionsParser(ConfigFeatureFlagOptions.class);
    try {
      parser.parse("--" + flagValuesOption + "={}");
      fail("Flags successfully parsed despite passing a private flag.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("Unrecognized option:");
    }
  }

  @Test
  public void parser_doesNotAllowKnownDefaultValuesToBeParsed() throws Exception {
    ConfigFeatureFlagOptions options = Options.getDefaults(ConfigFeatureFlagOptions.class);
    ImmutableSortedSet<Label> testValue = ImmutableSortedSet.of(Label.parseAbsolute("//what:heck"));
    options.knownDefaultFlags = testValue;
    String defaultValuesOption =
        options
            .asMap()
            .entrySet()
            .stream()
            .filter((entry) -> testValue.equals(entry.getValue()))
            .map(Map.Entry::getKey)
            .collect(onlyElement());
    OptionsParser parser = OptionsParser.newOptionsParser(ConfigFeatureFlagOptions.class);
    try {
      parser.parse("--" + defaultValuesOption + "={}");
      fail("Flags successfully parsed despite passing a private flag.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("Unrecognized option:");
    }
  }

  @Test
  public void parser_doesNotAllowUnknownValuesToBeParsed() throws Exception {
    ConfigFeatureFlagOptions options = Options.getDefaults(ConfigFeatureFlagOptions.class);
    ImmutableSortedSet<Label> testValue = ImmutableSortedSet.of(Label.parseAbsolute("//what:heck"));
    options.unknownFlags = testValue;
    String unknownFlagsOption =
        options
            .asMap()
            .entrySet()
            .stream()
            .filter((entry) -> testValue.equals(entry.getValue()))
            .map(Map.Entry::getKey)
            .collect(onlyElement());
    OptionsParser parser = OptionsParser.newOptionsParser(ConfigFeatureFlagOptions.class);
    try {
      parser.parse("--" + unknownFlagsOption + "={}");
      fail("Flags successfully parsed despite passing a private flag.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("Unrecognized option:");
    }
  }
}
