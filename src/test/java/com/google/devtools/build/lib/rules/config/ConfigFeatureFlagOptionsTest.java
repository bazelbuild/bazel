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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for feature flag option fragments. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagOptionsTest {

  @Test
  public void getFlagValues_startsEmpty() throws Exception {
    assertThat(new ConfigFeatureFlagOptions().getFlagValues()).isEmpty();
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
  public void getDefault_isEmpty() throws Exception {
    assertThat(
            ((ConfigFeatureFlagOptions) new ConfigFeatureFlagOptions().getDefault())
                .getFlagValues())
        .isEmpty();
  }

  @Test
  public void getHost_isEmpty() throws Exception {
    assertThat(
            ((ConfigFeatureFlagOptions) new ConfigFeatureFlagOptions().getHost()).getFlagValues())
        .isEmpty();
    assertThat(
            ((ConfigFeatureFlagOptions) new ConfigFeatureFlagOptions().getHost()).getFlagValues())
        .isEmpty();
  }

  @Test
  public void equals_forEquivalentMaps() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            getOptionsWith(ImmutableMap.<Label, String>of()),
            new ConfigFeatureFlagOptions(),
            new ConfigFeatureFlagOptions().getDefault(),
            new ConfigFeatureFlagOptions().getHost(),
            new ConfigFeatureFlagOptions().getHost())
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
  public void parser_doesNotAllowFlagValuesToBeParsed() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ConfigFeatureFlagOptions.class);
    try {
      parser.parse(
          "--" + Iterables.getOnlyElement(new ConfigFeatureFlagOptions().asMap().keySet()) + "={}");
      fail("Flags successfully parsed despite passing a private flag.");
    } catch (OptionsParsingException expected) {
      assertThat(expected).hasMessageThat().contains("Unrecognized option:");
    }
  }

  private ConfigFeatureFlagOptions getOptionsWith(Map<Label, String> values) {
    ConfigFeatureFlagOptions result = new ConfigFeatureFlagOptions();
    result.replaceFlagValues(values);
    return result;
  }
}
