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
    Label ruleLabel = Label.parseCanonicalUnchecked("//a:a");
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
        getConfigurationWithFlags(ImmutableMap.of(Label.parseCanonicalUnchecked("//a:a"), "valued"))
            .getFeatureFlagValue(new LabelArtifactOwner(Label.parseCanonicalUnchecked("//b:b")));
    assertThat(flagValue).isEmpty();
  }

  /** Generates a configuration fragment with the given set of flag-value pairs. */
  private static ConfigFeatureFlagConfiguration getConfigurationWithFlags(
      Map<Label, String> flags) {
    return new ConfigFeatureFlagConfiguration(ImmutableSortedMap.copyOf(flags));
  }
}
