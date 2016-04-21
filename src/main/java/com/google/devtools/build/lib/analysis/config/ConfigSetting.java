// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation for the config_setting rule.
 *
 * <p>This is a "pseudo-rule" in that its purpose isn't to generate output artifacts
 * from input artifacts. Rather, it provides configuration context to rules that
 * depend on it.
 */
public class ConfigSetting implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    // Get the required flag=value settings for this rule.
    Map<String, String> settings = NonconfigurableAttributeMapper.of(ruleContext.getRule())
        .get(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE, Type.STRING_DICT);
    if (settings.isEmpty()) {
      ruleContext.attributeError(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
          "no settings specified");
      return null;
    }

    ConfigMatchingProvider configMatcher;
    try {
      configMatcher = new ConfigMatchingProvider(ruleContext.getLabel(), settings,
          matchesConfig(settings, ruleContext.getConfiguration()));
    } catch (OptionsParsingException e) {
      ruleContext.attributeError(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
          "error while parsing configuration settings: " + e.getMessage());
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .add(FileProvider.class, new FileProvider(ruleContext.getLabel(),
            NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)))
        .add(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .add(ConfigMatchingProvider.class, configMatcher)
        .build();
  }

  /**
   * Given a list of [flagName, flagValue] pairs, returns true if flagName == flagValue for
   * every item in the list under this configuration, false otherwise.
   */
  private boolean matchesConfig(Map<String, String> expectedSettings, BuildConfiguration config)
      throws OptionsParsingException {
    // Rather than returning fast when we find a mismatch, continue looking at the other flags
    // to check that they're indeed valid flag specifications.
    boolean foundMismatch = false;

    // Since OptionsParser instantiation involves reflection, let's try to minimize that happening.
    Map<Class<? extends OptionsBase>, OptionsParser> parserCache = new HashMap<>();

    for (Map.Entry<String, String> setting : expectedSettings.entrySet()) {
      String optionName = setting.getKey();
      String expectedRawValue = setting.getValue();

      Class<? extends OptionsBase> optionClass = config.getOptionClass(optionName);
      if (optionClass == null) {
        throw new OptionsParsingException("unknown option: '" + optionName + "'");
      }

      OptionsParser parser = parserCache.get(optionClass);
      if (parser == null) {
        parser = OptionsParser.newOptionsParser(optionClass);
        parserCache.put(optionClass, parser);
      }
      parser.parse("--" + optionName + "=" + expectedRawValue);
      Object expectedParsedValue = parser.getOptions(optionClass).asMap().get(optionName);

      if (!optionMatches(config, optionName, expectedParsedValue)) {
        foundMismatch = true;
      }
    }
    return !foundMismatch;
  }

  /**
   * For single-value options, returns true iff the option's value matches the expected value.
   *
   * <p>For multi-value List options, returns true iff any of the option's values matches
   * the expected value. This means, e.g. "--tool_tag=foo --tool_tag=bar" would match the
   * expected condition { 'tool_tag': 'bar' }.
   *
   * <p>For multi-value Map options, returns true iff the last instance with the same key as the
   * expected key has the same value. This means, e.g. "--define foo=1 --define bar=2" would
   * match { 'define': 'foo=1' }, but "--define foo=1 --define bar=2 --define foo=3" would not
   * match. Note that the definition of --define states that the last instance takes precedence.
   */
  private static boolean optionMatches(BuildConfiguration config, String optionName,
      Object expectedValue) {
    Object actualValue = config.getOptionValue(optionName);
    if (actualValue == null) {
      return expectedValue == null;

    // Single-value case:
    } else if (!config.allowsMultipleValues(optionName)) {
      return actualValue.equals(expectedValue);
    }

    // Multi-value case:
    Preconditions.checkState(actualValue instanceof List);
    Preconditions.checkState(expectedValue instanceof List);
    List<?> actualList = (List<?>) actualValue;
    List<?> expectedList = (List<?>) expectedValue;

    if (actualList.isEmpty() || expectedList.isEmpty()) {
      return actualList.isEmpty() && expectedList.isEmpty();
    }

    // We're expecting a single value of a multi-value type: the options parser still embeds
    // that single value within a List container. Retrieve it here.
    Object expectedSingleValue = Iterables.getOnlyElement(expectedList);

    // Multi-value map:
    if (actualList.get(0) instanceof Map.Entry) {
      Map.Entry<?, ?> expectedEntry = (Map.Entry<?, ?>) expectedSingleValue;
      for (Map.Entry<?, ?> actualEntry : Lists.reverse((List<Map.Entry<?, ?>>) actualList)) {
        if (actualEntry.getKey().equals(expectedEntry.getKey())) {
          // Found a key match!
          return actualEntry.getValue().equals(expectedEntry.getValue());
        }
      }
      return false; // Never found any matching key.
    }

    // Multi-value list:
    return actualList.contains(expectedSingleValue);
  }
}
