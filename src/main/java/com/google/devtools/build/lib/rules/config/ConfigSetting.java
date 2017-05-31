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

package com.google.devtools.build.lib.rules.config;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationOptionDetails;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.TransitiveOptionDetails;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.AliasProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    // Get the required flag=value settings for this rule.
    Map<String, String> settings = NonconfigurableAttributeMapper.of(ruleContext.getRule())
        .get(ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE, Type.STRING_DICT);
    Map<Label, String> flagSettings =
        NonconfigurableAttributeMapper.of(ruleContext.getRule())
            .get(
                ConfigRuleClasses.ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
                BuildType.LABEL_KEYED_STRING_DICT);

    List<? extends TransitiveInfoCollection> flagValues =
        ruleContext.getPrerequisites(
            ConfigRuleClasses.ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE, Mode.TARGET);

    if (settings.isEmpty() && flagSettings.isEmpty()) {
      ruleContext.ruleError(
          String.format(
              "Either %s or %s must be specified and non-empty",
              ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
              ConfigRuleClasses.ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE));
      return null;
    }

    ConfigFeatureFlagMatch featureFlags =
        ConfigFeatureFlagMatch.fromAttributeValueAndPrerequisites(
            flagSettings, flagValues, (RuleErrorConsumer) ruleContext);

    boolean settingsMatch =
        matchesConfig(
            settings,
            BuildConfigurationOptionDetails.get(ruleContext.getConfiguration()),
            ruleContext);

    if (ruleContext.hasErrors()) {
      return null;
    }

    ConfigMatchingProvider configMatcher =
        new ConfigMatchingProvider(
            ruleContext.getLabel(),
            settings,
            featureFlags.getSpecifiedFlagValues(),
            settingsMatch && featureFlags.matches());

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addProvider(LicensesProviderImpl.EMPTY)
        .addProvider(ConfigMatchingProvider.class, configMatcher)
        .build();
  }

  /**
   * Given a list of [flagName, flagValue] pairs, returns true if flagName == flagValue for every
   * item in the list under this configuration, false otherwise.
   */
  private boolean matchesConfig(
      Map<String, String> expectedSettings,
      TransitiveOptionDetails options,
      RuleErrorConsumer errors) {
    // Rather than returning fast when we find a mismatch, continue looking at the other flags
    // to check that they're indeed valid flag specifications.
    boolean foundMismatch = false;

    // Since OptionsParser instantiation involves reflection, let's try to minimize that happening.
    Map<Class<? extends OptionsBase>, OptionsParser> parserCache = new HashMap<>();

    for (Map.Entry<String, String> setting : expectedSettings.entrySet()) {
      String optionName = setting.getKey();
      String expectedRawValue = setting.getValue();

      Class<? extends OptionsBase> optionClass = options.getOptionClass(optionName);
      if (optionClass == null) {
        errors.attributeError(
            ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
            String.format(
                "error while parsing configuration settings: unknown option: '%s'", optionName));
        foundMismatch = true;
        continue;
      }

      OptionsParser parser = parserCache.get(optionClass);
      if (parser == null) {
        parser = OptionsParser.newOptionsParser(optionClass);
        parserCache.put(optionClass, parser);
      }

      try {
        parser.parse("--" + optionName + "=" + expectedRawValue);
      } catch (OptionsParsingException ex) {
        errors.attributeError(
            ConfigRuleClasses.ConfigSettingRule.SETTINGS_ATTRIBUTE,
            "error while parsing configuration settings: " + ex.getMessage());
        foundMismatch = true;
        continue;
      }

      Object expectedParsedValue = parser.getOptions(optionClass).asMap().get(optionName);

      if (!optionMatches(options, optionName, expectedParsedValue)) {
        foundMismatch = true;
      }
    }
    return !foundMismatch;
  }

  /**
   * For single-value options, returns true iff the option's value matches the expected value.
   *
   * <p>For multi-value List options, returns true iff any of the option's values matches the
   * expected value. This means, e.g. "--tool_tag=foo --tool_tag=bar" would match the expected
   * condition { 'tool_tag': 'bar' }.
   *
   * <p>For multi-value Map options, returns true iff the last instance with the same key as the
   * expected key has the same value. This means, e.g. "--define foo=1 --define bar=2" would match {
   * 'define': 'foo=1' }, but "--define foo=1 --define bar=2 --define foo=3" would not match. Note
   * that the definition of --define states that the last instance takes precedence.
   */
  private static boolean optionMatches(
      TransitiveOptionDetails options, String optionName, Object expectedValue) {
    Object actualValue = options.getOptionValue(optionName);
    if (actualValue == null) {
      return expectedValue == null;

      // Single-value case:
    } else if (!options.allowsMultipleValues(optionName)) {
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

  private static final class ConfigFeatureFlagMatch {
    private final boolean matches;
    private final ImmutableMap<Label, String> specifiedFlagValues;

    private static final Joiner QUOTED_COMMA_JOINER = Joiner.on("', '");

    private ConfigFeatureFlagMatch(
        boolean matches, ImmutableMap<Label, String> specifiedFlagValues) {
      this.matches = matches;
      this.specifiedFlagValues = specifiedFlagValues;
    }

    /** Returns whether the specified flag values matched the actual flag values. */
    public boolean matches() {
      return matches;
    }

    /** Gets the specified flag values, with aliases converted to their original targets' labels. */
    public ImmutableMap<Label, String> getSpecifiedFlagValues() {
      return specifiedFlagValues;
    }

    /** Groups aliases in the list of prerequisites by the target they point to. */
    private static ListMultimap<Label, Label> collectAliases(
        Iterable<? extends TransitiveInfoCollection> prerequisites) {
      ImmutableListMultimap.Builder<Label, Label> targetsToAliases =
          new ImmutableListMultimap.Builder<>();
      for (TransitiveInfoCollection target : prerequisites) {
        targetsToAliases.put(target.getLabel(), AliasProvider.getDependencyLabel(target));
      }
      return targetsToAliases.build();
    }

    public static ConfigFeatureFlagMatch fromAttributeValueAndPrerequisites(
        Map<Label, String> attributeValue,
        Iterable<? extends TransitiveInfoCollection> prerequisites,
        RuleErrorConsumer errors) {
      Map<Label, String> specifiedFlagValues = new LinkedHashMap<>();
      boolean matches = true;
      boolean foundDuplicate = false;

      for (TransitiveInfoCollection target : prerequisites) {
        ConfigFeatureFlagProvider provider = ConfigFeatureFlagProvider.fromTarget(target);
        // We know the provider exists because only labels with ConfigFeatureFlagProvider can be
        // added to this attribute.
        assert provider != null;

        Label actualLabel = target.getLabel();
        Label specifiedLabel = AliasProvider.getDependencyLabel(target);
        String specifiedValue = attributeValue.get(specifiedLabel);
        if (specifiedFlagValues.containsKey(actualLabel)) {
          foundDuplicate = true;
        }
        specifiedFlagValues.put(actualLabel, specifiedValue);

        if (!provider.isValidValue(specifiedValue)) {
          errors.attributeError(
              ConfigRuleClasses.ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
              String.format(
                  "error while parsing user-defined configuration values: "
                      + "'%s' is not a valid value for '%s'",
                  specifiedValue, specifiedLabel));
          matches = false;
          continue;
        }
        if (!provider.getValue().equals(specifiedValue)) {
          matches = false;
        }
      }

      // attributeValue is the source of the prerequisites in prerequisites, so the final map built
      // from iterating over prerequisites should always be the same size, barring duplicates.
      assert foundDuplicate || attributeValue.size() == specifiedFlagValues.size();

      if (foundDuplicate) {
        ListMultimap<Label, Label> aliases = collectAliases(prerequisites);
        for (Label actualLabel : aliases.keySet()) {
          List<Label> aliasList = aliases.get(actualLabel);
          if (aliasList.size() > 1) {
            errors.attributeError(
                ConfigRuleClasses.ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
                String.format(
                    "flag '%s' referenced multiple times as ['%s']",
                    actualLabel, QUOTED_COMMA_JOINER.join(aliasList)));
          }
        }

        matches = false;
      }

      return new ConfigFeatureFlagMatch(matches, ImmutableMap.copyOf(specifiedFlagValues));
    }
  }
}
