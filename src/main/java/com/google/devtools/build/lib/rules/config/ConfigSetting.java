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

import static com.google.devtools.build.lib.analysis.config.CoreOptionConverters.BUILD_SETTING_CONVERTERS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BuildSettingProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationOptionDetails;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions.SelectRestriction;
import com.google.devtools.build.lib.analysis.config.TransitiveOptionDetails;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.config.ConfigRuleClasses.ConfigSettingRule;
import com.google.devtools.build.lib.util.ClassName;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
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
      throws InterruptedException, ActionConflictException {
    AttributeMap attributes = NonconfigurableAttributeMapper.of(ruleContext.getRule());

    // Get the built-in Blaze flag settings that match this rule.
    ImmutableMultimap<String, String> nativeFlagSettings =
        ImmutableMultimap.<String, String>builder()
            .putAll(attributes.get(ConfigSettingRule.SETTINGS_ATTRIBUTE, Type.STRING_DICT)
                .entrySet())
            .putAll(attributes.get(ConfigSettingRule.DEFINE_SETTINGS_ATTRIBUTE, Type.STRING_DICT)
                .entrySet()
                .stream()
                .map(in -> Maps.immutableEntry("define", in.getKey() + "=" + in.getValue()))
                .collect(ImmutableList.toImmutableList()))
            .build();

    // Get the user-defined flag settings that match this rule.
    Map<Label, String> userDefinedFlagSettings =
        attributes.get(
            ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE, BuildType.LABEL_KEYED_STRING_DICT);

    // Get the platform constraint settings that match this rule.
    List<Label> constraintValueSettings =
        attributes.get(ConfigSettingRule.CONSTRAINT_VALUES_ATTRIBUTE, BuildType.LABEL_LIST);

    // Check that this config_setting contains at least one of {values, define_values,
    // constraint_values}
    if (!valuesAreSet(
        nativeFlagSettings, userDefinedFlagSettings, constraintValueSettings, ruleContext)) {
      return null;
    }

    TransitiveOptionDetails optionDetails =
        BuildConfigurationOptionDetails.get(ruleContext.getConfiguration());
    ImmutableSet.Builder<String> requiredFragmentOptions = ImmutableSet.builder();

    boolean nativeFlagsMatch =
        matchesConfig(
            nativeFlagSettings.entries(), optionDetails, requiredFragmentOptions, ruleContext);

    UserDefinedFlagMatch userDefinedFlags =
        UserDefinedFlagMatch.fromAttributeValueAndPrerequisites(
            userDefinedFlagSettings, optionDetails, requiredFragmentOptions, ruleContext);

    boolean constraintValuesMatch = constraintValuesMatch(ruleContext);

    if (ruleContext.hasErrors()) {
      return null;
    }

    // For config_setting, transitive and direct are the same (it has no transitive deps).
    boolean includeRequiredFragments =
        ruleContext
                .getConfiguration()
                .getOptions()
                .get(CoreOptions.class)
                .includeRequiredConfigFragmentsProvider
            != IncludeConfigFragmentsEnum.OFF;

    ConfigMatchingProvider configMatcher =
        new ConfigMatchingProvider(
            ruleContext.getLabel(),
            nativeFlagSettings,
            userDefinedFlags.getSpecifiedFlagValues(),
            includeRequiredFragments ? requiredFragmentOptions.build() : ImmutableSet.<String>of(),
            nativeFlagsMatch && userDefinedFlags.matches() && constraintValuesMatch);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addProvider(LicensesProviderImpl.EMPTY)
        .addProvider(ConfigMatchingProvider.class, configMatcher)
        .build();
  }

  /**
   * Returns true if all <code>constraint_values</code> settings are valid and match this
   * configuration, false otherwise.
   *
   * <p>May generate rule errors on bad settings (e.g. wrong target types).
   */
  boolean constraintValuesMatch(RuleContext ruleContext) {
    List<ConstraintValueInfo> constraintValues = new ArrayList<>();
    for (TransitiveInfoCollection dep :
        ruleContext.getPrerequisites(
            ConfigSettingRule.CONSTRAINT_VALUES_ATTRIBUTE, Mode.DONT_CHECK)) {
      if (!PlatformProviderUtils.hasConstraintValue(dep)) {
        ruleContext.attributeError(
            ConfigSettingRule.CONSTRAINT_VALUES_ATTRIBUTE,
            String.format(dep.getLabel() + " is not a constraint_value"));
      } else {
        constraintValues.add(PlatformProviderUtils.constraintValue(dep));
      }
    }
    if (ruleContext.hasErrors()) {
      return false;
    }

    // The set of constraint_values in a config_setting should never contain multiple
    // constraint_values that map to the same constraint_setting. This method checks if there are
    // duplicates and records an error if so.
    try {
      ConstraintCollection.validateConstraints(constraintValues);
    } catch (ConstraintCollection.DuplicateConstraintException e) {
      ruleContext.ruleError(
          ConstraintCollection.DuplicateConstraintException.formatError(e.duplicateConstraints()));
        return false;
    }

    return ruleContext
        .getToolchainContext()
        .targetPlatform()
        .constraints()
        .containsAll(constraintValues);
  }

  private static RepositoryName getToolsRepository(RuleContext ruleContext) {
    try {
      return RepositoryName.create(
          ruleContext.attributes().get(ConfigSettingRule.TOOLS_REPOSITORY_ATTRIBUTE, Type.STRING));
    } catch (LabelSyntaxException ex) {
      throw new IllegalStateException(ex);
    }
  }

  /**
   * Returns whether the given label falls under the {@code //tools} package (including subpackages)
   * of the tools repository.
   */
  @VisibleForTesting
  static boolean isUnderToolsPackage(Label label, RepositoryName toolsRepository) {
    PackageIdentifier packageId = label.getPackageIdentifier();
    if (!packageId.getRepository().equals(toolsRepository)) {
      return false;
    }
    try {
      return packageId.getPackageFragment().subFragment(0, 1).equals(PathFragment.create("tools"));
    } catch (IndexOutOfBoundsException e) {
      // Top-level package (//).
      return false;
    }
  }

  /**
   * User error when value settings can't be properly parsed.
   */
  private static final String PARSE_ERROR_MESSAGE = "error while parsing configuration settings: ";

  /**
   * Check to make sure this config_setting contains and sets least one of {values, define_values,
   * flag_value or constraint_values}.
   */
  private boolean valuesAreSet(
      ImmutableMultimap<String, String> nativeFlagSettings,
      Map<Label, String> userDefinedFlagSettings,
      Iterable<Label> constraintValues,
      RuleErrorConsumer errors) {
    if (nativeFlagSettings.isEmpty()
        && userDefinedFlagSettings.isEmpty()
        && Iterables.isEmpty(constraintValues)) {
      errors.ruleError(
          String.format(
              "Either %s, %s or %s must be specified and non-empty",
              ConfigSettingRule.SETTINGS_ATTRIBUTE,
              ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
              ConfigSettingRule.CONSTRAINT_VALUES_ATTRIBUTE));
      return false;
    }
    return true;
  }

  /**
   * Given a list of [flagName, flagValue] pairs for native Blaze flags, returns true if flagName ==
   * flagValue for every item in the list under this configuration, false otherwise.
   *
   * <p>This also sets {@code requiredFragmentOptions} to the {@link FragmentOptions} that options
   * read by this {@code config_setting} belong to.
   */
  private static boolean matchesConfig(
      Collection<Map.Entry<String, String>> expectedSettings,
      TransitiveOptionDetails options,
      ImmutableSet.Builder<String> requiredFragmentOptions,
      RuleContext ruleContext) {
    // Rather than returning fast when we find a mismatch, continue looking at the other flags
    // to check they're indeed valid flag specifications.
    boolean foundMismatch = false;

    // Flags that appear multiple times are known as "multi-value options". Each time the options
    // parser parses one of their values it adds it to an existing list. In those cases we need to
    // make sure to examine only the value we just parsed: not the entire list.
    Multiset<String> optionsCount = HashMultiset.create();

    for (Map.Entry<String, String> setting : expectedSettings) {
      String optionName = setting.getKey();
      String expectedRawValue = setting.getValue();
      int previousOptionCount = optionsCount.add(optionName, 1);

      Class<? extends FragmentOptions> optionClass = options.getOptionClass(optionName);
      if (optionClass == null) {
        ruleContext.attributeError(
            ConfigSettingRule.SETTINGS_ATTRIBUTE,
            String.format(PARSE_ERROR_MESSAGE + "unknown option: '%s'", optionName));
        foundMismatch = true;
        continue;
      }

      if (optionName.equals("define")) {
        // --define is more like user-defined build flags than traditional native flags. Report it
        // like user-defined flags: the dependency is directly on the flag vs. the fragment that
        // contains the flag. This frees a rule that depends on "--define a=1" from preserving
        // another rule's dependency on "--define b=2". In other words, if both rules simply said
        // "I require CoreOptions" (which is the FragmentOptions --define belongs to), that would
        // hide the reality that they really have orthogonal dependencies: removing
        // "--define b=2" is perfectly safe for the rule that needs "--define a=1".
        int equalsIndex = expectedRawValue.indexOf('=');
        requiredFragmentOptions.add(
            "--define:"
                + (equalsIndex > 0
                    ? expectedRawValue.substring(0, equalsIndex)
                    : expectedRawValue));
      } else {
        // For other native flags, it's reasonable to report the fragment they belong to.
        requiredFragmentOptions.add(ClassName.getSimpleNameWithOuter(optionClass));
      }

      SelectRestriction selectRestriction = options.getSelectRestriction(optionName);
      if (selectRestriction != null) {
        boolean underToolsPackage =
            isUnderToolsPackage(ruleContext.getRule().getLabel(), getToolsRepository(ruleContext));
        if (!(selectRestriction.isVisibleWithinToolsPackage() && underToolsPackage)) {
          String errorMessage =
              String.format("option '%s' cannot be used in a config_setting", optionName);
          if (selectRestriction.isVisibleWithinToolsPackage()) {
            errorMessage +=
                String.format(
                    " (it is whitelisted to %s//tools/... only)",
                    getToolsRepository(ruleContext).getDefaultCanonicalForm());
          }
          if (selectRestriction.getErrorMessage() != null) {
            errorMessage += ". " + selectRestriction.getErrorMessage();
          }
          ruleContext.attributeError(ConfigSettingRule.SETTINGS_ATTRIBUTE, errorMessage);
          foundMismatch = true;
          continue;
        }
      }

      OptionsParser parser;
      try {
        parser = OptionsParser.builder().optionsClasses(optionClass).build();
        parser.parse("--" + optionName + "=" + expectedRawValue);
      } catch (OptionsParsingException ex) {
        ruleContext.attributeError(
            ConfigSettingRule.SETTINGS_ATTRIBUTE, PARSE_ERROR_MESSAGE + ex.getMessage());
        foundMismatch = true;
        continue;
      }

      Object expectedParsedValue = parser.getOptions(optionClass).asMap().get(optionName);
      if (previousOptionCount > 0) {
        // We've seen this option before, so it's a multi-value option with multiple entries.
        int listLength = ((List<?>) expectedParsedValue).size();
        expectedParsedValue = ((List<?>) expectedParsedValue).subList(listLength - 1, listLength);
      }
      if (!optionMatches(options, optionName, expectedParsedValue)) {
        foundMismatch = true;
      }
    }
    return !foundMismatch;
  }

  /**
   * For single-value options, returns true iff the option's value matches the expected value.
   *
   * <p>For multi-value List options returns true iff any of the option's values matches the
   * expected value(s). This means "--ios_multi_cpus=a --ios_multi_cpus=b --ios_multi_cpus=c"
   * matches the expected conditions {'ios_multi_cpus': 'a' } and { 'ios_multi_cpus': 'b,c' } but
   * not { 'ios_multi_cpus': 'd' }.
   *
   * <p>For multi-value Map options, returns true iff the last instance with the same key as the
   * expected key has the same value. This means "--define foo=1 --define bar=2" matches { 'define':
   * 'foo=1' }, but "--define foo=1 --define bar=2 --define foo=3" doesn't match. Note that the
   * definition of --define states that the last instance takes precedence. Also note that there's
   * no options-parsing support for multiple values in a single clause, e.g. { 'define':
   * 'foo=1,bar=2' } expands to { "foo": "1,bar=2" }, not {"foo": 1, "bar": "2"}.
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

    // Multi-value map:
    if (actualList.get(0) instanceof Map.Entry) {
      // The config_setting's expected value *must* be a single map entry (see method comments).
      Object expectedListValue = Iterables.getOnlyElement(expectedList);
      Map.Entry<?, ?> expectedEntry = (Map.Entry<?, ?>) expectedListValue;
      for (Object elem : Lists.reverse(actualList)) {
        Map.Entry<?, ?> actualEntry = (Map.Entry<?, ?>) elem;
        if (actualEntry.getKey().equals(expectedEntry.getKey())) {
          // Found a key match!
          return actualEntry.getValue().equals(expectedEntry.getValue());
        }
      }
      return false;
    }

    // Multi-value list:
    return actualList.containsAll(expectedList);
  }

  private static final class UserDefinedFlagMatch {
    private final boolean matches;
    private final ImmutableMap<Label, String> specifiedFlagValues;

    private static final Joiner QUOTED_COMMA_JOINER = Joiner.on("', '");

    private UserDefinedFlagMatch(boolean matches, ImmutableMap<Label, String> specifiedFlagValues) {
      this.matches = matches;
      this.specifiedFlagValues = specifiedFlagValues;
    }

    /** Returns whether the specified flag values matched the actual flag values. */
    public boolean matches() {
      return matches;
    }

    /** Gets the specified flag values, with aliases converted to their original targets' labels. */
    ImmutableMap<Label, String> getSpecifiedFlagValues() {
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

    /**
     * The 'flag_values' attribute takes a label->string dictionary of feature flags and
     * starlark-defined settings to their values in string form.
     *
     * @param attributeValue map of user-defined flag labels to their values as set in the
     *     'flag_values' attribute
     * @param optionDetails information about the configuration to match against
     * @param requiredFragmentOptions set of config fragments this config_setting requires. This
     *     method adds feature flag and Starlark-defined setting requirements to this set.
     * @param ruleContext this rule's RuleContext
     */
    static UserDefinedFlagMatch fromAttributeValueAndPrerequisites(
        Map<Label, String> attributeValue,
        TransitiveOptionDetails optionDetails,
        ImmutableSet.Builder<String> requiredFragmentOptions,
        RuleContext ruleContext) {
      Map<Label, String> specifiedFlagValues = new LinkedHashMap<>();
      boolean matches = true;
      boolean foundDuplicate = false;

      // Get the actual targets the 'flag_values' keys reference.
      Iterable<? extends TransitiveInfoCollection> prerequisites =
          ruleContext.getPrerequisites(ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE, Mode.TARGET);

      for (TransitiveInfoCollection target : prerequisites) {
        Label actualLabel = target.getLabel();
        Label specifiedLabel = AliasProvider.getDependencyLabel(target);
        String specifiedValue =
            maybeCanonicalizeLabel(attributeValue.get(specifiedLabel), target, ruleContext);
        if (specifiedFlagValues.containsKey(actualLabel)) {
          foundDuplicate = true;
        }
        specifiedFlagValues.put(actualLabel, specifiedValue);

        if (target.satisfies(ConfigFeatureFlagProvider.REQUIRE_CONFIG_FEATURE_FLAG_PROVIDER)) {
          // config_feature_flag
          requiredFragmentOptions.add(target.getLabel().toString());
          ConfigFeatureFlagProvider provider = ConfigFeatureFlagProvider.fromTarget(target);
          if (!provider.isValidValue(specifiedValue)) {
            ruleContext.attributeError(
                ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
                String.format(
                    "error while parsing user-defined configuration values: "
                        + "'%s' is not a valid value for '%s'",
                    specifiedValue, specifiedLabel));
            matches = false;
            continue;
          }
          if (!provider.getFlagValue().equals(specifiedValue)) {
            matches = false;
          }
        } else if (target.satisfies(BuildSettingProvider.REQUIRE_BUILD_SETTING_PROVIDER)) {
          // build setting
          requiredFragmentOptions.add(target.getLabel().toString());
          BuildSettingProvider provider = target.getProvider(BuildSettingProvider.class);
          Object configurationValue =
              optionDetails.getOptionValue(specifiedLabel) != null
                  ? optionDetails.getOptionValue(specifiedLabel)
                  : provider.getDefaultValue();
          Object convertedSpecifiedValue;
          try {
            convertedSpecifiedValue =
                BUILD_SETTING_CONVERTERS.get(provider.getType()).convert(specifiedValue);
          } catch (OptionsParsingException e) {
            ruleContext.attributeError(
                ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
                String.format(
                    "error while parsing user-defined configuration values: "
                        + "'%s' cannot be converted to %s type %s",
                    specifiedValue, specifiedLabel, provider.getType()));
            matches = false;
            continue;
          }

          if (configurationValue instanceof List) {
            // If the build_setting is a list, we use the same semantics as for multi-value native
            // flags: if *any* entry in the list matches the config_setting's expected entry, it's
            // a match. In other words, config_setting(flag_values {"//foo": "bar"} matches
            // //foo=["bar", "baz"].

            // If the config_setting expects "foo", convertedSpecifiedValue converts it to the
            // flag's native type, which produces ["foo"]. So unpack that again.
            Object specifiedUnpacked =
                Iterables.getOnlyElement((Iterable<?>) convertedSpecifiedValue);
            if (!((List<?>) configurationValue).contains(specifiedUnpacked)) {
              matches = false;
            }
          } else if (!configurationValue.equals(convertedSpecifiedValue)) {
            matches = false;
          }
        } else {
          ruleContext.attributeError(
              ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
              String.format(
                  "error while parsing user-defined configuration values: "
                      + "%s keys must be build settings or feature flags and %s is not",
                  ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE, specifiedLabel));
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
            ruleContext.attributeError(
                ConfigSettingRule.FLAG_SETTINGS_ATTRIBUTE,
                String.format(
                    "flag '%s' referenced multiple times as ['%s']",
                    actualLabel, QUOTED_COMMA_JOINER.join(aliasList)));
          }
        }
        matches = false;
      }

      return new UserDefinedFlagMatch(matches, ImmutableMap.copyOf(specifiedFlagValues));
    }
  }

  /**
   * Given a 'flag_values = {"//ref:to:flagTarget": "expectedValue"}' pair, if expectedValue is a
   * relative label (e.g. ":sometarget") and flagTarget's value(s) are label-typed, returns an
   * absolute form of the label under the config_setting's package. Else returns the original value
   * unchanged.
   *
   * <p>This lets config_setting use relative labels to match against the actual values, which are
   * already represented in absolute form.
   *
   * <p>The value is returned as a string because it's subsequently fed through the flag's type
   * converter (which maps a string to the final type). Invalid labels are treated no differently
   * (they don't trigger special errors here) because the type converter will also handle that.
   *
   * @param expectedValue the raw value the config_setting expects
   * @param flagTarget the target of the flag whose value is being checked
   * @param @param ruleContext this rule's RuleContext
   */
  private static String maybeCanonicalizeLabel(
      String expectedValue, TransitiveInfoCollection flagTarget, RuleContext ruleContext) {
    if (!flagTarget.satisfies(BuildSettingProvider.REQUIRE_BUILD_SETTING_PROVIDER)) {
      return expectedValue;
    }
    if (!BuildType.isLabelType(flagTarget.getProvider(BuildSettingProvider.class).getType())) {
      return expectedValue;
    }
    if (!expectedValue.startsWith(":")) {
      return expectedValue;
    }
    try {
      return Label.create(
              ruleContext.getRule().getPackage().getPackageIdentifier(), expectedValue.substring(1))
          .getCanonicalForm();
    } catch (LabelSyntaxException e) {
      // Swallow this: the subsequent type conversion already checks for this.
      return expectedValue;
    }
  }
}
