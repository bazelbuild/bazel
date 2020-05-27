// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.skylark;

import static com.google.devtools.build.lib.analysis.skylark.FunctionTransitionUtil.COMMAND_LINE_OPTION_PREFIX;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A marker class for configuration transitions that are defined in Starlark. */
public abstract class StarlarkTransition implements ConfigurationTransition {

  // Use the plain strings rather than reaching into the Alias class and adding a dependency edge.
  public static final String ALIAS_RULE_NAME = "alias";
  public static final String ALIAS_ACTUAL_ATTRIBUTE_NAME = "actual";

  /** The two groups of build settings that are relevant for a {@link StarlarkTransition} */
  public enum Settings {
    /** Build settings that are read by a {@link StarlarkTransition} */
    INPUTS,
    /** Build settings that are written by a {@link StarlarkTransition} */
    OUTPUTS,
    /** Build settings that are read and/or written by a {@link StarlarkTransition } */
    INPUTS_AND_OUTPUTS
  }

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  public StarlarkTransition(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  private List<String> getInputs() {
    return starlarkDefinedConfigTransition.getInputs();
  }

  private List<String> getOutputs() {
    return starlarkDefinedConfigTransition.getOutputs();
  }

  /** Exception class for exceptions thrown during application of a starlark-defined transition */
  // TODO(juliexxia): add more information to this exception e.g. originating target of transition
  public static class TransitionException extends Exception {
    private final String message;

    public TransitionException(String message) {
      this.message = message;
    }

    public TransitionException(Throwable cause) {
      this.message = cause.getMessage();
    }

    /** Returns the error message. */
    @Override
    public String getMessage() {
      return message;
    }
  }

  /**
   * Given a {@link ConfigurationTransition}, decompose (if possible) and find all referenced build
   * settings.
   *
   * <p>If a transition references a build setting via an alias, this set includes the alias' label
   * and *does not* include the actual label i.e. this method returns all referenced labels exactly
   * as they are.
   */
  public static ImmutableSet<Label> getAllBuildSettings(ConfigurationTransition root) {
    ImmutableSet.Builder<Label> keyBuilder = new ImmutableSet.Builder<>();
    try {
      root.visit(
          (StarlarkTransitionVisitor)
              transition -> {
                keyBuilder.addAll(
                    getRelevantStarlarkSettingsFromTransition(
                        transition, Settings.INPUTS_AND_OUTPUTS));
              });
    } catch (TransitionException e) {
      // Not actually thrown in the visitor, but declared.
    }
    return keyBuilder.build();
  }

  /** Given a set of labels, return a set of their package {@link PackageValue.Key}s. */
  public static ImmutableSet<PackageValue.Key> getPackageKeysFromLabels(
      ImmutableSet<Label> buildSettings) {
    ImmutableSet.Builder<PackageValue.Key> keyBuilder = new ImmutableSet.Builder<>();
    for (Label setting : buildSettings) {
      keyBuilder.add(PackageValue.key(setting.getPackageIdentifier()));
    }
    return keyBuilder.build();
  }

  /**
   * Given a {@link Label} that could be an {@link com.google.devtools.build.lib.rules.Alias} and a
   * set of packages, find the actual target that {@link Label} ultimately points to.
   *
   * <ul>
   *   This method assumes that
   *   <li>the packages of the entire {@link com.google.devtools.build.lib.rules.Alias} chain (if
   *       {@code setting} is indeed an alias) are included in {@code buildSettingPackages}
   *   <li>the alias chain terminates in a build setting
   * </ul>
   *
   * <p>This checking is likely done in {@link #verifyBuildSettingsAndGetAliases}.
   */
  private static Target getActual(
      Map<PackageValue.Key, PackageValue> buildSettingPackages, Label setting) {
    Target buildSettingTarget = getBuildSettingTarget(buildSettingPackages, setting);
    while (buildSettingTarget.getAssociatedRule().getRuleClass().equals(ALIAS_RULE_NAME)) {
      buildSettingTarget =
          getBuildSettingTarget(
              buildSettingPackages,
              (Label)
                  buildSettingTarget
                      .getAssociatedRule()
                      .getAttributeContainer()
                      .getAttr(ALIAS_ACTUAL_ATTRIBUTE_NAME));
    }
    return buildSettingTarget;
  }

  /**
   * Given a {@link ConfigurationTransition} find all build settings read or set by the transition
   * and load their packages.
   *
   * <p>In the case that build settings are referred to by aliases, we do a couple loops of package
   * loading. We generally don't expect build settings to be aliased multiple times so we don't
   * expect this while loop (and relevant null return) to happen more than two or three times (and
   * usually only once).
   *
   * @return the package keys and values of build settings or null if not all packages are
   *     available. if not null, and some build settings are referenced by alias, the returned map
   *     will include both alias and actual packages to allow for alias chain following at a later
   *     state.
   */
  @Nullable
  public static HashMap<PackageValue.Key, PackageValue> getBuildSettingPackages(
      SkyFunction.Environment env, ConfigurationTransition root)
      throws InterruptedException, TransitionException {
    HashMap<PackageValue.Key, PackageValue> buildSettingPackages = new HashMap<>();
    // This happens before cycle detection so keep track of all seen build settings to ensure
    // we don't get stuck in endless loops (e.g. //alias1->//alias2 && //alias2->alias1)
    Set<Label> allSeenBuildSettings = new HashSet<>();
    Set<Label> unverifiedBuildSettings = getAllBuildSettings(root);
    while (!unverifiedBuildSettings.isEmpty()) {
      for (Label buildSetting : unverifiedBuildSettings) {
        if (!allSeenBuildSettings.add(buildSetting)) {
          throw new TransitionException(
              String.format(
                  "Dependency cycle involving '%s' detected in aliased build settings",
                  buildSetting));
        }
      }
      ImmutableSet<PackageValue.Key> packageKeys =
          getPackageKeysFromLabels(ImmutableSet.copyOf(unverifiedBuildSettings));
      Map<SkyKey, SkyValue> newlyLoaded = env.getValues(packageKeys);
      if (env.valuesMissing()) {
        return null;
      }
      for (Map.Entry<SkyKey, SkyValue> buildSettingOrAliasPackage : newlyLoaded.entrySet()) {
        buildSettingPackages.put(
            (PackageValue.Key) buildSettingOrAliasPackage.getKey(),
            (PackageValue) buildSettingOrAliasPackage.getValue());
      }
      unverifiedBuildSettings =
          verifyBuildSettingsAndGetAliases(buildSettingPackages, unverifiedBuildSettings);
    }
    return buildSettingPackages;
  }

  /**
   * Method to be called after Starlark-transitions are applied. Checks outputs.
   *
   * <p>We only do validation on Starlark-defined build settings. Native options (designated with
   * {@code COMMAND_LINE_OPTION_PREFIX}) already have their output values checked in {@link
   * FunctionTransitionUtil#applyTransition}.
   *
   * <p>Remove build settings in {@code toOptions} that have been set to their default value. This
   * is how we ensure that an unset build setting and a set-to-default build settings represent the
   * same configuration.
   *
   * <p>Deduplicate redundant build settings from the result of split transitions. The first
   * encountered split key is used to represent the deduped build setting.
   *
   * @param root transition that was applied. Likely a {@link
   *     com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition} so we
   *     decompose and post-process all StarlarkTransitions out of whatever transition is passed
   *     here.
   * @param buildSettingPackages PackageValue.Key/Values of packages that contain all
   *     Starlark-defined build settings that were set by {@code root}. If any build settings are
   *     referenced by {@link com.google.devtools.build.lib.rules.Alias}, this contains all packages
   *     in the alias chain.
   * @param toOptions result of applying {@code root}
   * @return validated toOptions with default values filtered out
   * @throws TransitionException if an error occurred during Starlark transition application.
   */
  // TODO(juliexxia): the current implementation masks certain bad transitions and only checks the
  // final result. I.e. if a transition that writes a non int --//int-build-setting is composed
  // with another transition that writes --//int-build-setting (without reading it first), then
  // the bad output of transition 1 is masked.
  public static Map<String, BuildOptions> validate(
      ConfigurationTransition root,
      Map<PackageValue.Key, PackageValue> buildSettingPackages,
      Map<String, BuildOptions> toOptions)
      throws TransitionException {
    // collect settings changed during this transition and their types
    Map<Label, Rule> changedSettingToRule = Maps.newHashMap();
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              ImmutableSet<Label> changedSettings =
                  getRelevantStarlarkSettingsFromTransition(transition, Settings.OUTPUTS);
              for (Label setting : changedSettings) {
                changedSettingToRule.put(
                    setting, getActual(buildSettingPackages, setting).getAssociatedRule());
              }
            });

    // Return early if no starlark settings were affected.
    if (changedSettingToRule.isEmpty()) {
      return toOptions;
    }

    ImmutableMap.Builder<Label, Label> aliasToActualBuilder = new ImmutableMap.Builder<>();
    for (Map.Entry<Label, Rule> changedSettingWithRule : changedSettingToRule.entrySet()) {
      Label setting = changedSettingWithRule.getKey();
      Rule rule = changedSettingWithRule.getValue();
      if (!rule.getLabel().equals(setting)) {
        aliasToActualBuilder.put(setting, rule.getLabel());
      }
    }
    ImmutableMap<Label, Label> aliasToActual = aliasToActualBuilder.build();

    // Verify changed settings were changed to something reasonable for their type and filter out
    // default values.
    ImmutableMap.Builder<String, BuildOptions> cleanedOptionMap = ImmutableMap.builder();
    Set<BuildOptions> cleanedOptionSet = Sets.newLinkedHashSetWithExpectedSize(toOptions.size());
    for (String splitKey : toOptions.keySet()) {
      // Lazily initialized to optimize for the common case where we don't modify anything.
      BuildOptions.Builder cleanedOptions = null;
      // Clean up aliased values.
      BuildOptions options = unalias(toOptions.get(splitKey), aliasToActual);
      for (Map.Entry<Label, Rule> changedSettingWithRule : changedSettingToRule.entrySet()) {
        // If the build setting was referenced in the transition via an alias, this is that alias
        Label maybeAliasSetting = changedSettingWithRule.getKey();
        Rule rule = changedSettingWithRule.getValue();
        // If the build setting was *not* referenced in the transition by an alias, this is the same
        // value as {@code maybeAliasSetting} above.
        Label actualSetting = rule.getLabel();
        Object newValue = options.getStarlarkOptions().get(actualSetting);
        // TODO(b/154132845): fix NPE occasionally observed here.
        Preconditions.checkState(
            newValue != null,
            "Error while attempting to validate new values from starlark"
                + " transition(s) with the outputs %s. Post-transition configuration should include"
                + " '%s' but only includes starlark options: %s. If you run into this error"
                + " please ping b/154132845 or email blaze-configurability@google.com.",
            changedSettingToRule.keySet(),
            actualSetting,
            options.getStarlarkOptions().keySet());
        Object convertedValue;
        try {
          convertedValue =
              rule.getRuleClassObject()
                  .getBuildSetting()
                  .getType()
                  .convert(newValue, maybeAliasSetting);
        } catch (ConversionException e) {
          throw new TransitionException(e);
        }
        if (convertedValue.equals(
            rule.getAttributeContainer().getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME))) {
          if (cleanedOptions == null) {
            cleanedOptions = options.toBuilder();
          }
          cleanedOptions.removeStarlarkOption(rule.getLabel());
        }
      }
      // Keep the same instance if we didn't do anything to maintain reference equality later on.
      options = cleanedOptions != null ? cleanedOptions.build() : options;
      if (cleanedOptionSet.add(options)) {
        cleanedOptionMap.put(splitKey, options);
      }
    }
    return cleanedOptionMap.build();
  }

  /*
   * Resolve aliased build setting issues
   *
   * <p>If a build setting is transitioned upon via an alias, the resulting {@link
   * BuildOptions#getStarlarkOptions()} map will look like this:
   *
   * <entry1>alias-label -> new-value
   * <entry2>actual-label -> old-value
   *
   * <p>we need to collapse this to the correct single entry: actual-label -> new-value.
   * By the end of this method, the starlark options map in the returned {@link BuildOptions}
   * contains only keys that are actual build settings, no aliases.
   */
  private static BuildOptions unalias(
      BuildOptions options, ImmutableMap<Label, Label> aliasToActual) {
    if (aliasToActual.isEmpty()) {
      return options;
    }
    Collection<Label> aliases = aliasToActual.keySet();
    Collection<Label> actuals = aliasToActual.values();
    BuildOptions.Builder toReturn = options.toBuilder();
    for (Map.Entry<Label, Object> entry : options.getStarlarkOptions().entrySet()) {
      Label setting = entry.getKey();
      if (actuals.contains(setting)) {
        // if entry is keyed by an actual (e.g. <entry2> in javadoc), don't care about its value
        // it's stale
        continue;
      } else if (aliases.contains(setting)) {
        // if an entry is keyed by an alias (e.g. <entry1> in javadoc), newly key (overwrite) its
        // actual to its alias' value and remove the alias-keyed entry
        toReturn.addStarlarkOption(
            aliasToActual.get(setting), options.getStarlarkOptions().get(setting));
        toReturn.removeStarlarkOption(setting);
      } else {
        // else - just copy over
        toReturn.addStarlarkOption(entry.getKey(), entry.getValue());
      }
    }
    return toReturn.build();
  }

  /**
   * For a given transition, find all Starlark build settings that are read while applying it, then
   * return a map of their label to their default values.
   *
   * <p>If the build setting is referenced by an {@link com.google.devtools.build.lib.rules.Alias},
   * the returned map entry is still keyed by the alias.
   *
   * @param buildSettingPackages contains packages of all build settings read by Starlark
   *     transitions composing {@code root}. It may also contain other packages (e.g. packages of
   *     build settings *written* by relevant transitions) so do not iterate over for input
   *     packages.
   */
  public static ImmutableMap<Label, Object> getDefaultInputValues(
      Map<PackageValue.Key, PackageValue> buildSettingPackages, ConfigurationTransition root)
      throws TransitionException {
    ImmutableMap.Builder<Label, Object> defaultValues = new ImmutableMap.Builder<>();
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              ImmutableSet<Label> settings =
                  getRelevantStarlarkSettingsFromTransition(transition, Settings.INPUTS);
              for (Label setting : settings) {
                defaultValues.put(
                    setting,
                    getActual(buildSettingPackages, setting)
                        .getAssociatedRule()
                        .getAttributeContainer()
                        .getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME));
              }
            });
    return defaultValues.build();
  }

  /**
   * Return a target given its label and a set of package values we know to contain the target.
   *
   * <p>This method assumes {@code setting} is a valid build setting target or an alias of a build
   * setting target based on checking done in {@code #verifyBuildSettingsAndGetAliases}
   *
   * @param buildSettingPackages packages that include {@code setting}'s package
   */
  private static Target getBuildSettingTarget(
      Map<PackageValue.Key, PackageValue> buildSettingPackages, Label setting) {
    Package buildSettingPackage =
        buildSettingPackages.get(PackageValue.key(setting.getPackageIdentifier())).getPackage();
    Preconditions.checkNotNull(
        buildSettingPackage, "Reading build setting for which we don't have a package");
    Target buildSettingTarget;
    try {
      buildSettingTarget = buildSettingPackage.getTarget(setting.getName());
    } catch (NoSuchTargetException e) {
      // This should never happen, see javadoc.
      throw new IllegalStateException(e);
    }
    return buildSettingTarget;
  }

  /**
   * Given a preliminary set of alleged build setting labels and relevant packages, verify that the
   * given {@link Label}s actually correspond to build setting targets.
   *
   * <p>This method is meant to be run in a loop to handle aliased build settings. It also
   * explicitly bans configured 'actual' values for aliased build settings. Since build settings are
   * used to define configuration, there should be better ways to accomplish disparate
   * configurations than configured aliases. Also from a technical standpoint, it's unclear what
   * configuration is correct to use to resolve configured attributes.
   *
   * @param buildSettingPackages packages that include {@code buildSettingsToVerify}'s packages
   * @param buildSettingsToVerify alleged build setting labels
   * @return a set of "actual" labels of any build settings that are referenced by aliases (note -
   *     if the "actual" value of aliasA is aliasB, this method returns aliasB AKA we only follow
   *     one link in the alias chain per call of this method)
   */
  public static ImmutableSet<Label> verifyBuildSettingsAndGetAliases(
      Map<PackageValue.Key, PackageValue> buildSettingPackages, Set<Label> buildSettingsToVerify)
      throws TransitionException {
    ImmutableSet.Builder<Label> actualSettingBuilder = new ImmutableSet.Builder<>();
    for (Label allegedBuildSetting : buildSettingsToVerify) {
      Package buildSettingPackage =
          buildSettingPackages
              .get(PackageValue.key(allegedBuildSetting.getPackageIdentifier()))
              .getPackage();
      Preconditions.checkNotNull(
          buildSettingPackage, "Reading build setting for which we don't have a package");
      Target buildSettingTarget;
      try {
        buildSettingTarget = buildSettingPackage.getTarget(allegedBuildSetting.getName());
      } catch (NoSuchTargetException e) {
        throw new TransitionException(e);
      }
      if (buildSettingTarget.getAssociatedRule() == null) {
        throw new TransitionException(
            String.format(
                "attempting to transition on '%s' which is not a" + " build setting",
                allegedBuildSetting));
      }
      if (buildSettingTarget.getAssociatedRule().getRuleClass().equals(ALIAS_RULE_NAME)) {
        Object actualValue =
            buildSettingTarget
                .getAssociatedRule()
                .getAttributeContainer()
                .getAttr(ALIAS_ACTUAL_ATTRIBUTE_NAME);
        if (actualValue instanceof Label) {
          actualSettingBuilder.add((Label) actualValue);
          continue;
        } else if (actualValue instanceof SelectorList) {
          // configured "actual" value
          throw new TransitionException(
              String.format(
                  "attempting to transition on aliased build setting '%s', the actual value of"
                      + " which uses select(). Aliased build settings with configured actual values"
                      + " is not supported.",
                  allegedBuildSetting));
        } else {
          throw new IllegalStateException(
              String.format(
                  "Alias target '%s' with 'actual' attr value not equals to "
                      + "a label or a selectorlist",
                  allegedBuildSetting));
        }
      }
      if (!buildSettingTarget.getAssociatedRule().isBuildSetting()) {
        throw new TransitionException(
            String.format(
                "attempting to transition on '%s' which is not a" + " build setting",
                allegedBuildSetting));
      }
    }
    return actualSettingBuilder.build();
  }

  private static ImmutableSet<Label> getRelevantStarlarkSettingsFromTransition(
      StarlarkTransition transition, Settings settings) {
    Set<String> toGet = new HashSet<>();
    switch (settings) {
      case INPUTS:
        toGet.addAll(transition.getInputs());
        break;
      case OUTPUTS:
        toGet.addAll(transition.getOutputs());
        break;
      case INPUTS_AND_OUTPUTS:
        toGet.addAll(transition.getInputs());
        toGet.addAll(transition.getOutputs());
        break;
    }
    return ImmutableSet.copyOf(
        toGet.stream()
            .filter(setting -> !setting.startsWith(COMMAND_LINE_OPTION_PREFIX))
            .map(Label::parseAbsoluteUnchecked)
            .collect(Collectors.toSet()));
  }

  @Override
  public boolean equals(Object object) {
    if (object == this) {
      return true;
    }
    if (object instanceof StarlarkTransition) {
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition =
          ((StarlarkTransition) object).starlarkDefinedConfigTransition;
      return Objects.equals(starlarkDefinedConfigTransition, this.starlarkDefinedConfigTransition);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(starlarkDefinedConfigTransition);
  }

  /** Given a transition, figures out if it composes any Starlark transitions. */
  public static boolean doesStarlarkTransition(ConfigurationTransition root)
      throws TransitionException {
    AtomicBoolean doesStarlarkTransition = new AtomicBoolean(false);
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              doesStarlarkTransition.set(true);
            });
    return doesStarlarkTransition.get();
  }

  @FunctionalInterface
  // This is only used in this class to handle the cast and the exception
  @SuppressWarnings("FunctionalInterfaceMethodChanged")
  private interface StarlarkTransitionVisitor
      extends ConfigurationTransition.Visitor<TransitionException> {
    @Override
    default void accept(ConfigurationTransition transition) throws TransitionException {
      if (transition instanceof StarlarkTransition) {
        this.accept((StarlarkTransition) transition);
      }
    }

    void accept(StarlarkTransition transition) throws TransitionException;
  }
}
