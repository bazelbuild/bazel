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
import static com.google.devtools.build.lib.packages.RuleClass.Builder.SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/** A marker class for configuration transitions that are defined in Starlark. */
public abstract class StarlarkTransition implements ConfigurationTransition {

  /** The two groups of build settings that are relevant for a {@link StarlarkTransition} */
  public enum Settings {
    /** Build settings that are read by a {@link StarlarkTransition} */
    INPUTS,
    /** Build settings that are written by a {@link StarlarkTransition} */
    OUTPUTS,
  }

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  public StarlarkTransition(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  public void replayOn(ExtendedEventHandler eventHandler) {
    starlarkDefinedConfigTransition.getEventHandler().replayOn(eventHandler);
    starlarkDefinedConfigTransition.getEventHandler().clear();
  }

  public boolean hasErrors() {
    return starlarkDefinedConfigTransition.getEventHandler().hasErrors();
  }

  private List<String> getInputs() {
    return starlarkDefinedConfigTransition.getInputs();
  }

  private List<String> getOutputs() {
    return starlarkDefinedConfigTransition.getOutputs();
  }

  /** Exception class for exceptions thrown during application of a starlark-defined transition */
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
   * For a given transition, find all relevant Starlark-defined build settings. Then return all
   * package keys for those flags.
   *
   * <p>Currently this method does not handle the possibility of aliased build settings. We may not
   * actually load the package that actually contains the build setting but we won't know until we
   * fetch the actual target.
   *
   * @param root transition to inspect
   * @param inputsOrOutputs whether to return inputs or outputs
   */
  // TODO(juliexxia): handle the possibility of aliased build settings.
  public static ImmutableSet<SkyKey> getBuildSettingPackageKeys(
      ConfigurationTransition root, Settings inputsOrOutputs) {
    ImmutableSet.Builder<SkyKey> keyBuilder = new ImmutableSet.Builder<>();
    try {
      root.visit(
          (StarlarkTransitionVisitor)
              transition -> {
                keyBuilder.addAll(
                    getBuildSettingPackageKeys(
                        getRelevantStarlarkSettingsFromTransition(transition, inputsOrOutputs)));
              });
    } catch (TransitionException e) {
      // Not actually thrown in the visitor, but declared.
    }
    return keyBuilder.build();
  }

  private static ImmutableSet<SkyKey> getBuildSettingPackageKeys(
      ImmutableSet<Label> buildSettings) {
    ImmutableSet.Builder<SkyKey> keyBuilder = new ImmutableSet.Builder<>();
    for (Label setting : buildSettings) {
      keyBuilder.add(PackageValue.key(setting.getPackageIdentifier()));
    }
    return keyBuilder.build();
  }

  /**
   * Method to be called after Starlark-transitions are applied. Handles events and checks outputs.
   *
   * <p>Logs any events (e.g. {@code print()}s, errors} to output and throws an error if we had any
   * errors. Right now, Starlark transitions are the only kind that knows how to throw errors so we
   * know this will only report and throw if a Starlark transition caused a problem.
   *
   * <p>We only do validation on Starlark-defined build settings. Native options (designated with
   * {@code COMMAND_LINE_OPTION_PREFIX}) already have their output values checked in {@link
   * FunctionTransitionUtil#applyTransition}.
   *
   * <p>Remove build settings in {@code toOptions} that have been set to their default value. This
   * is how we ensure that an unset build setting and a set-to-default build settings represent the
   * same configuration.
   *
   * @param root transition that was applied. Likely a {@link ComposingTransition} so we decompose
   *     and post-process all StarlarkTransitions out of whatever transition is passed here.
   * @param buildSettingPackages SkyKeys/Values of packages that contain all Starlark-defined build
   *     settings that were set by {@code root}
   * @param toOptions result of applying {@code root}
   * @return validated toOptions with default values filtered out
   * @throws TransitionException if an error occurred during Starlark transition application.
   */
  // TODO(juliexxia): the current implementation masks certain bad transitions and only checks the
  // final result. I.e. if a transition that writes a non int --//int-build-setting is composed
  // with another transition that writes --//int-build-setting (without reading it first), then
  // the bad output of transition 1 is masked.
  public static List<BuildOptions> validate(
      ConfigurationTransition root,
      Map<SkyKey, SkyValue> buildSettingPackages,
      List<BuildOptions> toOptions)
      throws TransitionException {
    // collect settings changed during this transition and their types
    Map<Label, Rule> changedSettingToRule = Maps.newHashMap();
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              ImmutableSet<Label> changedSettings =
                  getRelevantStarlarkSettingsFromTransition(transition, Settings.OUTPUTS);
              for (Label setting : changedSettings) {
                Target buildSettingTarget =
                    getAndCheckBuildSettingTarget(buildSettingPackages, setting);
                changedSettingToRule.put(setting, buildSettingTarget.getAssociatedRule());
              }
            });

    // Verify changed settings were changed to something reasonable for their type and filter out
    // default values.
    Set<BuildOptions> cleanedOptionList = new LinkedHashSet<>(toOptions.size());
    for (BuildOptions options : toOptions) {
      // Lazily initialized to optimize for the common case where we don't modify anything.
      BuildOptions.Builder cleanedOptions = null;
      for (Map.Entry<Label, Rule> changedSettingWithRule : changedSettingToRule.entrySet()) {
        Label setting = changedSettingWithRule.getKey();
        Rule rule = changedSettingWithRule.getValue();
        Object newValue = options.getStarlarkOptions().get(setting);
        Object convertedValue;
        try {
          convertedValue =
              rule.getRuleClassObject().getBuildSetting().getType().convert(newValue, setting);
        } catch (ConversionException e) {
          throw new TransitionException(e);
        }
        if (convertedValue.equals(
            rule.getAttributeContainer().getAttr(SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME))) {
          if (cleanedOptions == null) {
            cleanedOptions = options.toBuilder();
          }
          cleanedOptions.removeStarlarkOption(setting);
        }
      }
      // Keep the same instance if we didn't do anything to maintain reference equality later on.
      cleanedOptionList.add(cleanedOptions != null ? cleanedOptions.build() : options);
    }
    return ImmutableList.copyOf(cleanedOptionList);
  }

  /**
   * For a given transition, find all Starlark build settings that are read while applying it, then
   * return a map of their label to their default values.
   */
  public static ImmutableMap<Label, Object> getDefaultInputValues(
      Environment env, ConfigurationTransition root)
      throws TransitionException, InterruptedException {
    ImmutableSet<SkyKey> buildSettingInputPackageKeys =
        getBuildSettingPackageKeys(root, Settings.INPUTS);
    Map<SkyKey, SkyValue> buildSettingPackages = env.getValues(buildSettingInputPackageKeys);
    if (env.valuesMissing()) {
      return null;
    }
    return getDefaultInputValues(buildSettingPackages, root);
  }

  /**
   * For a given transition, find all Starlark build settings that are read while applying it, then
   * return a map of their label to their default values.
   */
  public static ImmutableMap<Label, Object> getDefaultInputValues(
      Map<SkyKey, SkyValue> buildSettingPackages, ConfigurationTransition root)
      throws TransitionException {
    ImmutableMap.Builder<Label, Object> defaultValues = new ImmutableMap.Builder<>();
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              ImmutableSet<Label> settings =
                  getRelevantStarlarkSettingsFromTransition(transition, Settings.INPUTS);
              for (Label setting : settings) {
                Target buildSettingTarget =
                    getAndCheckBuildSettingTarget(buildSettingPackages, setting);
                defaultValues.put(
                    setting,
                    buildSettingTarget
                        .getAssociatedRule()
                        .getAttributeContainer()
                        .getAttr(SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME));
              }
            });
    return defaultValues.build();
  }

  private static Target getAndCheckBuildSettingTarget(
      Map<SkyKey, SkyValue> buildSettingPackages, Label setting) throws TransitionException {
    Package buildSettingPackage =
        ((PackageValue) buildSettingPackages.get(PackageValue.key(setting.getPackageIdentifier())))
            .getPackage();
    Preconditions.checkNotNull(
        buildSettingPackage, "Reading build setting for which we don't have a package");
    Target buildSettingTarget;
    try {
      buildSettingTarget = buildSettingPackage.getTarget(setting.getName());
    } catch (NoSuchTargetException e) {
      throw new TransitionException(e);
    }
    if (buildSettingTarget.getAssociatedRule() == null
        || buildSettingTarget.getAssociatedRule().getRuleClassObject().getBuildSetting() == null) {
      throw new TransitionException(
          String.format("attempting to transition on '%s' which is not a build setting", setting));
    }
    return buildSettingTarget;
  }

  // TODO(juliexxia): use an enum for "inputs"/"outputs" here and elsewhere in starlark transitions.
  private static ImmutableSet<Label> getRelevantStarlarkSettingsFromTransition(
      StarlarkTransition transition, Settings inputOrOutput) {
    List<String> toGet =
        inputOrOutput.equals(Settings.INPUTS) ? transition.getInputs() : transition.getOutputs();
    return ImmutableSet.copyOf(
        toGet.stream()
            .filter(setting -> !setting.startsWith(COMMAND_LINE_OPTION_PREFIX))
            .map(Label::parseAbsoluteUnchecked)
            .collect(Collectors.toSet()));
  }

  /**
   * For a given transition, for any Starlark-defined transitions that compose it, replay events. If
   * any events were errors, throw an error.
   */
  public static void replayEvents(ExtendedEventHandler eventHandler, ConfigurationTransition root)
      throws TransitionException {
    root.visit(
        (StarlarkTransitionVisitor)
            transition -> {
              // Replay events and errors and throw if there were errors
              boolean hasErrors = transition.hasErrors();
              transition.replayOn(eventHandler);
              if (hasErrors) {
                throw new TransitionException(
                    "Errors encountered while applying Starlark transition");
              }
            });
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
