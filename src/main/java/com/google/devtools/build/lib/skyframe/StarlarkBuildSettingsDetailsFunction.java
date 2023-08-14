// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** A builder for {@link StarlarkBuildSettingsDetailsValue} instances. */
final class StarlarkBuildSettingsDetailsFunction implements SkyFunction {

  // Use the plain strings rather than reaching into the Alias class and adding a dependency edge.
  // TODO(blaze-configurability-team): We can probably afford the edge now that this is
  //   inside of skyframe_cluster.
  private static final String ALIAS_RULE_NAME = "alias";
  private static final String ALIAS_ACTUAL_ATTRIBUTE_NAME = "actual";

  StarlarkBuildSettingsDetailsFunction() {}

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, StarlarkBuildSettingsDetailsException {
    StarlarkBuildSettingsDetailsValue.Key key = (StarlarkBuildSettingsDetailsValue.Key) skyKey;

    // Ideally, callers would bypass StarlarkBuildSettingsDetailsFunction entirely when the
    // key is empty but provide a fast escape here just in case.
    if (key.buildSettings().isEmpty()) {
      return StarlarkBuildSettingsDetailsValue.EMPTY;
    }

    try {
      ImmutableMap<PackageIdentifier, PackageValue> buildSettingPackages =
          getBuildSettingPackages(env, key.buildSettings());
      if (buildSettingPackages == null) {
        return null;
      }

      // Each setting is unique so don't need a merge function.
      ImmutableMap<Label, Rule> rawSettingToActualRule =
          key.buildSettings().stream()
              .collect(
                  toImmutableMap(
                      setting -> setting,
                      setting -> getActual(buildSettingPackages, setting).getAssociatedRule()));
      ImmutableSet<Rule> actualRules = ImmutableSet.copyOf(rawSettingToActualRule.values());

      // Calculate info based on the actual rules
      // Different rules have different labels so don't need a merge function
      ImmutableMap<Label, Object> buildSettingToDefault =
          actualRules.stream()
              .collect(
                  toImmutableMap(
                      Rule::getLabel,
                      rule -> {
                        if (rule.getRuleClassObject().getBuildSetting().allowsMultiple()) {
                          return ImmutableList.of(
                              rule.getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME));
                        }
                        return rule.getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME);
                      }));
      ImmutableMap<Label, Type<?>> buildSettingToType =
          actualRules.stream()
              .collect(
                  toImmutableMap(
                      Rule::getLabel,
                      rule -> rule.getRuleClassObject().getBuildSetting().getType()));
      ImmutableSet<Label> buildSettingIsAllowsMultiple =
          actualRules.stream()
              .filter(rule -> rule.getRuleClassObject().getBuildSetting().allowsMultiple())
              .map(Rule::getLabel)
              .collect(toImmutableSet());

      // Calculate the alias table (filtering out non-aliases!)
      ImmutableMap<Label, Label> aliasToActual =
          rawSettingToActualRule.entrySet().stream()
              .filter(entry -> !entry.getKey().equals(entry.getValue().getLabel()))
              .collect(toImmutableMap(Map.Entry::getKey, entry -> entry.getValue().getLabel()));

      return StarlarkBuildSettingsDetailsValue.create(
          buildSettingToDefault, buildSettingToType, buildSettingIsAllowsMultiple, aliasToActual);

    } catch (TransitionException e) {
      throw new StarlarkBuildSettingsDetailsException(e);
    }
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
  private static ImmutableMap<PackageIdentifier, PackageValue> getBuildSettingPackages(
      SkyFunction.Environment env, ImmutableSet<Label> buildSettings)
      throws InterruptedException, TransitionException {
    HashMap<PackageIdentifier, PackageValue> buildSettingPackages = new HashMap<>();
    // This happens before cycle detection so keep track of all seen build settings to ensure
    // we don't get stuck in endless loops (e.g. //alias1->//alias2 && //alias2->alias1)
    Set<Label> allSeenBuildSettings = new HashSet<>();
    ImmutableSet<Label> unverifiedBuildSettings = buildSettings;
    while (!unverifiedBuildSettings.isEmpty()) {
      for (Label buildSetting : unverifiedBuildSettings) {
        if (!allSeenBuildSettings.add(buildSetting)) {
          throw new TransitionException(
              String.format(
                  "Dependency cycle involving '%s' detected in aliased build settings",
                  buildSetting));
        }
      }
      ImmutableSet<PackageIdentifier> packageKeys =
          getPackageKeysFromLabels(unverifiedBuildSettings);
      SkyframeLookupResult newlyLoaded = env.getValuesAndExceptions(packageKeys);
      if (env.valuesMissing()) {
        return null;
      }
      for (PackageIdentifier packageKey : packageKeys) {
        try {
          SkyValue skyValue = newlyLoaded.getOrThrow(packageKey, NoSuchPackageException.class);
          buildSettingPackages.put(packageKey, (PackageValue) skyValue);
        } catch (NoSuchPackageException e) {
          throw new TransitionException(e);
        }
      }
      unverifiedBuildSettings =
          verifyBuildSettingsAndGetAliases(buildSettingPackages, unverifiedBuildSettings);
    }
    return ImmutableMap.copyOf(buildSettingPackages);
  }

  /** Given a set of labels, return a set of their package {@link PackageIdentifier} keys. */
  private static ImmutableSet<PackageIdentifier> getPackageKeysFromLabels(
      Set<Label> buildSettings) {
    ImmutableSet.Builder<PackageIdentifier> keyBuilder = new ImmutableSet.Builder<>();
    for (Label setting : buildSettings) {
      keyBuilder.add(setting.getPackageIdentifier());
    }
    return keyBuilder.build();
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
  private static ImmutableSet<Label> verifyBuildSettingsAndGetAliases(
      Map<PackageIdentifier, PackageValue> buildSettingPackages, Set<Label> buildSettingsToVerify)
      throws TransitionException {
    ImmutableSet.Builder<Label> actualSettingBuilder = new ImmutableSet.Builder<>();
    for (Label allegedBuildSetting : buildSettingsToVerify) {
      Package buildSettingPackage =
          buildSettingPackages.get(allegedBuildSetting.getPackageIdentifier()).getPackage();
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
                "attempting to transition on '%s' which is not a build setting",
                allegedBuildSetting));
      }
      if (buildSettingTarget.getAssociatedRule().getRuleClass().equals(ALIAS_RULE_NAME)) {
        Object actualValue =
            buildSettingTarget.getAssociatedRule().getAttr(ALIAS_ACTUAL_ATTRIBUTE_NAME);
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
                "attempting to transition on '%s' which is not a build setting",
                allegedBuildSetting));
      }
    }
    return actualSettingBuilder.build();
  }

  /**
   * Given a {@link Label} that could be an {@link com.google.devtools.build.lib.rules.Alias} and a
   * set of packages, find the actual target that {@link Label} ultimately points to.
   *
   * <p>This method assumes that the packages of the entire {@link
   * com.google.devtools.build.lib.rules.Alias} chain (if {@code setting} is indeed an alias) are
   * included in {@code buildSettingPackages}
   *
   * <p>This checking is likely done in {@link #verifyBuildSettingsAndGetAliases}.
   */
  private static Target getActual(
      Map<PackageIdentifier, PackageValue> buildSettingPackages, Label setting) {
    Target target = getTarget(buildSettingPackages, setting);
    while (target.getAssociatedRule().getRuleClass().equals(ALIAS_RULE_NAME)) {
      target =
          getTarget(
              buildSettingPackages,
              (Label) target.getAssociatedRule().getAttr(ALIAS_ACTUAL_ATTRIBUTE_NAME));
    }
    return target;
  }

  /**
   * Return a target given its label and a set of package values we know to contain the target.
   *
   * <p>This method is essentially a wrapper around PackageValue.getTarget.
   *
   * @param buildSettingPackages packages that include {@code setting}'s package
   */
  private static Target getTarget(
      Map<PackageIdentifier, PackageValue> buildSettingPackages, Label setting) {
    Package buildSettingPackage =
        buildSettingPackages.get(setting.getPackageIdentifier()).getPackage();
    Preconditions.checkNotNull(
        buildSettingPackage, "Reading build setting for which we don't have a package");
    Target target;
    try {
      target = buildSettingPackage.getTarget(setting.getName());
    } catch (NoSuchTargetException e) {
      // This should never happen, see javadoc.
      throw new IllegalStateException(e);
    }
    return target;
  }

  private static final class StarlarkBuildSettingsDetailsException extends SkyFunctionException {
    StarlarkBuildSettingsDetailsException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
