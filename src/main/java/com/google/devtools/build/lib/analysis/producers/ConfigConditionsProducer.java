// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.InconsistentNullConfigException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.List;
import javax.annotation.Nullable;

/** Computes the targets that key the configurable attributes used by this rule. */
final class ConfigConditionsProducer
    implements StateMachine, ConfiguredTargetAndDataProducer.ResultSink {
  interface ResultSink {
    void acceptConfigConditions(ConfigConditions configConditions);

    void acceptConfigConditionsError(ConfiguredValueCreationException error);
  }

  // -------------------- Input --------------------
  private final TargetAndConfiguration targetAndConfiguration;
  @Nullable private final PlatformInfo targetPlatformInfo;
  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  @Nullable // Null if there are no config labels.
  private final List<Label> configLabels;
  @Nullable // Null if there are no config labels.
  private final ConfiguredTargetAndData[] prerequisites;
  @Nullable // Null if there are no dependency errors.
  private DetailedExitCode mostImportantExitCode;

  ConfigConditionsProducer(
      TargetAndConfiguration targetAndConfiguration,
      @Nullable PlatformInfo targetPlatformInfo,
      TransitiveDependencyState transitiveState,
      ResultSink sink,
      StateMachine runAfter) {
    this.targetAndConfiguration = targetAndConfiguration;
    this.targetPlatformInfo = targetPlatformInfo;
    this.transitiveState = transitiveState;
    this.sink = sink;
    this.runAfter = runAfter;

    this.configLabels = computeConfigLabels(targetAndConfiguration.getTarget());
    this.prerequisites =
        configLabels == null ? null : new ConfiguredTargetAndData[configLabels.size()];
  }

  @Override
  public StateMachine step(Tasks tasks) {
    if (configLabels == null) {
      sink.acceptConfigConditions(ConfigConditions.EMPTY);
      return runAfter;
    }

    // Collect the actual deps without a configuration transition (since by definition config
    // conditions evaluate over the current target's configuration). If the dependency is
    // (erroneously) something that needs the null configuration, its analysis will be
    // short-circuited. That error will be reported later.
    for (int i = 0; i < configLabels.size(); ++i) {
      tasks.enqueue(
          new ConfiguredTargetAndDataProducer(
              ConfiguredTargetKey.builder()
                  .setLabel(configLabels.get(i))
                  .setConfiguration(targetAndConfiguration.getConfiguration())
                  .build(),
              /* transitionKeys= */ ImmutableList.of(),
              transitiveState,
              (ConfiguredTargetAndDataProducer.ResultSink) this,
              i));
    }
    return this::constructConfigConditions;
  }

  @Override
  public void acceptConfiguredTargetAndData(ConfiguredTargetAndData value, int index) {
    prerequisites[index] = value;
  }

  @Override
  public void acceptConfiguredTargetAndDataError(ConfiguredValueCreationException error) {
    emitErrorIfMostImportant(error.getDetailedExitCode());
  }

  @Override
  public void acceptConfiguredTargetAndDataError(NoSuchThingException error) {
    emitErrorIfMostImportant(error.getDetailedExitCode());
  }

  @Override
  public void acceptConfiguredTargetAndDataError(InconsistentNullConfigException error) {
    // A config label was evaluated with a null configuration. This should never happen as
    // ConfigConditions are only present if the parent is a Rule, then always evaluated with the
    // parent configuration.
    throw new IllegalArgumentException(
        "ConfigCondition dependency should never be evaluated with a null configuration.", error);
  }

  private StateMachine constructConfigConditions(Tasks tasks) {
    if (mostImportantExitCode != null) {
      return runAfter; // There was a previous error.
    }

    var asConfiguredTargets = new ImmutableMap.Builder<Label, ConfiguredTargetAndData>();
    var asConfigConditions = new ImmutableMap.Builder<Label, ConfigMatchingProvider>();
    for (int i = 0; i < configLabels.size(); ++i) {
      var label = configLabels.get(i);
      var prerequisite = prerequisites[i];
      asConfiguredTargets.put(label, prerequisite);
      try {
        asConfigConditions.put(
            label, ConfigConditions.fromConfiguredTarget(prerequisite, targetPlatformInfo));
      } catch (ConfigConditions.InvalidConditionException e) {
        var targetLabel = targetAndConfiguration.getLabel();
        String message =
            String.format(
                    "%s is not a valid select() condition for %s.\n",
                    prerequisite.getTargetLabel(), targetLabel)
                + String.format(
                    "To inspect the select(), run: bazel query --output=build %s.\n", targetLabel)
                + "For more help, see https://bazel.build/reference/be/functions#select.\n\n";
        sink.acceptConfigConditionsError(
            new ConfiguredValueCreationException(targetAndConfiguration, message));
        return runAfter;
      }
    }
    sink.acceptConfigConditions(
        ConfigConditions.create(
            asConfiguredTargets.buildOrThrow(), asConfigConditions.buildOrThrow()));
    return runAfter;
  }

  /**
   * Computes the config labels belonging to the given target.
   *
   * @return null if there were no config labels, implying a {@link ConfigConditions#EMPTY} result.
   */
  @Nullable
  private static List<Label> computeConfigLabels(Target target) {
    if (!(target instanceof Rule)) {
      return null;
    }

    var attrs = RawAttributeMapper.of(((Rule) target));
    if (!attrs.has(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE)) {
      return null;
    }

    // Collects the labels of the configured targets we need to resolve.
    List<Label> configLabels =
        attrs.get(RuleClass.CONFIG_SETTING_DEPS_ATTRIBUTE, BuildType.LABEL_LIST);
    if (configLabels.isEmpty()) {
      return null;
    }
    return configLabels;
  }

  private void emitErrorIfMostImportant(@Nullable DetailedExitCode newExitCode) {
    mostImportantExitCode =
        DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
            newExitCode, mostImportantExitCode);
    if (newExitCode.equals(mostImportantExitCode)) {
      sink.acceptConfigConditionsError(
          // The precise error is reported by the dependency that failed to load.
          // TODO(gregce): beautify this error: https://github.com/bazelbuild/bazel/issues/11984.
          new ConfiguredValueCreationException(
              targetAndConfiguration,
              "errors encountered resolving select() keys for "
                  + targetAndConfiguration.getLabel()));
    }
  }
}
