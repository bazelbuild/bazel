// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.NOT_TEST;
import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.PARALLEL;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ConflictException;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeIterableResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Drives the analysis & execution of an ActionLookupKey, which is wrapped inside a BuildDriverKey.
 */
public class BuildDriverFunction implements SkyFunction {
  private final TransitiveActionLookupValuesCollector transitiveActionLookupValuesCollector;
  private final Supplier<IncrementalArtifactConflictFinder> incrementalArtifactConflictFinder;
  private final Supplier<EventBus> eventBus;

  BuildDriverFunction(
      TransitiveActionLookupValuesCollector transitiveActionLookupValuesCollector,
      Supplier<IncrementalArtifactConflictFinder> incrementalArtifactConflictFinder,
      Supplier<EventBus> eventBus) {
    this.transitiveActionLookupValuesCollector = transitiveActionLookupValuesCollector;
    this.incrementalArtifactConflictFinder = incrementalArtifactConflictFinder;
    this.eventBus = eventBus;
  }

  private static class State implements SkyKeyComputeState {
    private ImmutableMap<ActionAnalysisMetadata, ConflictException> actionConflicts;
    private boolean sentTestAnalysisCompleteEvent = false;
  }
  /**
   * From the ConfiguredTarget/Aspect keys, get the top-level artifacts. Then evaluate them together
   * with the appropriate CompletionFunctions. This is the bridge between the conceptual analysis &
   * execution phases.
   *
   * <p>TODO(b/199053098): implement build-info, build-changelist, coverage & exception handling.
   */
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    BuildDriverKey buildDriverKey = (BuildDriverKey) skyKey;
    ActionLookupKey actionLookupKey = buildDriverKey.getActionLookupKey();
    TopLevelArtifactContext topLevelArtifactContext = buildDriverKey.getTopLevelArtifactContext();
    State state = env.getState(State::new);

    // Register a dependency on the BUILD_ID. We do this to make sure BuildDriverFunction is
    // reevaluated every build.
    PrecomputedValue.BUILD_ID.get(env);

    // Why SkyValue and not ActionLookupValue? The evaluation of some ActionLookupKey can result in
    // classes that don't implement ActionLookupValue
    // (e.g. ConfiguredTargetKey -> NonRuleConfiguredTargetValue).
    SkyValue topLevelSkyValue = env.getValue(actionLookupKey);

    if (env.valuesMissing()) {
      return null;
    }

    // Unconditionally check for action conflicts.
    // TODO(b/214371092): Only check when necessary.
    try (SilentCloseable c =
        Profiler.instance().profile("BuildDriverFunction.checkActionConflicts")) {
      if (state.actionConflicts == null) {
        state.actionConflicts =
            checkActionConflicts(actionLookupKey, buildDriverKey.strictActionConflictCheck());
      }
      if (!state.actionConflicts.isEmpty()) {
        throw new BuildDriverFunctionException(
            new TopLevelConflictException(
                "Action conflict(s) detected while analyzing top-level target "
                    + actionLookupKey.getLabel(),
                state.actionConflicts));
      }
    }

    Preconditions.checkState(
        topLevelSkyValue instanceof ConfiguredTargetValue
            || topLevelSkyValue instanceof TopLevelAspectsValue);
    if (topLevelSkyValue instanceof ConfiguredTargetValue) {
      requestConfiguredTargetExecution(
          ((ConfiguredTargetValue) topLevelSkyValue).getConfiguredTarget(),
          buildDriverKey,
          actionLookupKey,
          env,
          topLevelArtifactContext,
          state);
    } else {
      requestAspectExecution((TopLevelAspectsValue) topLevelSkyValue, env, topLevelArtifactContext);
    }

    return env.valuesMissing() ? null : new BuildDriverValue(topLevelSkyValue);
  }

  private void requestConfiguredTargetExecution(
      ConfiguredTarget configuredTarget,
      BuildDriverKey buildDriverKey,
      ActionLookupKey actionLookupKey,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext,
      State state)
      throws InterruptedException {
    ImmutableSet.Builder<Artifact> artifactsToBuild = ImmutableSet.builder();
    addExtraActionsIfRequested(
        configuredTarget.getProvider(ExtraActionArtifactsProvider.class), artifactsToBuild);
    if (buildDriverKey.getTestType() == NOT_TEST) {
      declareDependenciesAndCheckValues(
          env,
          Iterables.concat(
              artifactsToBuild.build(),
              Collections.singletonList(
                  TargetCompletionValue.key(
                      (ConfiguredTargetKey) actionLookupKey, topLevelArtifactContext, false))));
      return;
    }

    if (!state.sentTestAnalysisCompleteEvent) {
      SkyValue buildConfigurationValue = env.getValue(configuredTarget.getConfigurationKey());
      if (env.valuesMissing()) {
        return;
      }
      eventBus
          .get()
          .post(
              new TestAnalysisCompleteEvent(
                  configuredTarget, (BuildConfigurationValue) buildConfigurationValue));
      state.sentTestAnalysisCompleteEvent = true;
    }

    Preconditions.checkState(
        PARALLEL.equals(buildDriverKey.getTestType()),
        "Invalid test type, expect only parallel tests: %s",
        buildDriverKey);
    // Only run non-exclusive tests here. Exclusive tests need to be run sequentially later.
    declareDependenciesAndCheckValues(
        env,
        Iterables.concat(
            artifactsToBuild.build(),
            Collections.singletonList(
                TestCompletionValue.key(
                    (ConfiguredTargetKey) actionLookupKey,
                    topLevelArtifactContext,
                    /*exclusiveTesting=*/ false))));
  }

  private void requestAspectExecution(
      TopLevelAspectsValue topLevelAspectsValue,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {

    ImmutableSet.Builder<Artifact> artifactsToBuild = ImmutableSet.builder();
    List<SkyKey> aspectCompletionKeys = new ArrayList<>();
    for (SkyValue aspectValue : topLevelAspectsValue.getTopLevelAspectsValues()) {
      addExtraActionsIfRequested(
          ((AspectValue) aspectValue)
              .getConfiguredAspect()
              .getProvider(ExtraActionArtifactsProvider.class),
          artifactsToBuild);
      aspectCompletionKeys.add(
          AspectCompletionKey.create(
              ((AspectValue) aspectValue).getKey(), topLevelArtifactContext));
    }
    declareDependenciesAndCheckValues(
        env, Iterables.concat(artifactsToBuild.build(), aspectCompletionKeys));
  }

  /**
   * Declares dependencies and checks values for requested nodes in the graph.
   *
   * <p>Calls {@link SkyframeIterableResult} and iterates over the result. If any node is not done,
   * or during iteration any value has exception, {@link SkyFunction.Environment#valuesMissing} will
   * return true.
   */
  private static void declareDependenciesAndCheckValues(
      Environment env, Iterable<? extends SkyKey> skyKeys) throws InterruptedException {
    SkyframeIterableResult result = env.getOrderedValuesAndExceptions(skyKeys);
    while (result.hasNext()) {
      result.next();
    }
  }

  private ImmutableMap<ActionAnalysisMetadata, ConflictException> checkActionConflicts(
      ActionLookupKey actionLookupKey, boolean strictConflictCheck) throws InterruptedException {
    return incrementalArtifactConflictFinder
        .get()
        .findArtifactConflicts(
            transitiveActionLookupValuesCollector.collect(actionLookupKey), strictConflictCheck)
        .getConflicts();
  }

  private void addExtraActionsIfRequested(
      ExtraActionArtifactsProvider provider, ImmutableSet.Builder<Artifact> artifactsToBuild) {
    if (provider != null) {
      addArtifactsToBuilder(
          provider.getTransitiveExtraActionArtifacts().toList(), artifactsToBuild, null);
    }
  }

  private static void addArtifactsToBuilder(
      List<? extends Artifact> artifacts,
      ImmutableSet.Builder<Artifact> builder,
      RegexFilter filter) {
    for (Artifact artifact : artifacts) {
      if (filter.isIncluded(artifact.getOwnerLabel().toString())) {
        builder.add(artifact);
      }
    }
  }

  /** A SkyFunctionException wrapper for the actual TopLevelConflictException. */
  private static final class BuildDriverFunctionException extends SkyFunctionException {
    // The exception is transient here since it could be caused by external factors (conflict with
    // another target).
    BuildDriverFunctionException(TopLevelConflictException cause) {
      super(cause, Transience.TRANSIENT);
    }
  }

  interface TransitiveActionLookupValuesCollector {
    Sharder<ActionLookupValue> collect(ActionLookupKey key) throws InterruptedException;
  }
}
