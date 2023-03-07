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

import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.EXCLUSIVE;
import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.EXCLUSIVE_IF_LOCAL;
import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.NOT_TEST;
import static com.google.devtools.build.lib.skyframe.BuildDriverKey.TestType.PARALLEL;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.constraints.RuleContextConstraintSemantics;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics.EnvironmentCompatibility;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics.PlatformCompatibility;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics.TargetCompatibilityCheckException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ConflictException;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.SomeExecutionStartedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelEntityAnalysisConcludedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetPendingExecutionEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetReadyForSymlinkPlanting;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetSkippedEvent;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Drives the analysis & execution of an ActionLookupKey, which is wrapped inside a BuildDriverKey.
 */
public class BuildDriverFunction implements SkyFunction {
  private final TransitiveActionLookupValuesHelper transitiveActionLookupValuesHelper;
  private final Supplier<IncrementalArtifactConflictFinder> incrementalArtifactConflictFinder;
  private final Supplier<RuleContextConstraintSemantics> ruleContextConstraintSemantics;
  private final Supplier<RegexFilter> extraActionFilterSupplier;

  BuildDriverFunction(
      TransitiveActionLookupValuesHelper transitiveActionLookupValuesHelper,
      Supplier<IncrementalArtifactConflictFinder> incrementalArtifactConflictFinder,
      Supplier<RuleContextConstraintSemantics> ruleContextConstraintSemantics,
      Supplier<RegexFilter> extraActionFilterSupplier) {
    this.transitiveActionLookupValuesHelper = transitiveActionLookupValuesHelper;
    this.incrementalArtifactConflictFinder = incrementalArtifactConflictFinder;
    this.ruleContextConstraintSemantics = ruleContextConstraintSemantics;
    this.extraActionFilterSupplier = extraActionFilterSupplier;
  }

  private static class State implements SkyKeyComputeState {
    private ImmutableMap<ActionAnalysisMetadata, ConflictException> actionConflicts;
    // It's only necessary to do this check once.
    private boolean checkedForCompatibility = false;
    private boolean checkedForPlatformCompatibility = false;
  }
  /**
   * From the ConfiguredTarget/Aspect keys, get the top-level artifacts. Then evaluate them together
   * with the appropriate CompletionFunctions. This is the bridge between the conceptual analysis &
   * execution phases.
   *
   * <p>TODO(b/240944910): implement coverage.
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

    // This code path should not be run during error bubbling for several reasons:
    // 1. Correctness: to check for action conflicts, we need access to the transitive
    //    ConfiguredTargets, which will be null after AnalysisPhaseCompleteEvent in
    //    --discard_analysis_cache mode.
    // 2. Performance: this method is CPU intensive, and it does not offer anything while error
    //    bubbling.
    if (!env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors()) {
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
    }

    Preconditions.checkState(
        topLevelSkyValue instanceof ConfiguredTargetValue
            || topLevelSkyValue instanceof TopLevelAspectsValue);
    if (topLevelSkyValue instanceof ConfiguredTargetValue) {
      ConfiguredTargetValue configuredTargetValue = (ConfiguredTargetValue) topLevelSkyValue;
      ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
      // At this point, the target is considered "analyzed". It's important that this event is sent
      // before the TopLevelEntityAnalysisConcludedEvent: when the last of the analysis work is
      // concluded, we need to have the *complete* list of analyzed targets ready in
      // BuildResultListener.
      postTopLevelTargetAnalyzedEvent(env, configuredTargetValue, configuredTarget);

      BuildConfigurationValue buildConfigurationValue =
          configuredTarget.getConfigurationKey() == null
              ? null
              : (BuildConfigurationValue) env.getValue(configuredTarget.getConfigurationKey());
      if (env.valuesMissing()) {
        return null;
      }

      if (!state.checkedForCompatibility) {
        try {
          Boolean isConfiguredTargetCompatible =
              isConfiguredTargetCompatible(
                  env,
                  state,
                  configuredTarget,
                  buildConfigurationValue,
                  buildDriverKey.isExplicitlyRequested());
          if (isConfiguredTargetCompatible == null) {
            return null;
          }

          state.checkedForCompatibility = true;
          if (!isConfiguredTargetCompatible) {
            env.getListener().post(TopLevelTargetSkippedEvent.create(configuredTarget));
            // We still record analyzed but skipped tests, as this information is needed for the
            // result summary.
            if (!NOT_TEST.equals(buildDriverKey.getTestType())) {
              env.getListener()
                  .post(
                      TestAnalyzedEvent.create(
                          configuredTarget,
                          Preconditions.checkNotNull(buildConfigurationValue),
                          /*isSkipped=*/ true));
            }
            // Only send the event now to include the compatibility check in the measurement for
            // time spent on analysis work.
            env.getListener().post(TopLevelEntityAnalysisConcludedEvent.success(buildDriverKey));
            // We consider the evaluation of this BuildDriverKey successful at this point, even when
            // the target is skipped.
            return new BuildDriverValue(topLevelSkyValue, /*skipped=*/ true);
          }
        } catch (TargetCompatibilityCheckException e) {
          throw new BuildDriverFunctionException(e);
        }
      }

      env.getListener().post(TopLevelEntityAnalysisConcludedEvent.success(buildDriverKey));
      env.getListener()
          .post(
              TopLevelTargetPendingExecutionEvent.create(
                  configuredTarget, buildDriverKey.isTest()));
      requestConfiguredTargetExecution(
          configuredTarget,
          buildDriverKey,
          actionLookupKey,
          buildConfigurationValue,
          env,
          topLevelArtifactContext);
    } else {
      announceAspectAnalysisDoneAndRequestExecution(
          buildDriverKey, (TopLevelAspectsValue) topLevelSkyValue, env, topLevelArtifactContext);
    }

    if (env.valuesMissing()) {
      return null;
    }

    // If we get to this point, the execution of this target/aspect succeeded.

    if (EXCLUSIVE.equals(buildDriverKey.getTestType())
        || EXCLUSIVE_IF_LOCAL.equals(buildDriverKey.getTestType())) {
      Preconditions.checkState(topLevelSkyValue instanceof ConfiguredTargetValue);
      return new ExclusiveTestBuildDriverValue(
          topLevelSkyValue, ((ConfiguredTargetValue) topLevelSkyValue).getConfiguredTarget());
    }

    return new BuildDriverValue(topLevelSkyValue, /*skipped=*/ false);
  }

  private static void postTopLevelTargetAnalyzedEvent(
      Environment env,
      ConfiguredTargetValue configuredTargetValue,
      ConfiguredTarget configuredTarget) {
    env.getListener().post(TopLevelTargetAnalyzedEvent.create(configuredTarget));
    // It's possible that this code path is triggered AFTER the analysis cache clean up and the
    // transitive packages for package root resolution is already cleared. In such a case, the
    // symlinks should have already been planted.
    if (configuredTargetValue.getTransitivePackages() != null) {
      env.getListener()
          .post(
              TopLevelTargetReadyForSymlinkPlanting.create(
                  configuredTargetValue.getTransitivePackages()));
    }
  }

  /**
   * Checks if a ConfiguredTarget is compatible with the platform/environment. See {@link
   * TopLevelConstraintSemantics}.
   *
   * @return null if a value is missing in the environment.
   */
  @Nullable
  private Boolean isConfiguredTargetCompatible(
      Environment env,
      State state,
      ConfiguredTarget configuredTarget,
      BuildConfigurationValue buildConfigurationValue,
      boolean isExplicitlyRequested)
      throws InterruptedException, TargetCompatibilityCheckException {

    if (!state.checkedForPlatformCompatibility) {
      PlatformCompatibility platformCompatibility =
          TopLevelConstraintSemantics.compatibilityWithPlatformRestrictions(
              configuredTarget,
              env.getListener(),
              /*eagerlyThrowError=*/ true,
              isExplicitlyRequested);
      state.checkedForPlatformCompatibility = true;
      switch (platformCompatibility) {
        case INCOMPATIBLE_EXPLICIT:
        case INCOMPATIBLE_IMPLICIT:
          return false;
        case COMPATIBLE:
          break;
      }
    }

    EnvironmentCompatibility environmentCompatibility =
        TopLevelConstraintSemantics.compatibilityWithTargetEnvironment(
            configuredTarget,
            buildConfigurationValue,
            label -> getTarget(env, label),
            env.getListener());
    if (env.valuesMissing() || environmentCompatibility == null) {
      return null;
    }
    if (environmentCompatibility.isCompatible()) {
      return true;
    }
    if (environmentCompatibility.severeMissingEnvironments() == null) {
      return false;
    }
    String badTargetsUserMessage =
        TopLevelConstraintSemantics.getErrorMessageForTarget(
            ruleContextConstraintSemantics.get(),
            configuredTarget,
            environmentCompatibility.severeMissingEnvironments());
    throw new TargetCompatibilityCheckException(
        badTargetsUserMessage,
        FailureDetail.newBuilder()
            .setMessage(badTargetsUserMessage)
            .setAnalysis(Analysis.newBuilder().setCode(Code.TARGETS_MISSING_ENVIRONMENTS))
            .build());
  }

  @Nullable
  private static Target getTarget(Environment env, Label label)
      throws InterruptedException, NoSuchTargetException {
    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(label.getPackageIdentifier()));
    if (env.valuesMissing() || packageValue == null) {
      return null;
    }
    Package pkg = packageValue.getPackage();
    return pkg.getTarget(label.getName());
  }

  private void requestConfiguredTargetExecution(
      ConfiguredTarget configuredTarget,
      BuildDriverKey buildDriverKey,
      ActionLookupKey actionLookupKey,
      BuildConfigurationValue buildConfigurationValue,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    ImmutableSet.Builder<Artifact> artifactsToBuild = ImmutableSet.builder();
    addExtraActionsIfRequested(
        configuredTarget.getProvider(ExtraActionArtifactsProvider.class), artifactsToBuild);
    env.getListener().post(SomeExecutionStartedEvent.create());
    if (NOT_TEST.equals(buildDriverKey.getTestType())) {
      declareDependenciesAndCheckValues(
          env,
          Iterables.concat(
              Artifact.keys(artifactsToBuild.build()),
              Collections.singletonList(
                  TargetCompletionValue.key(
                      (ConfiguredTargetKey) actionLookupKey, topLevelArtifactContext, false))));
      return;
    }

    env.getListener()
        .post(
            TestAnalyzedEvent.create(
                configuredTarget,
                Preconditions.checkNotNull(buildConfigurationValue),
                /*isSkipped=*/ false));

    if (PARALLEL.equals(buildDriverKey.getTestType())) {
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
      return;
    }

    // Exclusive tests will be run with sequential Skyframe evaluations afterwards.
    declareDependenciesAndCheckValues(env, artifactsToBuild.build());
  }

  private void announceAspectAnalysisDoneAndRequestExecution(
      BuildDriverKey buildDriverKey,
      TopLevelAspectsValue topLevelAspectsValue,
      Environment env,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {

    env.getListener().post(SomeExecutionStartedEvent.create());
    ImmutableSet.Builder<Artifact> artifactsToBuild = ImmutableSet.builder();
    List<SkyKey> aspectCompletionKeys = new ArrayList<>();
    for (AspectValue aspectValue : topLevelAspectsValue.getTopLevelAspectsValues()) {
      AspectKey aspectKey = aspectValue.getKey();
      ConfiguredAspect configuredAspect = aspectValue.getConfiguredAspect();
      addExtraActionsIfRequested(
          configuredAspect.getProvider(ExtraActionArtifactsProvider.class), artifactsToBuild);

      // It's possible that this code path is triggered AFTER the analysis cache clean up and the
      // transitive packages for package root resolution is already cleared. In such a case, the
      // symlinks should have already been planted.
      if (aspectValue.getTransitivePackages() != null) {
        env.getListener()
            .post(
                TopLevelTargetReadyForSymlinkPlanting.create(aspectValue.getTransitivePackages()));
      }
      env.getListener().post(AspectAnalyzedEvent.create(aspectKey, configuredAspect));

      aspectCompletionKeys.add(AspectCompletionKey.create(aspectKey, topLevelArtifactContext));
    }
    // Send the AspectAnalyzedEvents first to make sure the BuildResultListener is up-to-date before
    // signaling that the analysis of this top level aspect has concluded.
    env.getListener().post(TopLevelEntityAnalysisConcludedEvent.success(buildDriverKey));

    declareDependenciesAndCheckValues(
        env, Iterables.concat(Artifact.keys(artifactsToBuild.build()), aspectCompletionKeys));
  }

  /**
   * Declares dependencies and checks values for requested nodes in the graph.
   *
   * <p>Calls {@link SkyFunction.Environment#getValuesAndExceptions} and iterates over the result.
   * If any node is not done, or during iteration any value has exception, {@link
   * SkyFunction.Environment#valuesMissing} will return true.
   */
  private static void declareDependenciesAndCheckValues(
      Environment env, Iterable<? extends SkyKey> skyKeys) throws InterruptedException {
    SkyframeLookupResult result = env.getValuesAndExceptions(skyKeys);
    for (SkyKey key : skyKeys) {
      if (result.get(key) == null) {
        return;
      }
    }
  }

  @VisibleForTesting
  ImmutableMap<ActionAnalysisMetadata, ConflictException> checkActionConflicts(
      ActionLookupKey actionLookupKey, boolean strictConflictCheck) throws InterruptedException {
    ActionLookupValuesCollectionResult transitiveValueCollectionResult =
        transitiveActionLookupValuesHelper.collect(actionLookupKey);

    ImmutableMap<ActionAnalysisMetadata, ConflictException> conflicts =
        incrementalArtifactConflictFinder
            .get()
            .findArtifactConflicts(
                transitiveValueCollectionResult.collectedValues(), strictConflictCheck)
            .getConflicts();
    if (conflicts.isEmpty()) {
      transitiveActionLookupValuesHelper.registerConflictFreeKeys(
          transitiveValueCollectionResult.visitedKeys());
    }
    return conflicts;
  }

  private void addExtraActionsIfRequested(
      ExtraActionArtifactsProvider provider, ImmutableSet.Builder<Artifact> artifactsToBuild) {
    if (provider != null) {
      addArtifactsToBuilder(
          provider.getTransitiveExtraActionArtifacts().toList(),
          artifactsToBuild,
          extraActionFilterSupplier.get());
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

    BuildDriverFunctionException(TargetCompatibilityCheckException cause) {
      super(cause, Transience.TRANSIENT);
    }
  }

  interface TransitiveActionLookupValuesHelper {

    /**
     * Perform the traversal of the transitive closure of the {@code key} and collect the
     * corresponding ActionLookupValues.
     */
    ActionLookupValuesCollectionResult collect(ActionLookupKey key) throws InterruptedException;

    /** Register with the helper that the {@code keys} are conflict-free. */
    void registerConflictFreeKeys(ImmutableSet<SkyKey> keys);
  }

  @AutoValue
  abstract static class ActionLookupValuesCollectionResult {
    abstract Sharder<ActionLookupValue> collectedValues();

    abstract ImmutableSet<SkyKey> visitedKeys();

    static ActionLookupValuesCollectionResult create(
        Sharder<ActionLookupValue> collectedValues, ImmutableSet<SkyKey> visitedKeys) {
      return new AutoValue_BuildDriverFunction_ActionLookupValuesCollectionResult(
          collectedValues, visitedKeys);
    }
  }
}
