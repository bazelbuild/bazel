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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Predicates;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent.ErrorTiming;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputDepOwnerMap;
import com.google.devtools.build.lib.actions.ActionInputDepOwners;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputMapSink;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.DiscoveredInputsEvent;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.MissingArtifactValue;
import com.google.devtools.build.lib.skyframe.ArtifactFunction.SourceArtifactException;
import com.google.devtools.build.lib.skyframe.ArtifactNestedSetFunction.ArtifactNestedSetEvalException;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionPostprocessing;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy.RewindPlan;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewoundEvent;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A {@link SkyFunction} that creates {@link ActionExecutionValue}s. There are four points where
 * this function can abort due to missing values in the graph:
 *
 * <ol>
 *   <li>For actions that discover inputs, if missing metadata needed to resolve an artifact from a
 *       string input in the action cache.
 *   <li>If missing metadata for artifacts in inputs (including the artifacts above).
 *   <li>For actions that discover inputs, if missing metadata for inputs discovered prior to
 *       execution.
 *   <li>For actions that discover inputs, but do so during execution, if missing metadata for
 *       inputs discovered during execution.
 * </ol>
 *
 * <p>If async action execution is enabled, or if a non-primary shared action coalesces with an
 * in-flight primary shared action's execution, this function can abort after declaring an external
 * dep on the execution's completion future.
 */
public final class ActionExecutionFunction implements SkyFunction {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ActionRewindStrategy actionRewindStrategy = new ActionRewindStrategy();
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final BlazeDirectories directories;
  private final Supplier<TimestampGranularityMonitor> tsgm;
  private final BugReporter bugReporter;

  public ActionExecutionFunction(
      SkyframeActionExecutor skyframeActionExecutor,
      BlazeDirectories directories,
      Supplier<TimestampGranularityMonitor> tsgm,
      BugReporter bugReporter) {
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.directories = directories;
    this.tsgm = tsgm;
    this.bugReporter = bugReporter;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ActionExecutionFunctionException, InterruptedException {
    try {
      return computeInternal(skyKey, env);
    } catch (ActionExecutionFunctionException e) {
      skyframeActionExecutor.recordExecutionError();
      throw e;
    }
  }

  @Nullable
  private SkyValue computeInternal(SkyKey skyKey, Environment env)
      throws ActionExecutionFunctionException, InterruptedException {
    ActionLookupData actionLookupData = (ActionLookupData) skyKey.argument();
    Action action = ActionUtils.getActionForLookupData(env, actionLookupData);
    if (action == null) {
      return null;
    }
    skyframeActionExecutor.noteActionEvaluationStarted(actionLookupData, action);
    if (Actions.dependsOnBuildId(action)) {
      PrecomputedValue.BUILD_ID.get(env);
    }

    // Look up the parts of the environment that influence the action.
    Collection<String> clientEnvironmentVariables = action.getClientEnvironmentVariables();
    ImmutableMap<String, String> clientEnv;
    if (!clientEnvironmentVariables.isEmpty()) {
      ImmutableSet<String> clientEnvironmentVariablesSet =
          ImmutableSet.copyOf(clientEnvironmentVariables);
      Iterable<SkyKey> depKeys =
          Iterables.transform(clientEnvironmentVariablesSet, ClientEnvironmentFunction::key);
      SkyframeLookupResult clientEnvLookup = env.getValuesAndExceptions(depKeys);
      if (env.valuesMissing()) {
        return null;
      }
      ImmutableMap.Builder<String, String> builder =
          ImmutableMap.builderWithExpectedSize(clientEnvironmentVariablesSet.size());
      for (SkyKey depKey : depKeys) {
        ClientEnvironmentValue envValue = (ClientEnvironmentValue) clientEnvLookup.get(depKey);
        if (envValue.getValue() != null) {
          builder.put((String) depKey.argument(), envValue.getValue());
        }
      }
      clientEnv = builder.buildOrThrow();
    } else {
      clientEnv = ImmutableMap.of();
    }

    // If two actions are shared and the first one executes, when the second one goes to execute, we
    // should detect that and short-circuit.
    //
    // Additionally, if an action restarted (in the Skyframe sense) after it executed because it
    // discovered new inputs during execution, we should detect that and short-circuit.
    //
    // Separately, we use InputDiscoveryState to avoid redoing work on Skyframe restarts for actions
    // that discover inputs. This is not [currently] relevant here, because it is [currently] not
    // possible for an action to both be shared and also discover inputs; see b/72764586.
    ActionExecutionState previousExecution = skyframeActionExecutor.probeActionExecution(action);

    // If this action was previously completed this build, then this evaluation must be happening
    // because of rewinding. Prevent any progress events from being published a second time for this
    // action; downstream consumers of action events reasonably don't expect them.
    if (!skyframeActionExecutor.shouldEmitProgressEvents(action)) {
      env = new ProgressEventSuppressingEnvironment(env);
    }

    if (action.discoversInputs()) {
      // If this action previously failed due to a lost input found during input discovery, ensure
      // that the input is regenerated before attempting discovery again.
      if (declareDepsOnLostDiscoveredInputsIfAny(env, action)) {
        return null;
      }
    }

    InputDiscoveryState state;
    if (action.discoversInputs()) {
      state = env.getState(InputDiscoveryState::new);
    } else {
      // Because this is a new state, all conditionals below about whether state has already done
      // something will return false, and so we will execute all necessary steps.
      state = new InputDiscoveryState();
    }
    if (!state.hasCollectedInputs()) {
      try {
        state.allInputs = collectInputs(action, env);
      } catch (AlreadyReportedActionExecutionException e) {
        throw new ActionExecutionFunctionException(e);
      }
      state.requestedArtifactNestedSetKeys = null;
      if (state.allInputs == null) {
        // Missing deps.
        return null;
      }
    }

    CheckInputResults checkedInputs = null;
    NestedSet<Artifact> allInputs =
        state.allInputs.getAllInputs(
            skyframeActionExecutor.actionFileSystemType().supportsInputDiscovery());

    if (!state.actionInputCollectedEventSent) {
      env.getListener()
          .post(
              ActionInputCollectedEvent.create(
                  action, allInputs, skyframeActionExecutor.getActionContextRegistry()));
      state.actionInputCollectedEventSent = true;
    }

    if (!state.hasArtifactData()) {
      Iterable<SkyKey> depKeys = getInputDepKeys(allInputs, state);
      SkyframeLookupResult inputDeps = env.getValuesAndExceptions(depKeys);
      if (previousExecution == null) {
        // Do we actually need to find our metadata?
        try {
          checkedInputs = checkInputs(env, action, inputDeps, allInputs, depKeys, actionLookupData);
        } catch (ActionExecutionException e) {
          throw new ActionExecutionFunctionException(e);
        }
      }
      if (env.valuesMissing()) {
        // There was missing artifact metadata in the graph. Wait for it to be present.
        // We must check this and return here before attempting to establish any Skyframe
        // dependencies of the action; see establishSkyframeDependencies why.
        return null;
      }
    }

    Object skyframeDepsResult;
    try {
      skyframeDepsResult = establishSkyframeDependencies(env, action);
    } catch (ActionExecutionException e) {
      throw new ActionExecutionFunctionException(
          skyframeActionExecutor.processAndGetExceptionToThrow(
              env.getListener(),
              /*primaryOutputPath=*/ null,
              action,
              e,
              new FileOutErr(),
              ErrorTiming.BEFORE_EXECUTION));
    }
    if (env.valuesMissing()) {
      return null;
    }

    if (checkedInputs != null) {
      checkState(!state.hasArtifactData(), "%s %s", state, action);
      state.inputArtifactData = checkedInputs.actionInputMap;
      state.expandedArtifacts = checkedInputs.expandedArtifacts;
      state.archivedTreeArtifacts = checkedInputs.archivedTreeArtifacts;
      state.filesetsInsideRunfiles = checkedInputs.filesetsInsideRunfiles;
      state.topLevelFilesets = checkedInputs.topLevelFilesets;
      if (skyframeActionExecutor.actionFileSystemType().isEnabled()) {
        state.actionFileSystem =
            skyframeActionExecutor.createActionFileSystem(
                directories.getRelativeOutputPath(),
                checkedInputs.actionInputMap,
                action.getOutputs(),
                env.restartPermitted());
      }
    }

    long actionStartTime = BlazeClock.nanoTime();
    ActionExecutionValue result;
    try {
      result =
          checkCacheAndExecuteIfNeeded(
              action,
              state,
              env,
              clientEnv,
              actionLookupData,
              previousExecution,
              skyframeDepsResult,
              actionStartTime);
    } catch (LostInputsActionExecutionException e) {
      return handleLostInputs(
          e,
          actionLookupData,
          action,
          actionStartTime,
          env,
          allInputs,
          getInputDepKeys(allInputs, state),
          state);
    } catch (ActionExecutionException e) {
      // In this case we do not report the error to the action reporter because we have already
      // done it in SkyframeActionExecutor.reportErrorIfNotAbortingMode() method. That method
      // prints the error in the top-level reporter and also dumps the recorded StdErr for the
      // action. Label can be null in the case of, e.g., the SystemActionOwner (for build-info.txt).
      throw new ActionExecutionFunctionException(new AlreadyReportedActionExecutionException(e));
    }

    if (env.valuesMissing()) {
      // This usually happens only for input-discovering actions. Other actions may have
      // valuesMissing() here in rare circumstances related to Fileset inputs being unavailable.
      // See comments in ActionInputMapHelper#getFilesets().
      return null;
    }

    return result;
  }

  private static Iterable<SkyKey> getInputDepKeys(
      NestedSet<Artifact> allInputs, InputDiscoveryState state) {
    // We "unwrap" the NestedSet and evaluate the first layer of direct Artifacts here in order to
    // save memory:
    // - This top layer costs 1 extra ArtifactNestedSetKey node.
    // - It's uncommon that 2 actions share the exact same set of inputs
    //   => the top layer offers little in terms of reusability.
    // More details: b/143205147.
    List<SkyKey> directKeys = Artifact.keys(allInputs.getLeaves());
    if (state.requestedArtifactNestedSetKeys == null) {
      state.requestedArtifactNestedSetKeys =
          CompactHashSet.create(
              Lists.transform(allInputs.getNonLeaves(), ArtifactNestedSetKey::create));
    }
    return Iterables.concat(directKeys, state.requestedArtifactNestedSetKeys);
  }

  private boolean declareDepsOnLostDiscoveredInputsIfAny(Environment env, Action action)
      throws InterruptedException, ActionExecutionFunctionException {
    ImmutableList<SkyKey> previouslyLostDiscoveredInputs =
        skyframeActionExecutor.getLostDiscoveredInputs(action);
    if (previouslyLostDiscoveredInputs != null) {
      SkyframeLookupResult lostInputValues =
          env.getValuesAndExceptions(previouslyLostDiscoveredInputs);
      if (env.valuesMissing()) {
        return true;
      }
      for (SkyKey lostInputKey : previouslyLostDiscoveredInputs) {
        try {
          SkyValue value =
              lostInputValues.getOrThrow(
                  lostInputKey, MissingInputFileException.class, ActionExecutionException.class);
          if (value == null) {
            return true;
          }
        } catch (MissingInputFileException e) {
          // MissingInputFileException comes from problems with source artifact construction.
          // Rewinding never invalidates source artifacts.
          throw new IllegalStateException(
              "MissingInputFileException unexpected from rewound generated discovered input. key="
                  + lostInputKey,
              e);
        } catch (ActionExecutionException e) {
          throw new ActionExecutionFunctionException(e);
        }
      }
    }
    return false;
  }

  /**
   * Clean up state associated with the current action execution attempt and return a {@link
   * Restart} value which rewinds the actions that generate the lost inputs.
   */
  private SkyFunction.Restart handleLostInputs(
      LostInputsActionExecutionException e,
      ActionLookupData actionLookupData,
      Action action,
      long actionStartTimeNanos,
      Environment env,
      NestedSet<Artifact> allInputs,
      Iterable<SkyKey> depKeys,
      InputDiscoveryState state)
      throws InterruptedException, ActionExecutionFunctionException {
    checkState(
        e.isPrimaryAction(actionLookupData),
        "non-primary action handling lost inputs exception: %s %s",
        actionLookupData,
        e);
    RewindPlan rewindPlan = null;
    try {
      ActionInputDepOwners inputDepOwners =
          createAugmentedInputDepOwners(e, action, env, allInputs, actionLookupData);

      // Collect the set of direct deps of this action which may be responsible for the lost inputs,
      // some of which may be discovered.
      ImmutableList<SkyKey> lostDiscoveredInputs = ImmutableList.of();
      ImmutableSet<SkyKey> failedActionDeps;
      if (e.isFromInputDiscovery()) {
        // Lost inputs found during input discovery are necessarily ordinary derived artifacts.
        // Their keys may not be direct deps yet, so to ensure that when this action is restarted
        // the lost inputs' generating actions are requested, they're added to
        // SkyframeActionExecutor's lostDiscoveredInputsMap. Also, lost inputs from input discovery
        // may come from nested sets, which may be directly represented in skyframe. To ensure that
        // applicable nested set nodes are rewound, this action's deps are also considered when
        // computing the rewind plan.
        lostDiscoveredInputs =
            e.getLostInputs().values().stream()
                .map(input -> Artifact.key((Artifact) input))
                .collect(toImmutableList());
        failedActionDeps =
            ImmutableSet.<SkyKey>builder().addAll(depKeys).addAll(lostDiscoveredInputs).build();
      } else if (state.discoveredInputs != null) {
        failedActionDeps =
            ImmutableSet.<SkyKey>builder()
                .addAll(depKeys)
                .addAll(Lists.transform(state.discoveredInputs.toList(), Artifact::key))
                .build();
      } else {
        failedActionDeps = ImmutableSet.copyOf(depKeys);
      }

      try {
        rewindPlan =
            actionRewindStrategy.getRewindPlan(
                action, actionLookupData, failedActionDeps, e, inputDepOwners, env);
      } catch (ActionExecutionException rewindingFailedException) {
        // This ensures coalesced shared actions aren't orphaned.
        skyframeActionExecutor.resetRewindingAction(actionLookupData, action, lostDiscoveredInputs);
        throw new ActionExecutionFunctionException(
            new AlreadyReportedActionExecutionException(
                skyframeActionExecutor.processAndGetExceptionToThrow(
                    env.getListener(),
                    e.getPrimaryOutputPath(),
                    action,
                    rewindingFailedException,
                    e.getFileOutErr(),
                    ActionExecutedEvent.ErrorTiming.AFTER_EXECUTION)));
      }

      if (e.isActionStartedEventAlreadyEmitted()) {
        env.getListener()
            .post(new ActionRewoundEvent(actionStartTimeNanos, BlazeClock.nanoTime(), action));
      }
      skyframeActionExecutor.resetRewindingAction(actionLookupData, action, lostDiscoveredInputs);
      for (Action actionToRestart : rewindPlan.getAdditionalActionsToRestart()) {
        skyframeActionExecutor.resetPreviouslyCompletedAction(actionLookupData, actionToRestart);
      }
      return rewindPlan.getNodesToRestart();
    } finally {
      if (rewindPlan == null && e.isActionStartedEventAlreadyEmitted()) {
        // Rewinding was unsuccessful. SkyframeActionExecutor's ActionRunner didn't emit an
        // ActionCompletionEvent because it hoped rewinding would fix things. Because it won't, this
        // must emit one to compensate.
        env.getListener()
            .post(
                new ActionCompletionEvent(
                    actionStartTimeNanos, BlazeClock.nanoTime(), action, actionLookupData));
      }
    }
  }

  /**
   * Returns an augmented version of {@code e.getOwners()}'s {@link ActionInputDepOwners}, adding
   * ownership information from {@code inputDeps}.
   *
   * <p>This compensates for how the ownership information in {@code e.getOwners()} is potentially
   * incomplete. E.g., it may lack knowledge of a runfiles middleman owning a fileset, even if it
   * knows that fileset owns a lost input.
   */
  private ActionInputDepOwners createAugmentedInputDepOwners(
      LostInputsActionExecutionException e,
      Action action,
      Environment env,
      NestedSet<Artifact> allInputs,
      ActionLookupData actionLookupDataForError)
      throws InterruptedException {

    Set<ActionInput> lostInputsAndOwnersSoFar = new HashSet<>();
    ActionInputDepOwners owners = e.getOwners();
    for (ActionInput lostInput : e.getLostInputs().values()) {
      lostInputsAndOwnersSoFar.add(lostInput);
      lostInputsAndOwnersSoFar.addAll(owners.getDepOwners(lostInput));
    }

    ActionInputDepOwnerMap inputDepOwners;
    try {
      inputDepOwners =
          getInputDepOwners(
              env,
              action,
              allInputs,
              lostInputsAndOwnersSoFar,
              actionLookupDataForError);
    } catch (ActionExecutionException unexpected) {
      // getInputDepOwners should not be able to throw, because it does the same work as
      // checkInputs, so if getInputDepOwners throws then checkInputs should have thrown, and if
      // checkInputs threw then we shouldn't have reached this point in action execution.
      throw new IllegalStateException(unexpected);
    }

    // Ownership information from inputDeps may be incomplete. Notably, it does not expand
    // filesets. Fileset and other ownership relationships should have been captured in the
    // exception's ActionInputDepOwners, and this copies that knowledge into the augmented version.
    for (ActionInput lostInput : e.getLostInputs().values()) {
      for (Artifact depOwner : owners.getDepOwners(lostInput)) {
        inputDepOwners.addOwner(lostInput, depOwner);
      }
    }
    return inputDepOwners;
  }

  /**
   * An action's inputs needed for execution. May not just be the result of Action#getInputs(). If
   * the action cache's view of this action contains additional inputs, it will request metadata for
   * them, so we consider those inputs as dependencies of this action as well. Returns null if some
   * dependencies were missing and this ActionExecutionFunction needs to restart.
   */
  @Nullable
  private AllInputs collectInputs(Action action, Environment env)
      throws InterruptedException, AlreadyReportedActionExecutionException {
    NestedSet<Artifact> allKnownInputs = action.getInputs();
    if (action.inputsKnown()) {
      return new AllInputs(allKnownInputs);
    }

    checkState(action.discoversInputs(), action);
    PackageRootResolverWithEnvironment resolver = new PackageRootResolverWithEnvironment(env);
    List<Artifact> actionCacheInputs =
        skyframeActionExecutor.getActionCachedInputs(action, resolver);
    if (actionCacheInputs == null) {
      checkState(env.valuesMissing(), action);
      return null;
    }
    return new AllInputs(
        allKnownInputs,
        action.getSchedulingDependencies(),
        actionCacheInputs,
        action.getAllowedDerivedInputs(),
        resolver.packageLookupsRequested);
  }

  static class AllInputs {
    final NestedSet<Artifact> defaultInputs;
    final NestedSet<Artifact> schedulingDependencies;
    @Nullable final NestedSet<Artifact> allowedDerivedInputs;
    @Nullable final List<Artifact> actionCacheInputs;
    @Nullable final List<ContainingPackageLookupValue.Key> packageLookupsRequested;

    AllInputs(NestedSet<Artifact> defaultInputs) {
      this.defaultInputs = checkNotNull(defaultInputs);
      this.schedulingDependencies = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      this.actionCacheInputs = null;
      this.allowedDerivedInputs = null;
      this.packageLookupsRequested = null;
    }

    AllInputs(
        NestedSet<Artifact> defaultInputs,
        NestedSet<Artifact> schedulingDependencies,
        List<Artifact> actionCacheInputs,
        NestedSet<Artifact> allowedDerivedInputs,
        List<ContainingPackageLookupValue.Key> packageLookupsRequested) {
      this.defaultInputs = checkNotNull(defaultInputs);
      this.schedulingDependencies = checkNotNull(schedulingDependencies);
      this.allowedDerivedInputs = checkNotNull(allowedDerivedInputs);
      this.actionCacheInputs = checkNotNull(actionCacheInputs);
      this.packageLookupsRequested = packageLookupsRequested;
    }

    /**
     * Compute the inputs to request from Skyframe.
     *
     * @param prune If true, only return default inputs and any inputs from action cache checker.
     *     Otherwise, return default inputs and all possible derived inputs of the action. Bazel's
     *     {@link com.google.devtools.build.lib.remote.RemoteActionFileSystem} requires the metadata
     *     from all derived inputs to know if they are remote or not during input discovery.
     */
    NestedSet<Artifact> getAllInputs(boolean prune) {
      NestedSetBuilder<Artifact> builder = new NestedSetBuilder<>(Order.STABLE_ORDER);
      builder.addTransitive(defaultInputs);
      builder.addTransitive(schedulingDependencies);

      if (actionCacheInputs == null) {
        return builder.build();
      }

      if (prune) {
        // actionCacheInputs is never a NestedSet.
        builder.addAll(actionCacheInputs);
      } else {
        builder.addTransitive(allowedDerivedInputs);
      }

      return builder.build();
    }
  }

  /**
   * Skyframe implementation of {@link PackageRootResolver}. Should be used only from SkyFunctions,
   * because it uses SkyFunction.Environment for evaluation of ContainingPackageLookupValue.
   */
  private static class PackageRootResolverWithEnvironment implements PackageRootResolver {
    final List<ContainingPackageLookupValue.Key> packageLookupsRequested = new ArrayList<>();
    private final Environment env;

    private PackageRootResolverWithEnvironment(Environment env) {
      this.env = env;
    }

    @Nullable
    @Override
    public Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
        throws PackageRootException, InterruptedException {
      checkState(
          packageLookupsRequested.isEmpty(),
          "resolver should only be called once: %s %s",
          packageLookupsRequested,
          execPaths);
      StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
      if (starlarkSemantics == null) {
        return null;
      }

      boolean siblingRepositoryLayout =
          starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT);

      // Create SkyKeys list based on execPaths.
      Map<PathFragment, ContainingPackageLookupValue.Key> depKeys = new HashMap<>();
      for (PathFragment path : execPaths) {
        PathFragment parent =
            checkNotNull(path.getParentDirectory(), "Must pass in files, not root directory");
        checkArgument(!parent.isAbsolute(), path);
        ContainingPackageLookupValue.Key depKey =
            ContainingPackageLookupValue.key(
                PackageIdentifier.discoverFromExecPath(path, true, siblingRepositoryLayout));
        depKeys.put(path, depKey);
        packageLookupsRequested.add(depKey);
      }

      SkyframeLookupResult values = env.getValuesAndExceptions(depKeys.values());
      Map<PathFragment, Root> result = new HashMap<>();
      for (PathFragment path : execPaths) {
        if (!depKeys.containsKey(path)) {
          continue;
        }
        ContainingPackageLookupValue value;
        try {
          value =
              (ContainingPackageLookupValue)
                  values.getOrThrow(
                      depKeys.get(path),
                      BuildFileNotFoundException.class,
                      InconsistentFilesystemException.class);
        } catch (BuildFileNotFoundException e) {
          throw PackageRootException.create(path, e);
        } catch (InconsistentFilesystemException e) {
          throw PackageRootException.create(path, e);
        }
        if (value != null && value.hasContainingPackage()) {
          // We have found corresponding root for current execPath.
          result.put(path, value.getContainingPackageRoot());
        } else {
          // We haven't found corresponding root for current execPath.
          result.put(path, null);
        }
      }
      return env.valuesMissing() ? null : result;
    }
  }

  @Nullable
  private ActionExecutionValue checkCacheAndExecuteIfNeeded(
      Action action,
      InputDiscoveryState state,
      Environment env,
      Map<String, String> clientEnv,
      ActionLookupData actionLookupData,
      @Nullable ActionExecutionState previousAction,
      Object skyframeDepsResult,
      long actionStartTime)
      throws ActionExecutionException, InterruptedException {
    if (previousAction != null) {
      // There are two cases where we can already have an ActionExecutionState for a specific
      // output:
      // 1. Another instance of a shared action won the race and got executed first.
      // 2. The action was already started earlier, and this SkyFunction got restarted since
      //    there's progress to be made.
      // In either case, we must use this ActionExecutionState to continue. Note that in the first
      // case, we don't have any input metadata available, so we couldn't re-execute the action even
      // if we wanted to.
      return previousAction.getResultOrDependOnFuture(
          env,
          actionLookupData,
          action,
          skyframeActionExecutor.getSharedActionCallback(
              env.getListener(), state.discoveredInputs != null, action, actionLookupData));
    }

    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets =
        state.getExpandedFilesets();

    ArtifactExpander artifactExpander =
        new Artifact.ArtifactExpanderImpl(
            Collections.unmodifiableMap(state.expandedArtifacts),
            Collections.unmodifiableMap(state.archivedTreeArtifacts),
            expandedFilesets);

    ArtifactPathResolver pathResolver =
        ArtifactPathResolver.createPathResolver(
            state.actionFileSystem, skyframeActionExecutor.getExecRoot());
    ActionMetadataHandler metadataHandler =
        ActionMetadataHandler.create(
            state.inputArtifactData,
            skyframeActionExecutor.useArchivedTreeArtifacts(action),
            skyframeActionExecutor.getOutputPermissions(),
            ImmutableSet.copyOf(action.getOutputs()),
            skyframeActionExecutor.getXattrProvider(),
            tsgm.get(),
            pathResolver,
            skyframeActionExecutor.getExecRoot().asFragment(),
            expandedFilesets);

    // We only need to check the action cache if we haven't done it on a previous run.
    if (!state.hasCheckedActionCache()) {
      state.token =
          skyframeActionExecutor.checkActionCache(
              env.getListener(),
              action,
              metadataHandler,
              metadataHandler,
              artifactExpander,
              actionStartTime,
              state.allInputs.actionCacheInputs,
              clientEnv);
    }

    if (state.token == null) {
      // We got a hit from the action cache -- no need to execute.
      checkState(
          !(action instanceof SkyframeAwareAction),
          "Error, we're not re-executing a "
              + "SkyframeAwareAction which should be re-executed unconditionally. Action: %s",
          action);
      return ActionExecutionValue.createFromOutputStore(
          metadataHandler.getOutputStore(), /* outputSymlinks= */ ImmutableList.of(), action);
    }

    metadataHandler.prepareForActionExecution();

    if (action.discoversInputs()) {
      Duration discoveredInputsDuration = Duration.ZERO;
      if (state.discoveredInputs == null) {
        if (!state.preparedInputDiscovery) {
          action.prepareInputDiscovery();
          state.preparedInputDiscovery = true;
        }
        try (SilentCloseable c = Profiler.instance().profile(ProfilerTask.INFO, "discoverInputs")) {
          state.discoveredInputs =
              skyframeActionExecutor.discoverInputs(
                  action,
                  actionLookupData,
                  metadataHandler,
                  metadataHandler,
                  env,
                  state.actionFileSystem);
        }
        discoveredInputsDuration = Duration.ofNanos(BlazeClock.nanoTime() - actionStartTime);
        if (env.valuesMissing()) {
          checkState(
              state.discoveredInputs == null,
              "Inputs were discovered but more deps were requested by %s",
              action);
          return null;
        }
        checkNotNull(
            state.discoveredInputs,
            "Input discovery returned null but no more deps were requested by %s",
            action);
      }

      addDiscoveredInputs(state, env, action);
      if (env.valuesMissing()) {
        return null;
      }

      // When discover inputs completes, post an event with the duration values.
      env.getListener()
          .post(
              new DiscoveredInputsEvent(
                  SpawnMetrics.Builder.forOtherExec()
                      .setParseTimeInMs((int) discoveredInputsDuration.toMillis())
                      .setTotalTimeInMs((int) discoveredInputsDuration.toMillis())
                      .build(),
                  action,
                  actionStartTime));
    }

    return skyframeActionExecutor.executeAction(
        env,
        action,
        metadataHandler,
        actionStartTime,
        actionLookupData,
        artifactExpander,
        expandedFilesets,
        state.topLevelFilesets,
        state.actionFileSystem,
        skyframeDepsResult,
        new ActionPostprocessingImpl(state),
        state.discoveredInputs != null);
  }

  /** Implementation of {@link ActionPostprocessing}. */
  private final class ActionPostprocessingImpl implements ActionPostprocessing {
    private final InputDiscoveryState state;

    ActionPostprocessingImpl(InputDiscoveryState state) {
      this.state = state;
    }

    @Override
    public void run(
        Environment env,
        Action action,
        ActionMetadataHandler metadataHandler,
        Map<String, String> clientEnv)
        throws InterruptedException, ActionExecutionException {
      // TODO(b/160603797): For the sake of action key computation, we should not need
      //  state.filesetsInsideRunfiles. In fact, for the metadataHandler, we are guaranteed to not
      //  expand any filesets since we request metadata for input/output Artifacts only.
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets =
          state.getExpandedFilesets();
      if (action.discoversInputs()) {
        state.discoveredInputs = action.getInputs();
        addDiscoveredInputs(state, env, action);
        if (env.valuesMissing()) {
          return;
        }
      }
      checkState(!env.valuesMissing(), action);
      skyframeActionExecutor.updateActionCache(
          action,
          metadataHandler,
          metadataHandler,
          new Artifact.ArtifactExpanderImpl(
              // Skipping the filesets in runfiles since those cannot participate in command line
              // creation.
              Collections.unmodifiableMap(state.expandedArtifacts),
              Collections.unmodifiableMap(state.archivedTreeArtifacts),
              expandedFilesets),
          state.token,
          clientEnv);
    }
  }

  private void addDiscoveredInputs(
      InputDiscoveryState state, Environment env, Action actionForError)
      throws InterruptedException, ActionExecutionException {
    // TODO(janakr): This code's assumptions are wrong in the face of Starlark actions with unused
    //  inputs, since ActionExecutionExceptions can come through here and should be aggregated. Fix.

    ActionInputMap inputData = state.inputArtifactData;

    // Filter down to unknown discovered inputs eagerly instead of using a lazy Iterables#filter to
    // reduce iteration cost.
    List<Artifact> unknownDiscoveredInputs = new ArrayList<>();
    for (Artifact input : state.discoveredInputs.toList()) {
      if (inputData.getInputMetadata(input) == null) {
        unknownDiscoveredInputs.add(input);
      }
    }

    if (unknownDiscoveredInputs.isEmpty()) {
      return;
    }

    SkyframeLookupResult nonMandatoryDiscovered =
        env.getValuesAndExceptions(Artifact.keys(unknownDiscoveredInputs));
    for (Artifact input : unknownDiscoveredInputs) {
      SkyValue retrievedMetadata;
      try {
        retrievedMetadata =
            nonMandatoryDiscovered.getOrThrow(Artifact.key(input), SourceArtifactException.class);
      } catch (SourceArtifactException e) {
        if (!input.isSourceArtifact()) {
          throw new IllegalStateException(
              String.format(
                  "Non-source artifact had SourceArtifactException %s %s",
                  input.toDebugString(), actionForError.prettyPrint()),
              e);
        }

        skyframeActionExecutor.printError(e.getMessage(), actionForError);
        // We don't create a specific cause for the artifact as we do in #handleMissingFile because
        // it likely has no label, so we'd have to use the Action's label anyway. Just use the
        // default ActionFailed event constructed by ActionExecutionException.
        String message = "discovered input file does not exist";
        DetailedExitCode code = createDetailedExitCodeForMissingDiscoveredInput(message);
        throw new ActionExecutionException(message, actionForError, false, code);
      }
      if (retrievedMetadata == null) {
        checkState(
            env.valuesMissing(),
            "%s had no metadata but all values were present for %s",
            input,
            actionForError);
        continue;
      }
      if (retrievedMetadata instanceof TreeArtifactValue) {
        TreeArtifactValue treeValue = (TreeArtifactValue) retrievedMetadata;
        state.expandedArtifacts.put(input, treeValue.getChildren());
        inputData.putTreeArtifact((SpecialArtifact) input, treeValue, /*depOwner=*/ null);
        treeValue
            .getArchivedRepresentation()
            .ifPresent(
                archivedRepresentation -> {
                  inputData.putWithNoDepOwner(
                      archivedRepresentation.archivedTreeFileArtifact(),
                      archivedRepresentation.archivedFileValue());
                  state.archivedTreeArtifacts.put(
                      (SpecialArtifact) input, archivedRepresentation.archivedTreeFileArtifact());
                });
      } else if (retrievedMetadata instanceof ActionExecutionValue) {
        inputData.putWithNoDepOwner(
            input, ((ActionExecutionValue) retrievedMetadata).getExistingFileArtifactValue(input));
      } else if (retrievedMetadata instanceof MissingArtifactValue) {
        inputData.putWithNoDepOwner(input, FileArtifactValue.MISSING_FILE_MARKER);
      } else if (retrievedMetadata instanceof FileArtifactValue) {
        inputData.putWithNoDepOwner(input, (FileArtifactValue) retrievedMetadata);
      } else {
        throw new IllegalStateException(
            "unknown metadata for " + input.getExecPathString() + ": " + retrievedMetadata);
      }
    }
  }

  @Nullable
  private static Object establishSkyframeDependencies(Environment env, Action action)
      throws ActionExecutionException, InterruptedException {
    // Before we may safely establish Skyframe dependencies, we must build all action inputs by
    // requesting their ArtifactValues.
    // This is very important to do, because the establishSkyframeDependencies method may request
    // FileValues for input files of this action (directly requesting them, or requesting some other
    // SkyValue whose builder requests FileValues), which may not yet exist if their generating
    // actions have not yet run.
    // See SkyframeAwareActionTest.testRaceConditionBetweenInputAcquisitionAndSkyframeDeps
    checkState(!env.valuesMissing(), action);

    if (action instanceof SkyframeAwareAction) {
      // Skyframe-aware actions should be executed unconditionally, i.e. bypass action cache
      // checking. See documentation of SkyframeAwareAction.
      checkState(action.executeUnconditionally(), action);

      SkyframeAwareAction skyframeAwareAction = (SkyframeAwareAction) action;
      ImmutableList<? extends SkyKey> keys = skyframeAwareAction.getDirectSkyframeDependencies();
      SkyframeLookupResult values = env.getValuesAndExceptions(keys);

      try {
        return skyframeAwareAction.processSkyframeValues(keys, values, env.valuesMissing());
      } catch (SkyframeAwareAction.ExceptionBase e) {
        throw new ActionExecutionException(
            e, action, false, DetailedExitCode.of(e.getFailureDetail()));
      }
    }
    return null;
  }

  private static class CheckInputResults {
    /** Metadata about Artifacts consumed by this Action. */
    private final ActionInputMap actionInputMap;
    /** Artifact expansion mapping for Runfiles tree and tree artifacts. */
    private final Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts;
    /** Archived representations for tree artifacts. */
    private final Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts;
    /** Artifact expansion mapping for Filesets embedded in Runfiles. */
    private final ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>>
        filesetsInsideRunfiles;
    /** Artifact expansion mapping for top level filesets. */
    private final ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets;

    CheckInputResults(
        ActionInputMap actionInputMap,
        Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
        Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets) {
      this.actionInputMap = actionInputMap;
      this.expandedArtifacts = expandedArtifacts;
      this.archivedTreeArtifacts = archivedTreeArtifacts;
      this.filesetsInsideRunfiles = ImmutableMap.copyOf(filesetsInsideRunfiles);
      this.topLevelFilesets = ImmutableMap.copyOf(topLevelFilesets);
    }
  }

  private interface AccumulateInputResultsFactory<S extends ActionInputMapSink, R> {
    R create(
        S actionInputMapSink,
        Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
        Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets);
  }

  /**
   * Declare dependency on all known inputs of action. Throws exception if any are known to be
   * missing. Some inputs may not yet be in the graph, in which case this returns {@code null}.
   */
  @Nullable
  private CheckInputResults checkInputs(
      Environment env,
      Action action,
      SkyframeLookupResult inputDeps,
      NestedSet<Artifact> allInputs,
      Iterable<SkyKey> requestedSkyKeys,
      ActionLookupData actionLookupDataForError)
      throws ActionExecutionException, InterruptedException {
    return accumulateInputs(
        env,
        action,
        inputDeps,
        allInputs,
        requestedSkyKeys,
        sizeHint -> new ActionInputMap(bugReporter, sizeHint),
        CheckInputResults::new,
        /*allowValuesMissingEarlyReturn=*/ true,
        actionLookupDataForError);
  }

  /**
   * Reconstructs the relationships between lost inputs and the direct deps responsible for them.
   */
  private ActionInputDepOwnerMap getInputDepOwners(
      Environment env,
      Action action,
      NestedSet<Artifact> allInputs,
      Collection<ActionInput> lostInputs,
      ActionLookupData actionLookupDataForError)
      throws ActionExecutionException, InterruptedException {
    // The rewinding strategy should be calculated with whatever information is available, instead
    // of returning null if there are missing dependencies, so this uses false for
    // allowValuesMissingEarlyReturn. (Lost inputs coinciding with missing dependencies is possible
    // with, at least, action file systems and include scanning.)
    return accumulateInputs(
        env,
        action,
        /*inputDeps=*/ null,
        allInputs,
        /*requestedSkyKeys=*/ null,
        ignoredInputDepsSize -> new ActionInputDepOwnerMap(lostInputs),
        (actionInputMapSink,
            expandedArtifacts,
            archivedArtifacts,
            filesetsInsideRunfiles,
            topLevelFilesets) -> actionInputMapSink,
        /*allowValuesMissingEarlyReturn=*/ false,
        actionLookupDataForError);
  }

  private static Predicate<Artifact> makeMandatoryInputPredicate(Action action) {
    if (!action.discoversInputs()) {
      return Predicates.alwaysTrue();
    }

    return new Predicate<>() {
      // Lazily flatten the NestedSet in case the predicate is never needed. It's only used in the
      // exceptional case of a missing artifact.
      private ImmutableSet<Artifact> mandatoryInputs = null;
      private ImmutableSet<Artifact> schedulingDependencies = null;

      @Override
      public boolean test(Artifact input) {
        if (!input.isSourceArtifact()) {
          return true;
        }
        if (mandatoryInputs == null) {
          mandatoryInputs = action.getMandatoryInputs().toSet();
        }

        if (mandatoryInputs.contains(input)) {
          return true;
        }

        if (schedulingDependencies == null) {
          schedulingDependencies = action.getSchedulingDependencies().toSet();
        }

        if (schedulingDependencies.contains(input)) {
          return true;
        }

        return false;
      }
    };
  }

  /**
   * May return {@code null} if {@code allowValuesMissingEarlyReturn} and {@code
   * env.valuesMissing()} are true and no inputs result in {@link ActionExecutionException}s.
   *
   * <p>If {@code inputDeps} (and therefore {@code requestedSkyKeys}) is null, assumes that deps
   * have already been checked for exceptions and inserted into the {@link
   * ArtifactNestedSetFunction} cache, so those steps are skipped. (This codepath currently only
   * used for action rewinding.)
   */
  @Nullable
  private <S extends ActionInputMapSink, R> R accumulateInputs(
      Environment env,
      Action action,
      @Nullable SkyframeLookupResult inputDeps,
      NestedSet<Artifact> allInputs,
      @Nullable Iterable<SkyKey> requestedSkyKeys,
      IntFunction<S> actionInputMapSinkFactory,
      AccumulateInputResultsFactory<S, R> accumulateInputResultsFactory,
      boolean allowValuesMissingEarlyReturn,
      ActionLookupData actionLookupDataForError)
      throws ActionExecutionException, InterruptedException {
    Predicate<Artifact> isMandatoryInput = makeMandatoryInputPredicate(action);
    ActionExecutionFunctionExceptionHandler actionExecutionFunctionExceptionHandler = null;
    if (inputDeps != null) {
      actionExecutionFunctionExceptionHandler =
          new ActionExecutionFunctionExceptionHandler(
              Suppliers.memoize(
                  () -> {
                    ImmutableList<Artifact> allInputsList = allInputs.toList();
                    SetMultimap<SkyKey, Artifact> skyKeyToArtifactSet =
                        MultimapBuilder.hashKeys().hashSetValues().build();
                    allInputsList.forEach(
                        input -> {
                          SkyKey key = Artifact.key(input);
                          if (key != input) {
                            skyKeyToArtifactSet.put(key, input);
                          }
                        });
                    return skyKeyToArtifactSet;
                  }),
              inputDeps,
              action,
              isMandatoryInput,
              requestedSkyKeys);
      actionExecutionFunctionExceptionHandler.accumulateAndMaybeThrowExceptions();
    }

    if (env.valuesMissing() && allowValuesMissingEarlyReturn) {
      return null;
    }

    ImmutableList<Artifact> allInputsList = allInputs.toList();

    // When there are no missing values or there was an error, we can start checking individual
    // files. We don't bother to optimize the error-ful case since it's rare.
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles =
        Maps.newHashMapWithExpectedSize(0);
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets =
        Maps.newHashMapWithExpectedSize(0);
    S inputArtifactData = actionInputMapSinkFactory.apply(allInputsList.size());
    Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts =
        Maps.newHashMapWithExpectedSize(128);
    Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts =
        Maps.newHashMapWithExpectedSize(128);

    for (Artifact input : allInputsList) {
      SkyValue value = ArtifactNestedSetFunction.getInstance().getValueForKey(Artifact.key(input));
      if (value == null) {
        if (isMandatoryInput.test(input)) {
          StringBuilder errorMessage = new StringBuilder();
          ImmutableSet<Artifact> outputs = ImmutableSet.copyOf(action.getOutputs());
          NestedSet<Artifact> nestedInputs = action.getInputs();
          ImmutableSet<Artifact> inputs = nestedInputs.toSet();
          if (action.discoversInputs()) {
            errorMessage.append("\nAction discovers inputs");
          } else {
            errorMessage.append("\nAction does not discover inputs");
          }
          if (outputs.contains(input)) {
            errorMessage.append("\nInput is an *output* of action");
          }
          if (inputs.contains(input)) {
            errorMessage.append("\nInput is an input of action, bottom-up path:\n");
            if (!findPathToKey(
                nestedInputs,
                input,
                n -> {
                  ImmutableList<Artifact> artifacts = n.toList();
                  errorMessage
                      .append("  ")
                      .append(artifacts.size())
                      .append(", ")
                      .append(Iterables.limit(artifacts, 10))
                      .append('\n');
                },
                Sets.newHashSet(nestedInputs.toNode()))) {
              errorMessage.append("Could not find input in action's NestedSet inputs");
            }
          } else {
            errorMessage.append("\nInput not present in action's inputs");
          }
          throw new IllegalStateException(
              String.format(
                  "Null value for mandatory %s with no errors or values missing: %s %s %s",
                  input.toDebugString(),
                  actionLookupDataForError,
                  action.prettyPrint(),
                  errorMessage));
        }
        continue;
      }
      if (value instanceof MissingArtifactValue) {
        if (isMandatoryInput.test(input)) {
          checkNotNull(
                  actionExecutionFunctionExceptionHandler,
                  "Missing artifact should have been caught already %s %s %s",
                  input,
                  value,
                  action)
              .accumulateMissingFileArtifactValue(input, (MissingArtifactValue) value);
          continue;
        } else {
          value = FileArtifactValue.MISSING_FILE_MARKER;
        }
      }
      ActionInputMapHelper.addToMap(
          inputArtifactData,
          expandedArtifacts,
          archivedTreeArtifacts,
          filesetsInsideRunfiles,
          topLevelFilesets,
          input,
          value,
          env,
          skyframeActionExecutor.requiresTreeMetadataWhenTreeFileIsInput());
    }

    if (actionExecutionFunctionExceptionHandler != null) {
      // After accumulating the inputs, we might find some mandatory artifact with
      // SourceFileInErrorArtifactValue.
      actionExecutionFunctionExceptionHandler.maybeThrowException();
    }

    return accumulateInputResultsFactory.create(
        inputArtifactData,
        expandedArtifacts,
        archivedTreeArtifacts,
        filesetsInsideRunfiles,
        topLevelFilesets);
  }

  private static <T> boolean findPathToKey(
      NestedSet<T> start, T target, Consumer<NestedSet<T>> receiver, Set<NestedSet.Node> seen) {
    if (start.getLeaves().contains(target)) {
      receiver.accept(start);
      return true;
    }
    for (NestedSet<T> next : start.getNonLeaves()) {
      if (seen.add(next.toNode()) && findPathToKey(next, target, receiver, seen)) {
        receiver.accept(start);
        return true;
      }
    }
    return false;
  }

  static LabelCause createLabelCause(
      Artifact input,
      DetailedExitCode detailedExitCode,
      Label labelInCaseOfBug,
      BugReporter bugReporter) {
    if (input.getOwner() == null) {
      bugReporter.sendBugReport(
          new IllegalStateException(
              String.format(
                  "Mandatory artifact %s with exit code %s should have owner (%s)",
                  input, detailedExitCode, labelInCaseOfBug)));
    }
    return createLabelCauseNullOwnerOk(input, detailedExitCode, labelInCaseOfBug, bugReporter);
  }

  private static LabelCause createLabelCauseNullOwnerOk(
      Artifact input,
      DetailedExitCode detailedExitCode,
      Label actionLabel,
      BugReporter bugReporter) {
    if (!input.isSourceArtifact()) {
      bugReporter.logUnexpected(
          "Unexpected exit code %s for generated artifact %s (%s)",
          detailedExitCode, input, actionLabel);
    }
    return new LabelCause(
        MoreObjects.firstNonNull(input.getOwner(), actionLabel), detailedExitCode);
  }

  /**
   * Clears bookkeeping used by action rewinding.
   *
   * <p>Should be called once execution is over, otherwise there may be a memory leak of the action
   * rewinding bookkeeping information.
   */
  public void complete(ExtendedEventHandler eventHandler) {
    actionRewindStrategy.reset(eventHandler);
  }

  /**
   * State to save work across restarts of ActionExecutionFunction due to missing values in the
   * graph for actions that discover inputs. There are three places where we save work, all for
   * actions that discover inputs:
   *
   * <ol>
   *   <li>If not all known input metadata (coming from Action#getInputs) is available yet, then the
   *       calculated set of inputs (including the inputs resolved from the action cache) is saved.
   *   <li>If not all discovered inputs' metadata is available yet, then the known input metadata
   *       together with the set of discovered inputs is saved, as well as the Token used to
   *       identify this action to the action cache.
   *   <li>If, after execution, new inputs are discovered whose metadata is not yet available, then
   *       the same data as in the previous case is saved, along with the actual result of
   *       execution.
   * </ol>
   */
  static class InputDiscoveryState implements SkyKeyComputeState {
    AllInputs allInputs;
    /** Mutable map containing metadata for known artifacts. */
    ActionInputMap inputArtifactData = null;

    Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts = null;
    Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts = null;
    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsInsideRunfiles = null;
    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets = null;
    Token token = null;
    NestedSet<Artifact> discoveredInputs = null;
    FileSystem actionFileSystem = null;
    boolean preparedInputDiscovery = false;
    boolean actionInputCollectedEventSent = false;

    /**
     * Stores the ArtifactNestedSetKeys created from the inputs of this actions. Objective: avoid
     * creating a new ArtifactNestedSetKey for the same NestedSet each time we run
     * ActionExecutionFunction for the same action. This is wiped everytime allInputs is updated.
     */
    CompactHashSet<SkyKey> requestedArtifactNestedSetKeys = null;

    boolean hasCollectedInputs() {
      return allInputs != null;
    }

    boolean hasArtifactData() {
      boolean result = inputArtifactData != null;
      checkState(result == (expandedArtifacts != null), this);
      checkState(result == (archivedTreeArtifacts != null), this);
      return result;
    }

    boolean hasCheckedActionCache() {
      // If token is null because there was an action cache hit, this method is never called again
      // because we return immediately.
      return token != null;
    }

    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> getExpandedFilesets() {
      if (topLevelFilesets == null || topLevelFilesets.isEmpty()) {
        return filesetsInsideRunfiles;
      }

      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetsMap =
          Maps.newHashMapWithExpectedSize(filesetsInsideRunfiles.size() + topLevelFilesets.size());
      filesetsMap.putAll(filesetsInsideRunfiles);
      filesetsMap.putAll(topLevelFilesets);
      return ImmutableMap.copyOf(filesetsMap);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("token", token)
          .add("allInputs", allInputs)
          .add("inputArtifactData", inputArtifactData)
          .add("discoveredInputs", discoveredInputs)
          .toString();
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * ActionExecutionFunction#compute}.
   */
  static final class ActionExecutionFunctionException extends SkyFunctionException {

    private final ActionExecutionException actionException;

    ActionExecutionFunctionException(ActionExecutionException e) {
      // We conservatively assume that the error is transient. We don't have enough information to
      // distinguish non-transient errors (e.g. compilation error from a deterministic compiler)
      // from transient ones (e.g. IO error).
      // TODO(bazel-team): Have ActionExecutionExceptions declare their transience.
      super(e, Transience.TRANSIENT);
      this.actionException = e;
    }

    @Override
    public boolean isCatastrophic() {
      return actionException.isCatastrophe();
    }
  }

  /** Helper subclass for the error-handling logic for ActionExecutionFunction#accumulateInputs. */
  private final class ActionExecutionFunctionExceptionHandler {
    private final Supplier<SetMultimap<SkyKey, Artifact>> skyKeyToDerivedArtifactSetForExceptions;
    private final SkyframeLookupResult inputDeps;
    private final Action action;
    private final Predicate<Artifact> isMandatoryInput;
    private final Iterable<SkyKey> requestedSkyKeys;
    private final List<LabelCause> missingArtifactCauses = Lists.newArrayListWithCapacity(0);
    private final List<NestedSet<Cause>> transitiveCauses = Lists.newArrayListWithCapacity(0);
    private ActionExecutionException firstActionExecutionException;

    ActionExecutionFunctionExceptionHandler(
        Supplier<SetMultimap<SkyKey, Artifact>> skyKeyToDerivedArtifactSetForExceptions,
        SkyframeLookupResult inputDeps,
        Action action,
        Predicate<Artifact> isMandatoryInput,
        Iterable<SkyKey> requestedSkyKeys) {
      this.skyKeyToDerivedArtifactSetForExceptions = skyKeyToDerivedArtifactSetForExceptions;
      this.inputDeps = inputDeps;
      this.action = action;
      this.isMandatoryInput = isMandatoryInput;
      this.requestedSkyKeys = requestedSkyKeys;
    }

    /**
     * Goes through the list of evaluated SkyKeys and handles any exception that arises, taking into
     * account whether the corresponding artifact(s) is a mandatory input.
     *
     * <p>Also updates ArtifactNestedSetFunction#skyKeyToSkyValue if an Artifact's value is
     * non-null.
     *
     * @throws ActionExecutionException if the eval of any mandatory artifact threw an exception.
     *     This may not be the most complete exception because it may lack missing input file causes
     *     that did not throw exceptions, but were present as "missing file artifact values" in the
     *     global {@link ArtifactNestedSetFunction#artifactSkyKeyToSkyValue} map. Unfortunately,
     *     that map is not trustworthy in the exceptional case, since it may not have been populated
     *     with all data from this build before an exception shut the build down.
     */
    void accumulateAndMaybeThrowExceptions() throws ActionExecutionException {
      for (SkyKey key : requestedSkyKeys) {
        try {
          SkyValue value =
              inputDeps.getOrThrow(
                  key,
                  SourceArtifactException.class,
                  ActionExecutionException.class,
                  ArtifactNestedSetEvalException.class);
          if (key instanceof ArtifactNestedSetKey || value == null) {
            continue;
          }
          ArtifactNestedSetFunction.getInstance().updateValueForKey(key, value);
        } catch (SourceArtifactException e) {
          handleSourceArtifactExceptionFromSkykey(key, e);
        } catch (ActionExecutionException e) {
          handleActionExecutionExceptionFromSkykey(key, e);
        } catch (ArtifactNestedSetEvalException e) {
          for (Pair<SkyKey, Exception> skyKeyAndException : e.getNestedExceptions().toList()) {
            SkyKey skyKey = skyKeyAndException.getFirst();
            Exception inputException = skyKeyAndException.getSecond();
            checkState(
                inputException instanceof SourceArtifactException
                    || inputException instanceof ActionExecutionException,
                "Unexpected exception type: %s, key: %s",
                inputException,
                skyKey);
            if (inputException instanceof SourceArtifactException) {
              handleSourceArtifactExceptionFromSkykey(
                  skyKey, (SourceArtifactException) inputException);
              continue;
            }
            handleActionExecutionExceptionFromSkykey(
                skyKey, (ActionExecutionException) inputException);
          }
        }
      }
      maybeThrowException();
    }

    private void handleActionExecutionExceptionFromSkykey(SkyKey key, ActionExecutionException e) {
      if (key instanceof Artifact) {
        handleActionExecutionExceptionPerArtifact((Artifact) key, e);
        return;
      }
      Set<Artifact> associatedInputs = skyKeyToDerivedArtifactSetForExceptions.get().get(key);
      if (associatedInputs.isEmpty()) {
        // This can happen if an action prunes its inputs, e.g. the way StarlarkAction implements
        // unused_inputs_list. An input may no longer be present in getInputs(), but its generating
        // action could still be a Skyframe dependency because Skyframe eagerly adds a dep group to
        // a dirty node if all prior dep groups are clean. If the pruned input is in error, it
        // propagates during error bubbling, and we reach this point.
        // TODO(lberki): Can inputs be immutable instead?
        logger.atWarning().log(
            "While handling errors for %s, encountered error from %s which is not associated with"
                + " any inputs",
            action.prettyPrint(), key);
        if (firstActionExecutionException == null) {
          firstActionExecutionException = e;
        }
      } else {
        for (Artifact input : associatedInputs) {
          handleActionExecutionExceptionPerArtifact(input, e);
        }
      }
    }

    private void handleSourceArtifactExceptionFromSkykey(SkyKey key, SourceArtifactException e) {
      if (!(key instanceof Artifact) || !((Artifact) key).isSourceArtifact()) {
        bugReporter.logUnexpected(
            e, "Unexpected SourceArtifactException for key: %s, %s", key, action.prettyPrint());
        missingArtifactCauses.add(
            new LabelCause(action.getOwner().getLabel(), e.getDetailedExitCode()));
        return;
      }

      if (isMandatoryInput.test((Artifact) key)) {
        missingArtifactCauses.add(
            createLabelCauseNullOwnerOk(
                (Artifact) key,
                e.getDetailedExitCode(),
                action.getOwner().getLabel(),
                bugReporter));
      }
    }

    void accumulateMissingFileArtifactValue(Artifact input, MissingArtifactValue value) {
      missingArtifactCauses.add(
          createLabelCause(
              input, value.getDetailedExitCode(), action.getOwner().getLabel(), bugReporter));
    }

    /** @throws ActionExecutionException if there is any accumulated exception from the inputs. */
    void maybeThrowException() throws ActionExecutionException {
      for (LabelCause missingInput : missingArtifactCauses) {
        skyframeActionExecutor.printError(missingInput.getMessage(), action);
      }
      // We need to rethrow the first exception because it can contain a useful error message.
      if (firstActionExecutionException != null) {
        if (missingArtifactCauses.isEmpty()
            && (checkNotNull(transitiveCauses, action).size() == 1)) {
          // In the case a single action failed, just propagate the exception upward. This avoids
          // having to copy the root causes to the upwards transitive closure.
          throw firstActionExecutionException;
        }
        NestedSetBuilder<Cause> allCauses =
            NestedSetBuilder.<Cause>stableOrder().addAll(missingArtifactCauses);
        transitiveCauses.forEach(allCauses::addTransitive);
        throw new ActionExecutionException(
            firstActionExecutionException.getMessage(),
            firstActionExecutionException.getCause(),
            action,
            allCauses.build(),
            firstActionExecutionException.isCatastrophe(),
            firstActionExecutionException.getDetailedExitCode());
      }

      if (!missingArtifactCauses.isEmpty()) {
        throw throwSourceErrorException(action, missingArtifactCauses);
      }
    }

    private void handleActionExecutionExceptionPerArtifact(
        Artifact input, ActionExecutionException e) {
      if (isMandatoryInput.test(input)) {
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || (!firstActionExecutionException.isCatastrophe() && e.isCatastrophe())) {
          firstActionExecutionException = e;
        }
        transitiveCauses.add(e.getRootCauses());
      }
    }
  }

  /**
   * Called when there are no action execution errors (whose reporting hides missing sources), but
   * there was at least one missing/io exception-triggering source artifact. Returns a {@link
   * DetailedExitCode} constructed from {@code sourceArtifactErrorCauses} specific to a single such
   * artifact and an error message suitable as the message to a thrown exception that summarizes the
   * findings.
   */
  static Pair<DetailedExitCode, String> createSourceErrorCodeAndMessage(
      List<? extends Cause> sourceArtifactErrorCauses, Object debugInfo) {
    AtomicBoolean sawSourceArtifactException = new AtomicBoolean();
    AtomicBoolean sawMissingFile = new AtomicBoolean();
    DetailedExitCode prioritizedDetailedExitCode =
        sourceArtifactErrorCauses.stream()
            .map(Cause::getDetailedExitCode)
            .peek(
                code -> {
                  if (code.getFailureDetail() == null) {
                    BugReport.sendBugReport(
                        new NullPointerException(
                            "Code " + code + " had no failure detail for " + debugInfo));
                    return;
                  }
                  switch (code.getFailureDetail().getExecution().getCode()) {
                    case SOURCE_INPUT_IO_EXCEPTION:
                      sawSourceArtifactException.set(true);
                      break;
                    case SOURCE_INPUT_MISSING:
                      sawMissingFile.set(true);
                      break;
                    default:
                      BugReport.sendBugReport(
                          new IllegalStateException(
                              "Unexpected error code in " + code + " for " + debugInfo));
                  }
                })
            .max(DetailedExitCodeComparator.INSTANCE)
            .get();
    String errorMessage =
        sourceArtifactErrorCauses.size()
            + " input file(s) "
            + Joiner.on(" or ")
                .skipNulls()
                .join(
                    sawSourceArtifactException.get() ? "are in error" : null,
                    sawMissingFile.get() ? "do not exist" : null);
    return Pair.of(prioritizedDetailedExitCode, errorMessage);
  }

  private ActionExecutionException throwSourceErrorException(
      Action action, List<? extends Cause> sourceArtifactErrorCauses)
      throws ActionExecutionException {
    Pair<DetailedExitCode, String> codeAndMessage =
        createSourceErrorCodeAndMessage(sourceArtifactErrorCauses, action);
    ActionExecutionException ex =
        new ActionExecutionException(
            codeAndMessage.getSecond(),
            action,
            NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceArtifactErrorCauses),
            /*catastrophe=*/ false,
            codeAndMessage.getFirst());
    skyframeActionExecutor.printError(ex.getMessage(), action);
    // Don't actually return: throw exception directly so caller can't get it wrong.
    throw ex;
  }

  private static DetailedExitCode createDetailedExitCodeForMissingDiscoveredInput(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(Code.DISCOVERED_INPUT_DOES_NOT_EXIST))
            .build());
  }
}
