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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputDepOwnerMap;
import com.google.devtools.build.lib.actions.ActionInputDepOwners;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputMapSink;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactSkyKey;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.LostInputsExecException.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.MissingDepException;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.cpp.IncludeScannable;
import com.google.devtools.build.lib.skyframe.ActionRewindStrategy.RewindPlan;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntFunction;
import javax.annotation.Nullable;

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
 */
public class ActionExecutionFunction implements SkyFunction, CompletionReceiver {
  private final ActionRewindStrategy actionRewindStrategy = new ActionRewindStrategy();
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final BlazeDirectories directories;
  private final AtomicReference<TimestampGranularityMonitor> tsgm;
  private ConcurrentMap<Action, ContinuationState> stateMap;

  public ActionExecutionFunction(
      SkyframeActionExecutor skyframeActionExecutor,
      BlazeDirectories directories,
      AtomicReference<TimestampGranularityMonitor> tsgm) {
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.directories = directories;
    this.tsgm = tsgm;
    stateMap = Maps.newConcurrentMap();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ActionExecutionFunctionException, InterruptedException {
    ActionLookupData actionLookupData = (ActionLookupData) skyKey.argument();
    Action action = getActionForLookupData(env, actionLookupData);
    skyframeActionExecutor.noteActionEvaluationStarted(actionLookupData, action);
    if (actionDependsOnBuildId(action)) {
      PrecomputedValue.BUILD_ID.get(env);
    }

    PrecomputedValue.FETCH_REMOTE_OUTPUTS.get(env);

    // Look up the parts of the environment that influence the action.
    Map<SkyKey, SkyValue> clientEnvLookup =
        env.getValues(
            Iterables.transform(
                action.getClientEnvironmentVariables(), ClientEnvironmentFunction::key));
    if (env.valuesMissing()) {
      return null;
    }
    Map<String, String> clientEnv = new HashMap<>();
    for (Map.Entry<SkyKey, SkyValue> entry : clientEnvLookup.entrySet()) {
      ClientEnvironmentValue envValue = (ClientEnvironmentValue) entry.getValue();
      if (envValue.getValue() != null) {
        clientEnv.put((String) entry.getKey().argument(), envValue.getValue());
      }
    }

    // For restarts of this ActionExecutionFunction we use a ContinuationState variable, below, to
    // avoid redoing work.
    //
    // However, if two actions are shared and the first one executes, when the
    // second one goes to execute, we should detect that and short-circuit, even without taking
    // ContinuationState into account.
    //
    // Additionally, if an action restarted (in the Skyframe sense) after it executed because it
    // discovered new inputs during execution, we should detect that and short-circuit.
    boolean actionAlreadyRan = skyframeActionExecutor.probeActionExecution(action);

    // If this action was previously completed this build, then this evaluation must be happening
    // because of rewinding. Prevent any ProgressLike events from being published a second time for
    // this action; downstream consumers of action events reasonably don't expect them.
    env = getProgressEventSuppressingEnvironmentIfPreviouslyCompleted(action, env);

    if (action.discoversInputs()) {
      // If this action previously failed due to a lost input found during input discovery, ensure
      // that the input is regenerated before attempting discovery again.
      if (declareDepsOnLostDiscoveredInputsIfAny(env, action)) {
        return null;
      }
    }

    ContinuationState state;
    if (action.discoversInputs()) {
      state = getState(action);
    } else {
      // Because this is a new state, all conditionals below about whether state has already done
      // something will return false, and so we will execute all necessary steps.
      state = new ContinuationState();
    }
    if (!state.hasCollectedInputs()) {
      state.allInputs = collectInputs(action, env);
      if (state.allInputs == null) {
        // Missing deps.
        return null;
      }
    } else if (state.allInputs.keysRequested != null) {
      // Preserve the invariant that we ask for the same deps each build.
      env.getValues(state.allInputs.keysRequested);
      Preconditions.checkState(!env.valuesMissing(), "%s %s", action, state);
    }
    CheckInputResults checkedInputs = null;
    Iterable<SkyKey> inputDepKeys =
        toKeys(
            state.allInputs.getAllInputs(),
            action.discoversInputs() ? action.getMandatoryInputs() : null);
    // Declare deps on known inputs to action. We do this unconditionally to maintain our
    // invariant of asking for the same deps each build.
    Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps =
        env.getValuesOrThrow(
            inputDepKeys, MissingInputFileException.class, ActionExecutionException.class);
    try {
      if (!actionAlreadyRan && !state.hasArtifactData()) {
        // Do we actually need to find our metadata?
        checkedInputs = checkInputs(env, action, inputDeps);
      }
    } catch (ActionExecutionException e) {
      // Remove action from state map in case it's there (won't be unless it discovers inputs).
      stateMap.remove(action);
      throw new ActionExecutionFunctionException(e);
    }

    if (env.valuesMissing()) {
      // There was missing artifact metadata in the graph. Wait for it to be present.
      // We must check this and return here before attempting to establish any Skyframe dependencies
      // of the action; see establishSkyframeDependencies why.
      return null;
    }

    Object skyframeDepsResult;
    try {
      skyframeDepsResult = establishSkyframeDependencies(env, action);
    } catch (ActionExecutionException e) {
      // Remove action from state map in case it's there (won't be unless it discovers inputs).
      stateMap.remove(action);
      throw new ActionExecutionFunctionException(e);
    }
    if (env.valuesMissing()) {
      return null;
    }

    if (checkedInputs != null) {
      Preconditions.checkState(!state.hasArtifactData(), "%s %s", state, action);
      state.inputArtifactData = checkedInputs.actionInputMap;
      state.expandedArtifacts = checkedInputs.expandedArtifacts;
      state.expandedFilesets = checkedInputs.expandedFilesets;
      if (skyframeActionExecutor.usesActionFileSystem()) {
        state.actionFileSystem =
            skyframeActionExecutor.createActionFileSystem(
                directories.getRelativeOutputPath(),
                checkedInputs.actionInputMap,
                action.getOutputs());
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
              actionAlreadyRan,
              skyframeDepsResult,
              actionStartTime);
    } catch (LostInputsActionExecutionException e) {
      // Remove action from state map in case it's there (won't be unless it discovers inputs).
      stateMap.remove(action);

      // Reconstruct the relationship between lost inputs and this action's direct deps if any of
      // the lost inputs came from runfiles:
      ActionInputDepOwners runfilesDepOwners;
      Set<ActionInput> lostRunfiles = e.getInputOwners().getRunfilesInputsAndOwners();
      if (!lostRunfiles.isEmpty()) {
        try {
          runfilesDepOwners = getInputDepOwners(env, action, inputDeps, lostRunfiles);
        } catch (ActionExecutionException unexpected) {
          // getInputDepOwners should not be able to throw, because it does the same work as
          // checkInputs, so if getInputDepOwners throws then checkInputs should have thrown, and if
          // checkInputs threw then we shouldn't have reached this point in action execution.
          throw new IllegalStateException(unexpected);
        }
      } else {
        runfilesDepOwners = ActionInputDepOwners.EMPTY_INSTANCE;
      }

      ImmutableList<Artifact> lostDiscoveredInputs = ImmutableList.of();
      Iterable<? extends SkyKey> failedActionDeps;
      if (e.isFromInputDiscovery()) {
        // Lost inputs found during input discovery are necessarily artifacts, and already Skyframe
        // deps of this action.
        lostDiscoveredInputs =
            e.getLostInputs().stream()
                .map(i -> (Artifact) i)
                .collect(ImmutableList.toImmutableList());
        failedActionDeps = lostDiscoveredInputs;
      } else if (state.discoveredInputs != null) {
        failedActionDeps = Iterables.concat(inputDepKeys, state.discoveredInputs);
      } else {
        failedActionDeps = inputDepKeys;
      }

      RewindPlan rewindPlan;
      try {
        rewindPlan =
            actionRewindStrategy.getRewindPlan(action, failedActionDeps, e, runfilesDepOwners, env);
      } catch (ActionExecutionException rewindingFailedException) {
        stateMap.remove(action);
        if (e.isActionStartedEventAlreadyEmitted()) {
          // SkyframeActionExecutor's ActionRunner didn't emit an ActionCompletionEvent because it
          // hoped rewinding would fix things. Now we know that rewinding won't work.
          env.getListener()
              .post(new ActionCompletionEvent(actionStartTime, action, actionLookupData));
        }
        // This call to processAndGetExceptionToThrow will emit an ActionExecutedEvent and report
        // the error. The previous call to processAndGetExceptionToThrow didn't.
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
      skyframeActionExecutor.resetFailedActionExecution(action, lostDiscoveredInputs);
      for (Action actionToRestart : rewindPlan.getAdditionalActionsToRestart()) {
        skyframeActionExecutor.resetPreviouslyCompletedActionExecution(actionToRestart);
      }
      return rewindPlan.getNodesToRestart();
    } catch (ActionExecutionException e) {
      // Remove action from state map in case it's there (won't be unless it discovers inputs).
      stateMap.remove(action);
      // In this case we do not report the error to the action reporter because we have already
      // done it in SkyframeActionExecutor.reportErrorIfNotAbortingMode() method. That method
      // prints the error in the top-level reporter and also dumps the recorded StdErr for the
      // action. Label can be null in the case of, e.g., the SystemActionOwner (for build-info.txt).
      throw new ActionExecutionFunctionException(new AlreadyReportedActionExecutionException(e));
    }

    if (env.valuesMissing()) {
      // Only input-discovering actions are present in the stateMap. Other actions may have
      // valuesMissing() here in rare circumstances related to Fileset inputs being unavailable.
      // See comments in ActionInputMapHelper#getFilesets().
      Preconditions.checkState(!action.discoversInputs() || stateMap.containsKey(action), action);
      return null;
    }

    // Remove action from state map in case it's there (won't be unless it discovers inputs).
    stateMap.remove(action);
    return result;
  }

  private Environment getProgressEventSuppressingEnvironmentIfPreviouslyCompleted(
      Action action, Environment env) {
    if (skyframeActionExecutor.probeCompletedAndReset(action)) {
      return new ProgressEventSuppressingEnvironment(env);
    }
    return env;
  }

  private boolean declareDepsOnLostDiscoveredInputsIfAny(Environment env, Action action)
      throws InterruptedException, ActionExecutionFunctionException {
    ImmutableList<Artifact> previouslyLostDiscoveredInputs =
        skyframeActionExecutor.getLostDiscoveredInputs(action);
    if (previouslyLostDiscoveredInputs != null) {
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>>
          lostInputValues =
              env.getValuesOrThrow(
                  previouslyLostDiscoveredInputs,
                  MissingInputFileException.class,
                  ActionExecutionException.class);
      if (env.valuesMissing()) {
        return true;
      }
      for (Map.Entry<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>>
          lostInput : lostInputValues.entrySet()) {
        try {
          lostInput.getValue().get();
        } catch (MissingInputFileException e) {
          // MissingInputFileException comes from problems with source artifact construction.
          // Rewinding never invalidates source artifacts.
          throw new IllegalStateException(
              "MissingInputFileException unexpected from rewound generated discovered input. key="
                  + lostInput.getKey(),
              e);
        } catch (ActionExecutionException e) {
          throw new ActionExecutionFunctionException(e);
        }
      }
    }
    return false;
  }

  static Action getActionForLookupData(Environment env, ActionLookupData actionLookupData)
      throws InterruptedException {
    // Because of the phase boundary separating analysis and execution, all needed
    // ActionLookupValues must have already been evaluated.
    ActionLookupValue actionLookupValue =
        Preconditions.checkNotNull(
            (ActionLookupValue) env.getValue(actionLookupData.getActionLookupKey()),
            "ActionLookupValue missing: %s",
            actionLookupData);
    return actionLookupValue.getAction(actionLookupData.getActionIndex());
  }

  /**
   * An action's inputs needed for execution. May not just be the result of Action#getInputs(). If
   * the action cache's view of this action contains additional inputs, it will request metadata for
   * them, so we consider those inputs as dependencies of this action as well. Returns null if some
   * dependencies were missing and this ActionExecutionFunction needs to restart.
   */
  @Nullable
  private AllInputs collectInputs(Action action, Environment env) throws InterruptedException {
    Iterable<Artifact> allKnownInputs = action.getInputs();
    if (action.inputsDiscovered()) {
      return new AllInputs(allKnownInputs);
    }

    Preconditions.checkState(action.discoversInputs(), action);
    PackageRootResolverWithEnvironment resolver = new PackageRootResolverWithEnvironment(env);
    Iterable<Artifact> actionCacheInputs =
        skyframeActionExecutor.getActionCachedInputs(action, resolver);
    if (actionCacheInputs == null) {
      Preconditions.checkState(env.valuesMissing(), action);
      return null;
    }
    return new AllInputs(allKnownInputs, actionCacheInputs, resolver.keysRequested);
  }

  private static class AllInputs {
    final Iterable<Artifact> defaultInputs;
    @Nullable final Iterable<Artifact> actionCacheInputs;
    @Nullable final List<SkyKey> keysRequested;

    AllInputs(Iterable<Artifact> defaultInputs) {
      this.defaultInputs = Preconditions.checkNotNull(defaultInputs);
      this.actionCacheInputs = null;
      this.keysRequested = null;
    }

    AllInputs(
        Iterable<Artifact> defaultInputs,
        Iterable<Artifact> actionCacheInputs,
        List<SkyKey> keysRequested) {
      this.defaultInputs = Preconditions.checkNotNull(defaultInputs);
      this.actionCacheInputs = Preconditions.checkNotNull(actionCacheInputs);
      this.keysRequested = keysRequested;
    }

    Iterable<Artifact> getAllInputs() {
      return actionCacheInputs == null
          ? defaultInputs
          : Iterables.concat(defaultInputs, actionCacheInputs);
    }
  }

  /**
   * Skyframe implementation of {@link PackageRootResolver}. Should be used only from SkyFunctions,
   * because it uses SkyFunction.Environment for evaluation of ContainingPackageLookupValue.
   */
  private static class PackageRootResolverWithEnvironment implements PackageRootResolver {
    final List<SkyKey> keysRequested = new ArrayList<>();
    private final Environment env;

    private PackageRootResolverWithEnvironment(Environment env) {
      this.env = env;
    }

    @Override
    public Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
        throws InterruptedException {
      Preconditions.checkState(
          keysRequested.isEmpty(),
          "resolver should only be called once: %s %s",
          keysRequested,
          execPaths);
      // Create SkyKeys list based on execPaths.
      Map<PathFragment, SkyKey> depKeys = new HashMap<>();
      for (PathFragment path : execPaths) {
        PathFragment parent =
            Preconditions.checkNotNull(
                path.getParentDirectory(), "Must pass in files, not root directory");
        Preconditions.checkArgument(!parent.isAbsolute(), path);
        try {
          SkyKey depKey =
              ContainingPackageLookupValue.key(PackageIdentifier.discoverFromExecPath(path, true));
          depKeys.put(path, depKey);
          keysRequested.add(depKey);
        } catch (LabelSyntaxException e) {
          // This code is only used to do action cache checks. If one of the file names we got from
          // the action cache is corrupted, or if the action cache is from a different Bazel
          // binary, then the path may not be valid for this Bazel binary, and trigger this
          // exception. In that case, it's acceptable for us to ignore the exception - we'll get an
          // action cache miss and re-execute the action, which is what we should do.
          continue;
        }
      }

      Map<SkyKey, SkyValue> values = env.getValues(depKeys.values());
      if (env.valuesMissing()) {
        return null;
      }

      Map<PathFragment, Root> result = new HashMap<>();
      for (PathFragment path : execPaths) {
        if (!depKeys.containsKey(path)) {
          continue;
        }
        ContainingPackageLookupValue value =
            (ContainingPackageLookupValue) values.get(depKeys.get(path));
        if (value.hasContainingPackage()) {
          // We have found corresponding root for current execPath.
          result.put(
              path,
              SkyframeExecutor.maybeTransformRootForRepository(
                  value.getContainingPackageRoot(),
                  value.getContainingPackageName().getRepository()));
        } else {
          // We haven't found corresponding root for current execPath.
          result.put(path, null);
        }
      }
      return result;
    }
  }

  private ActionExecutionValue checkCacheAndExecuteIfNeeded(
      Action action,
      ContinuationState state,
      Environment env,
      Map<String, String> clientEnv,
      ActionLookupData actionLookupData,
      boolean actionAlreadyRan,
      Object skyframeDepsResult,
      long actionStartTime)
      throws ActionExecutionException, InterruptedException {
    // If this is a shared action and the other action is the one that executed, we must use that
    // other action's value, provided here, since it is populated with metadata for the outputs.
    if (actionAlreadyRan) {
      return skyframeActionExecutor.executeAction(
          env.getListener(),
          action,
          /* metadataHandler= */ null,
          /* actionStartTime= */ -1,
          /* actionExecutionContext= */ null,
          actionLookupData);
    }
    // The metadataHandler may be recreated if we discover inputs.
    ArtifactPathResolver pathResolver = ArtifactPathResolver.createPathResolver(
        state.actionFileSystem, skyframeActionExecutor.getExecRoot());
    ActionMetadataHandler metadataHandler =
        new ActionMetadataHandler(
            state.inputArtifactData,
            /* missingArtifactsAllowed= */ action.discoversInputs(),
            action.getOutputs(),
            tsgm.get(),
            pathResolver,
            state.actionFileSystem == null ? new OutputStore() : new MinimalOutputStore());
    // We only need to check the action cache if we haven't done it on a previous run.
    if (!state.hasCheckedActionCache()) {
      state.token =
          skyframeActionExecutor.checkActionCache(
              env.getListener(),
              action,
              metadataHandler,
              actionStartTime,
              state.allInputs.actionCacheInputs,
              clientEnv);
    }

    if (state.token == null) {
      // We got a hit from the action cache -- no need to execute.
      Preconditions.checkState(
          !(action instanceof SkyframeAwareAction),
          "Error, we're not re-executing a "
              + "SkyframeAwareAction which should be re-executed unconditionally. Action: %s",
          action);
      return ActionExecutionValue.createFromOutputStore(
          metadataHandler.getOutputStore(),
          /*outputSymlinks=*/ null,
          (action instanceof IncludeScannable)
              ? ((IncludeScannable) action).getDiscoveredModules()
              : null,
          actionDependsOnBuildId(action));
    }

    // Delete the metadataHandler's cache of the action's outputs, since they are being deleted.
    metadataHandler.discardOutputMetadata();

    // This may be recreated if we discover inputs.
    // TODO(shahan): this isn't used when using ActionFileSystem so we can avoid creating some
    // unused objects.
    if (action.discoversInputs()) {
      if (state.discoveredInputs == null) {
        try {
          try {
            state.updateFileSystemContext(
                skyframeActionExecutor, env, metadataHandler, ImmutableMap.of());
          } catch (IOException e) {
            throw new ActionExecutionException(
                "Failed to update filesystem context: ", e, action, /*catastrophe=*/ false);
          }
          state.discoveredInputs =
              skyframeActionExecutor.discoverInputs(
                  action, metadataHandler, metadataHandler, env, state.actionFileSystem);
          Preconditions.checkState(
              env.valuesMissing() == (state.discoveredInputs == null),
              "discoverInputs() must return null iff requesting more dependencies.");
          if (state.discoveredInputs == null) {
            return null;
          }
        } catch (MissingDepException e) {
          Preconditions.checkState(env.valuesMissing(), action);
          return null;
        }
      }
      addDiscoveredInputs(
          state.inputArtifactData, state.expandedArtifacts, state.discoveredInputs, env);
      if (env.valuesMissing()) {
        return null;
      }
      metadataHandler =
          new ActionMetadataHandler(
              state.inputArtifactData,
              /* missingArtifactsAllowed= */ false,
              action.getOutputs(),
              tsgm.get(),
              pathResolver,
              state.actionFileSystem == null ? new OutputStore() : new MinimalOutputStore());
      // Set the MetadataHandler to accept output information.
      metadataHandler.discardOutputMetadata();
    }

    // Make sure this is a regular HashMap rather than ImmutableMapBuilder so that we are safe
    // in case of collisions.
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings = new HashMap<>();
    for (Artifact actionInput : action.getInputs()) {
      if (!actionInput.isFileset()) {
        continue;
      }

      ImmutableList<FilesetOutputSymlink> mapping =
          ActionInputMapHelper.getFilesets(env, actionInput);
      if (mapping == null) {
        return null;
      }
      filesetMappings.put(actionInput, mapping);
    }

    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> topLevelFilesets =
        ImmutableMap.copyOf(filesetMappings);

    // Aggregate top-level Filesets with Filesets nested in Runfiles. Both should be used to update
    // the FileSystem context.
    state.expandedFilesets.forEach(filesetMappings::put);
    ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets =
        ImmutableMap.copyOf(filesetMappings);
    try {
      state.updateFileSystemContext(skyframeActionExecutor, env, metadataHandler, expandedFilesets);
    } catch (IOException e) {
      throw new ActionExecutionException(
          "Failed to update filesystem context: ", e, action, /*catastrophe=*/ false);
    }
    try (ActionExecutionContext actionExecutionContext =
        skyframeActionExecutor.getContext(
            metadataHandler,
            metadataHandler,
            Collections.unmodifiableMap(state.expandedArtifacts),
            expandedFilesets,
            topLevelFilesets,
            state.actionFileSystem,
            skyframeDepsResult)) {
      if (!state.hasExecutedAction()) {
        state.value =
            skyframeActionExecutor.executeAction(
                env.getListener(),
                action,
                metadataHandler,
                actionStartTime,
                actionExecutionContext,
                actionLookupData);
      }
    } catch (IOException e) {
      throw new ActionExecutionException(
          "Failed to close action output", e, action, /*catastrophe=*/ false);
    }

    if (action.discoversInputs()) {
      Iterable<Artifact> newInputs = filterKnownInputs(action.getInputs(), state.inputArtifactData);
      Map<SkyKey, SkyValue> metadataFoundDuringActionExecution =
          env.getValues(toKeys(newInputs, action.getMandatoryInputs()));
      state.discoveredInputs = newInputs;
      if (env.valuesMissing()) {
        return null;
      }
      if (!metadataFoundDuringActionExecution.isEmpty()) {
        // We are in the interesting case of an action that discovered its inputs during
        // execution, and found some new ones, but the new ones were already present in the graph.
        // We must therefore cache the metadata for those new ones.
        for (Map.Entry<SkyKey, SkyValue> entry : metadataFoundDuringActionExecution.entrySet()) {
          state.inputArtifactData.putWithNoDepOwner(
              ArtifactSkyKey.artifact(entry.getKey()), (FileArtifactValue) entry.getValue());
        }
        metadataHandler =
            new ActionMetadataHandler(
                state.inputArtifactData,
                /*missingArtifactsAllowed=*/ false,
                action.getOutputs(),
                tsgm.get(),
                pathResolver,
                metadataHandler.getOutputStore());
      }
    }
    Preconditions.checkState(!env.valuesMissing(), action);
    skyframeActionExecutor.afterExecution(
        action, metadataHandler, state.token, clientEnv, actionLookupData);
    return state.value;
  }

  private static final Function<Artifact, SkyKey> TO_NONMANDATORY_SKYKEY =
      new Function<Artifact, SkyKey>() {
        @Nullable
        @Override
        public SkyKey apply(@Nullable Artifact artifact) {
          return ArtifactSkyKey.key(artifact, /*isMandatory=*/ false);
        }
      };

  private static Iterable<SkyKey> newlyDiscoveredInputsToSkyKeys(
      Iterable<Artifact> discoveredInputs, ActionInputMap inputArtifactData) {
    return Iterables.transform(
        filterKnownInputs(discoveredInputs, inputArtifactData), TO_NONMANDATORY_SKYKEY);
  }

  private static void addDiscoveredInputs(
      ActionInputMap inputData,
      Map<Artifact, Collection<Artifact>> expandedArtifacts,
      Iterable<Artifact> discoveredInputs,
      Environment env)
      throws InterruptedException {
    // We do not do a getValuesOrThrow() call for the following reasons:
    // 1. No exceptions can be thrown for non-mandatory inputs;
    // 2. Any derived inputs must be in the transitive closure of this action's inputs. Therefore,
    // if there was an error building one of them, then that exception would have percolated up to
    // this action already, through one of its declared inputs, and we would not have reached input
    // discovery.
    // Therefore there is no need to catch and rethrow exceptions as there is with #checkInputs.
    Map<SkyKey, SkyValue> nonMandatoryDiscovered =
        env.getValues(newlyDiscoveredInputsToSkyKeys(discoveredInputs, inputData));
    if (!env.valuesMissing()) {
      for (Map.Entry<SkyKey, SkyValue> entry : nonMandatoryDiscovered.entrySet()) {
        Artifact input = ArtifactSkyKey.artifact(entry.getKey());
        if (entry.getValue() instanceof TreeArtifactValue) {
          TreeArtifactValue treeValue = (TreeArtifactValue) entry.getValue();
          expandedArtifacts.put(input, ImmutableSet.<Artifact>copyOf(treeValue.getChildren()));
          for (Map.Entry<Artifact.TreeFileArtifact, FileArtifactValue> child :
              treeValue.getChildValues().entrySet()) {
            inputData.putWithNoDepOwner(child.getKey(), child.getValue());
          }
          inputData.putWithNoDepOwner(input, treeValue.getSelfData());
        } else {
          inputData.putWithNoDepOwner(input, (FileArtifactValue) entry.getValue());
        }
      }
    }
  }

  private static Object establishSkyframeDependencies(Environment env, Action action)
      throws ActionExecutionException, InterruptedException {
    // Before we may safely establish Skyframe dependencies, we must build all action inputs by
    // requesting their ArtifactValues.
    // This is very important to do, because the establishSkyframeDependencies method may request
    // FileValues for input files of this action (directly requesting them, or requesting some other
    // SkyValue whose builder requests FileValues), which may not yet exist if their generating
    // actions have not yet run.
    // See SkyframeAwareActionTest.testRaceConditionBetweenInputAcquisitionAndSkyframeDeps
    Preconditions.checkState(!env.valuesMissing(), action);

    if (action instanceof SkyframeAwareAction) {
      // Skyframe-aware actions should be executed unconditionally, i.e. bypass action cache
      // checking. See documentation of SkyframeAwareAction.
      Preconditions.checkState(action.executeUnconditionally(), action);

      try {
        return ((SkyframeAwareAction) action).establishSkyframeDependencies(env);
      } catch (SkyframeAwareAction.ExceptionBase e) {
        throw new ActionExecutionException(e, action, false);
      }
    }
    return null;
  }

  private static Iterable<SkyKey> toKeys(
      Iterable<Artifact> inputs, Iterable<Artifact> mandatoryInputs) {
    if (mandatoryInputs == null) {
      // This is a non inputs-discovering action, so no need to distinguish mandatory from regular
      // inputs.
      return Iterables.transform(
          inputs,
          new Function<Artifact, SkyKey>() {
            @Override
            public SkyKey apply(Artifact artifact) {
              return ArtifactSkyKey.key(artifact, true);
            }
          });
    } else {
      Collection<SkyKey> discoveredArtifacts = new HashSet<>();
      Set<Artifact> mandatory = Sets.newHashSet(mandatoryInputs);
      for (Artifact artifact : inputs) {
        discoveredArtifacts.add(ArtifactSkyKey.key(artifact, mandatory.contains(artifact)));
      }
      return discoveredArtifacts;
    }
  }

  private static class CheckInputResults {
    /** Metadata about Artifacts consumed by this Action. */
    private final ActionInputMap actionInputMap;
    /** Artifact expansion mapping for Runfiles tree and tree artifacts. */
    private final Map<Artifact, Collection<Artifact>> expandedArtifacts;
    /** Artifact expansion mapping for Filesets embedded in Runfiles. */
    private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets;

    public CheckInputResults(ActionInputMap actionInputMap,
        Map<Artifact, Collection<Artifact>> expandedArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets) {
      this.actionInputMap = actionInputMap;
      this.expandedArtifacts = expandedArtifacts;
      this.expandedFilesets = expandedFilesets;
    }
  }

  private interface AccumulateInputResultsFactory<S extends ActionInputMapSink, R> {
    R create(
        S actionInputMapSink,
        Map<Artifact, Collection<Artifact>> expandedArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets);
  }

  /**
   * Declare dependency on all known inputs of action. Throws exception if any are known to be
   * missing. Some inputs may not yet be in the graph, in which case the builder should abort.
   */
  private CheckInputResults checkInputs(
      Environment env,
      Action action,
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps)
      throws ActionExecutionException, InterruptedException {
    return accumulateInputs(env, action, inputDeps, ActionInputMap::new, CheckInputResults::new);
  }

  /**
   * Reconstructs the relationships between lost inputs and the direct deps responsible for them.
   */
  private ActionInputDepOwners getInputDepOwners(
      Environment env,
      Action action,
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps,
      Collection<ActionInput> lostInputs)
      throws ActionExecutionException, InterruptedException {
    return accumulateInputs(
        env,
        action,
        inputDeps,
        ignoredInputDepsSize -> new ActionInputDepOwnerMap(lostInputs),
        (actionInputMapSink, expandedArtifacts, expandedFilesets) -> actionInputMapSink);
  }

  private <S extends ActionInputMapSink, R> R accumulateInputs(
      Environment env,
      Action action,
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps,
      IntFunction<S> actionInputMapSinkFactory,
      AccumulateInputResultsFactory<S, R> accumulateInputResultsFactory)
      throws ActionExecutionException, InterruptedException {
    int missingCount = 0;
    int actionFailures = 0;
    // Only populate input data if we have the input values, otherwise they'll just go unused.
    // We still want to loop through the inputs to collect missing deps errors. During the
    // evaluator "error bubbling", we may get one last chance at reporting errors even though
    // some deps are still missing.
    boolean populateInputData = !env.valuesMissing();
    NestedSetBuilder<Cause> rootCauses = NestedSetBuilder.stableOrder();
    S inputArtifactData = actionInputMapSinkFactory.apply(populateInputData ? inputDeps.size() : 0);
    Map<Artifact, Collection<Artifact>> expandedArtifacts =
        new HashMap<>(populateInputData ? 128 : 0);
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = new HashMap<>();

    ActionExecutionException firstActionExecutionException = null;
    for (Map.Entry<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>>
        depsEntry : inputDeps.entrySet()) {
      Artifact input = ArtifactSkyKey.artifact(depsEntry.getKey());
      try {
        SkyValue value = depsEntry.getValue().get();
        if (populateInputData) {
          ActionInputMapHelper.addToMap(
              inputArtifactData,
              expandedArtifacts,
              expandedFilesets,
              input,
              value,
              env);
        }
      } catch (MissingInputFileException e) {
        missingCount++;
        if (input.getOwner() != null) {
          rootCauses.add(new LabelCause(input.getOwner(), e.getMessage()));
        }
      } catch (ActionExecutionException e) {
        actionFailures++;
        // Prefer a catastrophic exception as the one we propagate.
        if (firstActionExecutionException == null
            || (!firstActionExecutionException.isCatastrophe() && e.isCatastrophe())) {
          firstActionExecutionException = e;
        }
        rootCauses.addTransitive(e.getRootCauses());
      }
    }
    // We need to rethrow first exception because it can contain useful error message
    if (firstActionExecutionException != null) {
      if (missingCount == 0 && actionFailures == 1) {
        // In the case a single action failed, just propagate the exception upward. This avoids
        // having to copy the root causes to the upwards transitive closure.
        throw firstActionExecutionException;
      }
      throw new ActionExecutionException(
          firstActionExecutionException.getMessage(),
          firstActionExecutionException.getCause(),
          action,
          rootCauses.build(),
          firstActionExecutionException.isCatastrophe(),
          firstActionExecutionException.getExitCode());
    }

    if (missingCount > 0) {
      for (Cause missingInput : rootCauses.build()) {
        env.getListener()
            .handle(
                Event.error(
                    action.getOwner().getLocation(),
                    String.format(
                        "%s: missing input file '%s'",
                        action.getOwner().getLabel(), missingInput.getLabel())));
      }
      throw new ActionExecutionException(
          missingCount + " input file(s) do not exist",
          action,
          rootCauses.build(),
          /*catastrophe=*/ false);
    }
    return accumulateInputResultsFactory.create(
        inputArtifactData, expandedArtifacts, expandedFilesets);
  }

  private static Iterable<Artifact> filterKnownInputs(
      Iterable<Artifact> newInputs, ActionInputMap inputArtifactData) {
    return Iterables.filter(newInputs, input -> inputArtifactData.getMetadata(input) == null);
  }

  static boolean actionDependsOnBuildId(Action action) {
    // Volatile build actions may need to execute even if none of their known inputs have changed.
    // Depending on the build id ensures that these actions have a chance to execute.
    // SkyframeAwareActions do not need to depend on the build id because their volatility is due to
    // their dependence on Skyframe nodes that are not captured in the action cache. Any changes to
    // those nodes will cause this action to be rerun, so a build id dependency is unnecessary.
    return (action.isVolatile() && !(action instanceof SkyframeAwareAction))
        || action instanceof NotifyOnActionCacheHit;
  }

  /** All info/warning messages associated with actions should be always displayed. */
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Should be called once execution is over, and the intra-build cache of in-progress computations
   * should be discarded. If the cache is non-empty (due to an interrupted/failed build), failure to
   * call complete() can both cause a memory leak and incorrect results on the subsequent build.
   */
  @Override
  public void complete() {
    // Discard all remaining state (there should be none after a successful execution).
    stateMap = Maps.newConcurrentMap();
  }

  private ContinuationState getState(Action action) {
    ContinuationState state = stateMap.get(action);
    if (state == null) {
      state = new ContinuationState();
      Preconditions.checkState(stateMap.put(action, state) == null, action);
    }
    return state;
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
  private static class ContinuationState {
    AllInputs allInputs;
    /** Mutable map containing metadata for known artifacts. */
    ActionInputMap inputArtifactData = null;

    Map<Artifact, Collection<Artifact>> expandedArtifacts = null;
    Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets = null;
    Token token = null;
    Iterable<Artifact> discoveredInputs = null;
    ActionExecutionValue value = null;
    FileSystem actionFileSystem = null;

    boolean hasCollectedInputs() {
      return allInputs != null;
    }

    boolean hasArtifactData() {
      boolean result = inputArtifactData != null;
      Preconditions.checkState(result == (expandedArtifacts != null), this);
      return result;
    }

    boolean hasCheckedActionCache() {
      // If token is null because there was an action cache hit, this method is never called again
      // because we return immediately.
      return token != null;
    }

    boolean hasExecutedAction() {
      return value != null;
    }

    /** Must be called to assign values to the given variables as they change. */
    void updateFileSystemContext(
        SkyframeActionExecutor executor,
        Environment env,
        ActionMetadataHandler metadataHandler,
        ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesets)
        throws IOException {
      if (actionFileSystem != null) {
        executor.updateActionFileSystemContext(
            actionFileSystem, env, metadataHandler.getOutputStore()::injectOutputData, filesets);
      }
    }

    @Override
    public String toString() {
      return token
          + ", "
          + value
          + ", "
          + allInputs
          + ", "
          + inputArtifactData
          + ", "
          + discoveredInputs;
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
}
