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
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
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
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} that creates {@link ActionExecutionValue}s. There are four points where
 * this function can abort due to missing values in the graph:
 * <ol>
 *   <li>For actions that discover inputs, if missing metadata needed to resolve an artifact from a
 *   string input in the action cache.</li>
 *   <li>If missing metadata for artifacts in inputs (including the artifacts above).</li>
 *   <li>For actions that discover inputs, if missing metadata for inputs discovered prior to
 *   execution.</li>
 *   <li>For actions that discover inputs, but do so during execution, if missing metadata for
 *   inputs discovered during execution.</li>
 * </ol>
 */
public class ActionExecutionFunction implements SkyFunction, CompletionReceiver {
  private final SkyframeActionExecutor skyframeActionExecutor;
  private final TimestampGranularityMonitor tsgm;
  private ConcurrentMap<Action, ContinuationState> stateMap;

  public ActionExecutionFunction(SkyframeActionExecutor skyframeActionExecutor,
      TimestampGranularityMonitor tsgm) {
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.tsgm = tsgm;
    stateMap = Maps.newConcurrentMap();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws ActionExecutionFunctionException,
      InterruptedException {
    Action action = (Action) skyKey.argument();
    // TODO(bazel-team): Non-volatile NotifyOnActionCacheHit actions perform worse in Skyframe than
    // legacy when they are not at the top of the action graph. In legacy, they are stored
    // separately, so notifying non-dirty actions is cheap. In Skyframe, they depend on the
    // BUILD_ID, forcing invalidation of upward transitive closure on each build.
    if ((action.isVolatile() && !(action instanceof SkyframeAwareAction))
        || action instanceof NotifyOnActionCacheHit) {
      // Volatile build actions may need to execute even if none of their known inputs have changed.
      // Depending on the buildID ensure that these actions have a chance to execute.
      PrecomputedValue.BUILD_ID.get(env);
    }
    // For restarts of this ActionExecutionFunction we use a ContinuationState variable, below, to
    // avoid redoing work. However, if two actions are shared and the first one executes, when the
    // second one goes to execute, we should detect that and short-circuit, even without taking
    // ContinuationState into account.
    boolean sharedActionAlreadyRan = skyframeActionExecutor.probeActionExecution(action);
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
    Pair<Map<Artifact, FileArtifactValue>, Map<Artifact, Collection<ArtifactFile>>> checkedInputs =
        null;
    try {
      // Declare deps on known inputs to action. We do this unconditionally to maintain our
      // invariant of asking for the same deps each build.
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps
          = env.getValuesOrThrow(toKeys(state.allInputs.getAllInputs(),
                  action.discoversInputs() ? action.getMandatoryInputs() : null),
              MissingInputFileException.class, ActionExecutionException.class);

      if (!sharedActionAlreadyRan && !state.hasArtifactData()) {
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

    try {
      establishSkyframeDependencies(env, action);
    } catch (ActionExecutionException e) {
      throw new ActionExecutionFunctionException(e);
    }
    if (env.valuesMissing()) {
      return null;
    }

    if (checkedInputs != null) {
      Preconditions.checkState(!state.hasArtifactData(), "%s %s", state, action);
      state.inputArtifactData = checkedInputs.first;
      state.expandedArtifacts = checkedInputs.second;
    }

    ActionExecutionValue result;
    try {
      result = checkCacheAndExecuteIfNeeded(action, state, env);
    } catch (ActionExecutionException e) {
      // Remove action from state map in case it's there (won't be unless it discovers inputs).
      stateMap.remove(action);
      // In this case we do not report the error to the action reporter because we have already
      // done it in SkyframeExecutor.reportErrorIfNotAbortingMode() method. That method
      // prints the error in the top-level reporter and also dumps the recorded StdErr for the
      // action. Label can be null in the case of, e.g., the SystemActionOwner (for build-info.txt).
      throw new ActionExecutionFunctionException(new AlreadyReportedActionExecutionException(e));
    }

    if (env.valuesMissing()) {
      Preconditions.checkState(stateMap.containsKey(action), action);
      return null;
    }

    // Remove action from state map in case it's there (won't be unless it discovers inputs).
    stateMap.remove(action);
    return result;
  }

  /**
   * An action's inputs needed for execution. May not just be the result of Action#getInputs(). If
   * the action cache's view of this action contains additional inputs, it will request metadata for
   * them, so we consider those inputs as dependencies of this action as well. Returns null if some
   * dependencies were missing and this ActionExecutionFunction needs to restart.
   * @throws ActionExecutionFunctionException
   */
  @Nullable
  private AllInputs collectInputs(Action action, Environment env)
      throws ActionExecutionFunctionException {
    Iterable<Artifact> allKnownInputs = Iterables.concat(
        action.getInputs(), action.getRunfilesSupplier().getArtifacts());
    if (action.inputsKnown()) {
      return new AllInputs(allKnownInputs);
    }

    Preconditions.checkState(action.discoversInputs(), action);
    PackageRootResolverWithEnvironment resolver = new PackageRootResolverWithEnvironment(env);
    Iterable<Artifact> actionCacheInputs;
    try {
      actionCacheInputs = skyframeActionExecutor.getActionCachedInputs(action, resolver);
    } catch (PackageRootResolutionException rre) {
      throw new ActionExecutionFunctionException(
          new ActionExecutionException("Failed to get cached inputs", rre, action, true));
    }
    if (actionCacheInputs == null) {
      Preconditions.checkState(env.valuesMissing(), action);
      return null;
    }
    return new AllInputs(allKnownInputs, actionCacheInputs, resolver.keysRequested);
  }

  private static class AllInputs {
    final Iterable<Artifact> defaultInputs;
    @Nullable
    final Iterable<Artifact> actionCacheInputs;
    @Nullable
    final List<SkyKey> keysRequested;

    AllInputs(Iterable<Artifact> defaultInputs) {
      this.defaultInputs = Preconditions.checkNotNull(defaultInputs);
      this.actionCacheInputs = null;
      this.keysRequested = null;
    }

    AllInputs(Iterable<Artifact> defaultInputs, Iterable<Artifact> actionCacheInputs,
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

    public PackageRootResolverWithEnvironment(Environment env) {
      this.env = env;
    }

    @Override
    public Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
        throws PackageRootResolutionException {
      Preconditions.checkState(keysRequested.isEmpty(),
          "resolver should only be called once: %s %s", keysRequested, execPaths);
      // Create SkyKeys list based on execPaths.
      Map<PathFragment, SkyKey> depKeys = new HashMap<>();
      for (PathFragment path : execPaths) {
        PathFragment parent = Preconditions.checkNotNull(
            path.getParentDirectory(), "Must pass in files, not root directory");
        Preconditions.checkArgument(!parent.isAbsolute(), path);
        SkyKey depKey =
            ContainingPackageLookupValue.key(PackageIdentifier.createInDefaultRepo(parent));
        depKeys.put(path, depKey);
        keysRequested.add(depKey);
      }

      Map<SkyKey,
          ValueOrException2<NoSuchPackageException, InconsistentFilesystemException>> values =
              env.getValuesOrThrow(depKeys.values(), NoSuchPackageException.class,
                  InconsistentFilesystemException.class);
      // Check values even if some are missing so that we can throw an appropriate exception if
      // needed.

      Map<PathFragment, Root> result = new HashMap<>();
      for (PathFragment path : execPaths) {
        ContainingPackageLookupValue value;
        try {
          value = (ContainingPackageLookupValue) values.get(depKeys.get(path)).get();
        } catch (NoSuchPackageException | InconsistentFilesystemException e) {
          throw new PackageRootResolutionException("Could not determine containing package for "
              + path, e);
        }

        if (value == null) {
          Preconditions.checkState(env.valuesMissing(), path);
          continue;
        }
        if (value.hasContainingPackage()) {
          // We have found corresponding root for current execPath.
          result.put(path, Root.asSourceRoot(value.getContainingPackageRoot()));
        } else {
          // We haven't found corresponding root for current execPath.
          result.put(path, null);
        }
      }

      // If some values are missing, return null.
      return env.valuesMissing() ? null : result;
    }
  }

  private ActionExecutionValue checkCacheAndExecuteIfNeeded(
      Action action,
      ContinuationState state,
      Environment env) throws ActionExecutionException, InterruptedException {
    // If this is a shared action and the other action is the one that executed, we must use that
    // other action's value, provided here, since it is populated with metadata for the outputs.
    if (!state.hasArtifactData()) {
      return skyframeActionExecutor.executeAction(action, null, -1, null);
    }
    // This may be recreated if we discover inputs.
    ActionMetadataHandler metadataHandler = new ActionMetadataHandler(state.inputArtifactData,
        action.getOutputs(), tsgm);
    long actionStartTime = System.nanoTime();
    // We only need to check the action cache if we haven't done it on a previous run.
    if (!state.hasCheckedActionCache()) {
      state.token = skyframeActionExecutor.checkActionCache(action, metadataHandler,
          actionStartTime, state.allInputs.actionCacheInputs);
    }

    if (state.token == null) {
      // We got a hit from the action cache -- no need to execute.
      return new ActionExecutionValue(
          metadataHandler.getOutputArtifactFileData(),
          ImmutableMap.<Artifact, TreeArtifactValue>of(),
          metadataHandler.getAdditionalOutputData());
    }

    // This may be recreated if we discover inputs.
    PerActionFileCache perActionFileCache = new PerActionFileCache(state.inputArtifactData);
    ActionExecutionContext actionExecutionContext = null;
    try {
      if (action.discoversInputs()) {
        if (!state.hasDiscoveredInputs()) {
          try {
            state.discoveredInputs = skyframeActionExecutor.discoverInputs(action,
                perActionFileCache, metadataHandler, env);
          } catch (MissingDepException e) {
            Preconditions.checkState(env.valuesMissing(), action);
            return null;
          }
        }
        // state.discoveredInputs can be null even after include scanning if action discovers them
        // during execution.
        if (state.discoveredInputs != null
            && !state.inputArtifactData.keySet().containsAll(state.discoveredInputs)) {
          Map<Artifact, FileArtifactValue> inputArtifactData =
              addDiscoveredInputs(state.inputArtifactData, state.discoveredInputs, env);
          if (env.valuesMissing()) {
            return null;
          }
          state.inputArtifactData = inputArtifactData;
          perActionFileCache = new PerActionFileCache(state.inputArtifactData);
          metadataHandler =
              new ActionMetadataHandler(state.inputArtifactData, action.getOutputs(), tsgm);
        }
      }
      actionExecutionContext =
          skyframeActionExecutor.getContext(perActionFileCache,
              metadataHandler, state.expandedArtifacts);
      if (!state.hasExecutedAction()) {
        state.value = skyframeActionExecutor.executeAction(action,
            metadataHandler, actionStartTime, actionExecutionContext);
      }
    } finally {
      if (actionExecutionContext != null) {
        try {
          actionExecutionContext.getFileOutErr().close();
        } catch (IOException e) {
          // Nothing we can do here.
        }
      }
    }
    if (action.discoversInputs()) {
      Map<Artifact, FileArtifactValue> metadataFoundDuringActionExecution =
          declareAdditionalDependencies(env, action, state.inputArtifactData.keySet());
      if (state.discoveredInputs == null) {
        // Include scanning didn't find anything beforehand -- these are the definitive discovered
        // inputs.
        state.discoveredInputs = metadataFoundDuringActionExecution.keySet();
        if (env.valuesMissing()) {
          return null;
        }
        if (!metadataFoundDuringActionExecution.isEmpty()) {
          // We are in the interesting case of an action that discovered its inputs during
          // execution, and found some new ones, but the new ones were already present in the graph.
          // We must therefore cache the metadata for those new ones.
          Map<Artifact, FileArtifactValue> inputArtifactData = new HashMap<>();
          inputArtifactData.putAll(state.inputArtifactData);
          inputArtifactData.putAll(metadataFoundDuringActionExecution);
          state.inputArtifactData = inputArtifactData;
          metadataHandler =
              new ActionMetadataHandler(state.inputArtifactData, action.getOutputs(), tsgm);
        }
      } else if (!metadataFoundDuringActionExecution.isEmpty()) {
        // The action has run and discovered more inputs. This is a bug, probably the result of
        // the action dynamically executing locally instead of remotely, and a discrepancy between
        // our include scanning and the action's compiler. Fail the build so that the user notices,
        // and also report the issue.
        String errorMessage =
            action.prettyPrint()
                + " discovered unexpected inputs. This indicates a mismatch between "
                + Constants.PRODUCT_NAME
                + " and the action's compiler. Please report this issue. The ";
        if (metadataFoundDuringActionExecution.size() > 10) {
          errorMessage += "first ten ";
        }
        errorMessage += "additional inputs found were: ";
        int artifactPrinted = 0;
        for (Artifact extraArtifact : metadataFoundDuringActionExecution.keySet()) {
          if (artifactPrinted >= 10) {
            break;
          }
          if (artifactPrinted > 0) {
            errorMessage += ", ";
          }
          artifactPrinted++;
          errorMessage += extraArtifact.prettyPrint();
        }
        ActionExecutionException exception =
            new ActionExecutionException(errorMessage, action, /*catastrophe=*/ false);
        LoggingUtil.logToRemote(Level.SEVERE, errorMessage, exception);
        throw exception;
      }
    }
    Preconditions.checkState(!env.valuesMissing(), action);
    skyframeActionExecutor.afterExecution(action, metadataHandler, state.token);
    return state.value;
  }

  private static Map<Artifact, FileArtifactValue> addDiscoveredInputs(
      Map<Artifact, FileArtifactValue> originalInputData, Collection<Artifact> discoveredInputs,
      Environment env) {
    Map<Artifact, FileArtifactValue> result = new HashMap<>(originalInputData);
    Set<SkyKey> keys = new HashSet<>();
    for (Artifact artifact : discoveredInputs) {
      if (!result.containsKey(artifact)) {
        // Note that if the artifact is derived, the mandatory flag is ignored.
        keys.add(ArtifactValue.key(artifact, /*mandatory=*/false));
      }
    }
    // We do not do a getValuesOrThrow() call for the following reasons:
    // 1. No exceptions can be thrown for non-mandatory inputs;
    // 2. Any derived inputs must be in the transitive closure of this action's inputs. Therefore,
    // if there was an error building one of them, then that exception would have percolated up to
    // this action already, through one of its declared inputs, and we would not have reached input
    // discovery.
    // Therefore there is no need to catch and rethrow exceptions as there is with #checkInputs.
    Map<SkyKey, SkyValue> data = env.getValues(keys);
    if (env.valuesMissing()) {
      return null;
    }
    result.putAll(transformArtifactMetadata(data));
    return result;
  }

  private void establishSkyframeDependencies(Environment env, Action action)
      throws ActionExecutionException {
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
        ((SkyframeAwareAction) action).establishSkyframeDependencies(env);
      } catch (SkyframeAwareAction.ExceptionBase e) {
        throw new ActionExecutionException(e, action, false);
      }
    }
  }

  private static Iterable<SkyKey> toKeys(Iterable<Artifact> inputs,
      Iterable<Artifact> mandatoryInputs) {
    if (mandatoryInputs == null) {
      // This is a non inputs-discovering action, so no need to distinguish mandatory from regular
      // inputs.
      return Iterables.transform(inputs, new Function<Artifact, SkyKey>() {
        @Override
        public SkyKey apply(Artifact artifact) {
          return ArtifactValue.key(artifact, true);
        }
      });
    } else {
      Collection<SkyKey> discoveredArtifacts = new HashSet<>();
      Set<Artifact> mandatory = Sets.newHashSet(mandatoryInputs);
      for (Artifact artifact : inputs) {
        discoveredArtifacts.add(ArtifactValue.key(artifact, mandatory.contains(artifact)));
      }
      return discoveredArtifacts;
    }
  }

  /**
   * Declare dependency on all known inputs of action. Throws exception if any are known to be
   * missing. Some inputs may not yet be in the graph, in which case the builder should abort.
   */
  private Pair<Map<Artifact, FileArtifactValue>, Map<Artifact, Collection<ArtifactFile>>>
  checkInputs(Environment env, Action action,
      Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps)
      throws ActionExecutionException {
    int missingCount = 0;
    int actionFailures = 0;
    boolean catastrophe = false;
    // Only populate input data if we have the input values, otherwise they'll just go unused.
    // We still want to loop through the inputs to collect missing deps errors. During the
    // evaluator "error bubbling", we may get one last chance at reporting errors even though
    // some deps are still missing.
    boolean populateInputData = !env.valuesMissing();
    NestedSetBuilder<Label> rootCauses = NestedSetBuilder.stableOrder();
    Map<Artifact, FileArtifactValue> inputArtifactData =
        new HashMap<>(populateInputData ? inputDeps.size() : 0);
    Map<Artifact, Collection<ArtifactFile>> expandedArtifacts =
        new HashMap<>(populateInputData ? 128 : 0);

    ActionExecutionException firstActionExecutionException = null;
    for (Map.Entry<SkyKey, ValueOrException2<MissingInputFileException,
        ActionExecutionException>> depsEntry : inputDeps.entrySet()) {
      Artifact input = ArtifactValue.artifact(depsEntry.getKey());
      try {
        ArtifactValue value = (ArtifactValue) depsEntry.getValue().get();
        if (populateInputData && value instanceof AggregatingArtifactValue) {
          AggregatingArtifactValue aggregatingValue = (AggregatingArtifactValue) value;
          for (Pair<Artifact, FileArtifactValue> entry : aggregatingValue.getInputs()) {
            inputArtifactData.put(entry.first, entry.second);
          }
          // We have to cache the "digest" of the aggregating value itself, because the action cache
          // checker may want it.
          inputArtifactData.put(input, aggregatingValue.getSelfData());
          expandedArtifacts.put(input,
              Collections2.transform(aggregatingValue.getInputs(),
                  new Function<Pair<Artifact, FileArtifactValue>, ArtifactFile>() {
                    @Override
                    public ArtifactFile apply(Pair<Artifact, FileArtifactValue> pair) {
                      return pair.first;
                    }
                  }));
        } else if (populateInputData && value instanceof FileArtifactValue) {
          // TODO(bazel-team): Make sure middleman "virtual" artifact data is properly processed.
          inputArtifactData.put(input, (FileArtifactValue) value);
        }
      } catch (MissingInputFileException e) {
        missingCount++;
        if (input.getOwner() != null) {
          rootCauses.add(input.getOwner());
        }
      } catch (ActionExecutionException e) {
        actionFailures++;
        if (firstActionExecutionException == null) {
          firstActionExecutionException = e;
        }
        catastrophe = catastrophe || e.isCatastrophe();
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
      throw new ActionExecutionException(firstActionExecutionException.getMessage(),
          firstActionExecutionException.getCause(), action, rootCauses.build(), catastrophe,
          firstActionExecutionException.getExitCode());
    }

    if (missingCount > 0) {
      for (Label missingInput : rootCauses.build()) {
        env.getListener().handle(Event.error(action.getOwner().getLocation(), String.format(
            "%s: missing input file '%s'", action.getOwner().getLabel(), missingInput)));
      }
      throw new ActionExecutionException(missingCount + " input file(s) do not exist", action,
          rootCauses.build(), /*catastrophe=*/false);
    }
    return Pair.of(
        Collections.unmodifiableMap(inputArtifactData),
        Collections.unmodifiableMap(expandedArtifacts));
  }

  /**
   * Returns a map of artifact to artifact metadata for any of {@code action}s inputs that are not
   * already in {@code knownInputs}. If some metadata was not available yet, the artifact is still
   * present in the map but with null metadata.
   */
  private static Map<Artifact, FileArtifactValue> declareAdditionalDependencies(Environment env,
      Action action, Set<Artifact> knownInputs) {
    Preconditions.checkState(action.discoversInputs(), action);
    Iterable<Artifact> newArtifacts =
        Iterables.filter(action.getInputs(), Predicates.not(Predicates.in(knownInputs)));
    return transformArtifactMetadata(
        env.getValues(toKeys(newArtifacts, action.getMandatoryInputs())));
  }

  private static Map<Artifact, FileArtifactValue> transformArtifactMetadata(
      Map<SkyKey, SkyValue> map) {
    Map<Artifact, FileArtifactValue> result = new HashMap<>(); // May contain null values.
    for (Map.Entry<SkyKey, SkyValue> entry : map.entrySet()) {
      result.put(ArtifactValue.artifact(entry.getKey()), (FileArtifactValue) entry.getValue());
    }
    return result;
  }

  /**
   * All info/warning messages associated with actions should be always displayed.
   */
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Exception to be thrown if an action is missing Skyframe dependencies that it finds are missing
   * during execution/input discovery.
   */
  public static class MissingDepException extends RuntimeException {}

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
   * <ol>
   *   <li>If not all known input metadata (coming from Action#getInputs) is available yet, then the
   *   calculated set of inputs (including the inputs resolved from the action cache) is saved.</li>
   *   <li>If not all discovered inputs' metadata is available yet, then the known input metadata
   *   together with the set of discovered inputs is saved, as well as the Token used to identify
   *   this action to the action cache.</li>
   *   <li>If, after execution, new inputs are discovered whose metadata is not yet available, then
   *   the same data as in the previous case is saved, along with the actual result of execution.
   *   </li>
   * </ol>
   */
  private static class ContinuationState {
    AllInputs allInputs;
    Map<Artifact, FileArtifactValue> inputArtifactData = null;
    Map<Artifact, Collection<ArtifactFile>> expandedArtifacts = null;
    Token token = null;
    Collection<Artifact> discoveredInputs = null;
    ActionExecutionValue value = null;

    boolean hasCollectedInputs() {
      return allInputs != null;
    }

    boolean hasArtifactData() {
      boolean result = inputArtifactData != null;
      Preconditions.checkState(result == (expandedArtifacts != null), this);
      return result;
    }

    // This will always be false for actions that don't discover their inputs, but we never restart
    // those actions in any case. For actions that do discover their inputs, they either discover
    // them before execution, in which case discoveredInputs will be non-null if that has already
    // happened, or after execution, in which case we set discoveredInputs then.
    boolean hasDiscoveredInputs() {
      return discoveredInputs != null;
    }

    boolean hasCheckedActionCache() {
      // If token is null because there was an action cache hit, this method is never called again
      // because we return immediately.
      return token != null;
    }

    boolean hasExecutedAction() {
      return value != null;
    }

    @Override
    public String toString() {
      return token + ", " + value + ", " + allInputs + ", " + inputArtifactData + ", "
          + discoveredInputs;
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ActionExecutionFunction#compute}.
   */
  private static final class ActionExecutionFunctionException extends SkyFunctionException {

    private final ActionExecutionException actionException;

    public ActionExecutionFunctionException(ActionExecutionException e) {
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
