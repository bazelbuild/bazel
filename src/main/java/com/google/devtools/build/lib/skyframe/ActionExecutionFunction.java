// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.AlreadyReportedActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;

/**
 * A builder for {@link ActionExecutionValue}s.
 */
public class ActionExecutionFunction implements SkyFunction, CompletionReceiver {

  private static final Predicate<Artifact> IS_SOURCE_ARTIFACT = new Predicate<Artifact>() {
    @Override
    public boolean apply(Artifact input) {
      return input.isSourceArtifact();
    }
  };

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
    Map<Artifact, FileArtifactValue> inputArtifactData = null;
    Map<Artifact, Collection<Artifact>> expandedMiddlemen = null;
    boolean alreadyRan = skyframeActionExecutor.probeActionExecution(action);
    try {
      Pair<Map<Artifact, FileArtifactValue>, Map<Artifact, Collection<Artifact>>> checkedInputs =
          checkInputs(env, action, alreadyRan); // Declare deps on known inputs to action.

      if (checkedInputs != null) {
        inputArtifactData = checkedInputs.first;
        expandedMiddlemen = checkedInputs.second;
      }
    } catch (ActionExecutionException e) {
      throw new ActionExecutionFunctionException(e);
    }
    // TODO(bazel-team): Non-volatile NotifyOnActionCacheHit actions perform worse in Skyframe than
    // legacy when they are not at the top of the action graph. In legacy, they are stored
    // separately, so notifying non-dirty actions is cheap. In Skyframe, they depend on the
    // BUILD_ID, forcing invalidation of upward transitive closure on each build.
    if (action.isVolatile() || action instanceof NotifyOnActionCacheHit) {
      // Volatile build actions may need to execute even if none of their known inputs have changed.
      // Depending on the buildID ensure that these actions have a chance to execute.
      PrecomputedValue.BUILD_ID.get(env);
    }
    if (env.valuesMissing()) {
      return null;
    }

    ActionExecutionValue result;
    try {
      result = checkCacheAndExecuteIfNeeded(action, inputArtifactData, expandedMiddlemen, env);
    } catch (ActionExecutionException e) {
      // In this case we do not report the error to the action reporter because we have already
      // done it in SkyframeExecutor.reportErrorIfNotAbortingMode() method. That method
      // prints the error in the top-level reporter and also dumps the recorded StdErr for the
      // action. Label can be null in the case of, e.g., the SystemActionOwner (for build-info.txt).
      throw new ActionExecutionFunctionException(new AlreadyReportedActionExecutionException(e));
    }

    if (env.valuesMissing()) {
      return null;
    }

    return result;
  }
  
  /**
   * Skyframe implementation of {@link PackageRootResolver}. Should be used only from SkyFunctions,
   * because it uses SkyFunction.Environment for evaluation of ContainingPackageLookupValue.
   */
  private static class PackageRootResolverWithEnvironment implements PackageRootResolver {
    private final Environment env;

    public PackageRootResolverWithEnvironment(Environment env) {
      this.env = env;
    }

    @Override
    public Map<PathFragment, Root> findPackageRoots(Iterable<PathFragment> execPaths) {
      Map<PathFragment, SkyKey> depKeys = new HashMap<>(); 
      // Create SkyKeys list based on execPaths.
      for (PathFragment path : execPaths) {
        depKeys.put(path,
            ContainingPackageLookupValue.key(PackageIdentifier.createInDefaultRepo(path)));
      }
      Map<SkyKey, SkyValue> values = env.getValues(depKeys.values());
      if (env.valuesMissing()) {
        // Some values are not computed yet.
        return null;
      }
      Map<PathFragment, Root> result = new HashMap<>();
      for (PathFragment path : execPaths) {
        // TODO(bazel-team): Add check for errors here, when loading phase will be removed.
        // For now all possible errors that ContainingPackageLookupFunction can generate
        // are caught in previous phases.
        ContainingPackageLookupValue value =
            (ContainingPackageLookupValue) values.get(depKeys.get(path));
        if (value.hasContainingPackage()) {
          // We have found corresponding root for current execPath.
          result.put(path, Root.asSourceRoot(value.getContainingPackageRoot()));
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
      Map<Artifact, FileArtifactValue> inputArtifactData,
      Map<Artifact, Collection<Artifact>> expandedMiddlemen,
      Environment env) throws ActionExecutionException, InterruptedException {
    // If this is the second time we are here (because the action discovers inputs, and we had
    // to restart the value builder after declaring our dependence on newly discovered inputs),
    // the result returned here is the already-computed result from the first run.
    // Similarly, if this is a shared action and the other action is the one that executed, we
    // must use that other action's value, provided here, since it is populated with metadata
    // for the outputs.
    if (inputArtifactData == null) {
      return skyframeActionExecutor.executeAction(action, null, -1, null);
    }
    ContinuationState state;
    if (action.discoversInputs()) {
      state = getState(action);
    } else {
      state = new ContinuationState();
    }
    // This may be recreated if we discover inputs.
    FileAndMetadataCache fileAndMetadataCache = new FileAndMetadataCache(
          inputArtifactData,
          expandedMiddlemen,
          skyframeActionExecutor.getExecRoot(),
          action.getOutputs(),
          // Only give the metadata cache the ability to look up Skyframe values if the action
          // might have undeclared inputs. If those undeclared inputs are generated, they are
          // present in Skyframe, so we can save a stat by looking them up directly.
          action.discoversInputs() ? env : null,
          tsgm);
    MetadataHandler metadataHandler =
          skyframeActionExecutor.constructMetadataHandler(fileAndMetadataCache);
    long actionStartTime = System.nanoTime();
    // We only need to check the action cache if we haven't done it on a previous run.
    if (!state.hasDiscoveredInputs()) {
      Token token = skyframeActionExecutor.checkActionCache(action, metadataHandler,
          new PackageRootResolverWithEnvironment(env), actionStartTime);
      if (token == Token.NEED_TO_RERUN) {
        // Sadly, there is no state that we can preserve here across this restart.
        return null;
      }
      state.token = token;
    }

    if (state.token == null) {
      if (action.discoversInputs()) {
        // Action may have had its inputs updated. Keep track of those new inputs.
        declareAdditionalDependencies(env, action);
      }
      // We got a hit from the action cache -- no need to execute.
      return new ActionExecutionValue(
          fileAndMetadataCache.getOutputData(),
          fileAndMetadataCache.getAdditionalOutputData());
    }

    // This may be recreated if we discover inputs.
    ActionExecutionContext actionExecutionContext =
        skyframeActionExecutor.constructActionExecutionContext(fileAndMetadataCache,
            metadataHandler);
    boolean inputsDiscoveredDuringActionExecution = false;
    ActionExecutionValue result;
    Token token;
    try {
      if (action.discoversInputs()) {
        if (!state.hasDiscoveredInputs()) {
          state.discoveredInputs =
              skyframeActionExecutor.discoverInputs(action, actionExecutionContext);
          if (state.discoveredInputs == null) {
            // Action had nothing to tell us about discovered inputs before execution. We'll have to
            // add them afterwards.
            inputsDiscoveredDuringActionExecution = true;
          }
        }
        if (state.discoveredInputs != null
            && !inputArtifactData.keySet().containsAll(state.discoveredInputs)) {
          inputArtifactData = addDiscoveredInputs(inputArtifactData, state.discoveredInputs,
              env);
          if (env.valuesMissing()) {
            // This is the only place that we actually preserve meaningful state across restarts.
            return null;
          }
          fileAndMetadataCache = new FileAndMetadataCache(
              inputArtifactData,
              expandedMiddlemen,
              skyframeActionExecutor.getExecRoot(),
              action.getOutputs(),
              null,
              tsgm
          );
          actionExecutionContext = skyframeActionExecutor.constructActionExecutionContext(
              fileAndMetadataCache, fileAndMetadataCache);
        }
      }
      // Clear state before actual execution of action. It will never be needed again because
      // skyframeActionExecutor is guaranteed to have a result after this.
      token = state.token;
      if (action.discoversInputs()) {
        removeState(action);
      }
      state = null;
      result = skyframeActionExecutor.executeAction(action,
          fileAndMetadataCache, actionStartTime, actionExecutionContext);
    } finally {
      try {
        actionExecutionContext.getFileOutErr().close();
      } catch (IOException e) {
        // Nothing we can do here.
      }
      if (inputsDiscoveredDuringActionExecution) {
        declareAdditionalDependencies(env, action);
      }
    }
    skyframeActionExecutor.afterExecution(action, fileAndMetadataCache, token);
    return result;
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
    for (Map.Entry<SkyKey, SkyValue> depsEntry : data.entrySet()) {
      Artifact input = ArtifactValue.artifact(depsEntry.getKey());
      result.put(input,
          Preconditions.checkNotNull((FileArtifactValue) depsEntry.getValue(), input));
    }
    return result;
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
  private Pair<Map<Artifact, FileArtifactValue>, Map<Artifact, Collection<Artifact>>> checkInputs(
      Environment env, Action action, boolean alreadyRan) throws ActionExecutionException {
    Map<SkyKey, ValueOrException2<MissingInputFileException, ActionExecutionException>> inputDeps =
        env.getValuesOrThrow(toKeys(action.getInputs(), action.discoversInputs()
            ? action.getMandatoryInputs() : null), MissingInputFileException.class,
            ActionExecutionException.class);

    // If the action was already run, then break out early. This avoids the cost of constructing the
    // input map and expanded middlemen if they're not going to be used.
    if (alreadyRan) {
      return null;
    }

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
    Map<Artifact, Collection<Artifact>> expandedMiddlemen =
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
          expandedMiddlemen.put(input,
              Collections2.transform(aggregatingValue.getInputs(),
                  Pair.<Artifact, FileArtifactValue>firstFunction()));
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
          firstActionExecutionException.getCause(), action, rootCauses.build(), catastrophe);
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
        Collections.unmodifiableMap(expandedMiddlemen));
  }

  private static void declareAdditionalDependencies(Environment env, Action action) {
    if (action.discoversInputs()) {
      // TODO(bazel-team): Should this be all inputs, or just source files?
      env.getValues(toKeys(Iterables.filter(action.getInputs(), IS_SOURCE_ARTIFACT),
          action.getMandatoryInputs()));
    }
  }

  /**
   * All info/warning messages associated with actions should be always displayed.
   */
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

  private void removeState(Action action) {
    Preconditions.checkNotNull(stateMap.remove(action), action);
  }

  /**
   * State to save work across restarts of ActionExecutionFunction due to missing discovered inputs.
   */
  private static class ContinuationState {
    Token token = null;
    Collection<Artifact> discoveredInputs = null;

    // This will always be false for actions that don't discover their inputs, but we never restart
    // those actions in any case. For actions that do discover their inputs, they either discover
    // them before execution, in which case discoveredInputs will be non-null if that has already
    // happened, or after execution, in which case they returned null when Action#discoverInputs()
    // was called, and won't restart due to missing dependencies before execution.
    boolean hasDiscoveredInputs() {
      return discoveredInputs != null;
    }

    @Override
    public String toString() {
      return token + ", " + discoveredInputs;
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
