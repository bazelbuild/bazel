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
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
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

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A builder for {@link ActionExecutionValue}s.
 */
public class ActionExecutionFunction implements SkyFunction {

  private static final Predicate<Artifact> IS_SOURCE_ARTIFACT = new Predicate<Artifact>() {
    @Override
    public boolean apply(Artifact input) {
      return input.isSourceArtifact();
    }
  };

  private final SkyframeActionExecutor skyframeActionExecutor;
  private final TimestampGranularityMonitor tsgm;

  public ActionExecutionFunction(SkyframeActionExecutor skyframeActionExecutor,
      TimestampGranularityMonitor tsgm) {
    this.skyframeActionExecutor = skyframeActionExecutor;
    this.tsgm = tsgm;
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
    } finally {
      declareAdditionalDependencies(env, action);
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
    // Don't initialize the cache if the result has already been computed and this is just a
    // rerun.
    FileAndMetadataCache fileAndMetadataCache = null;
    MetadataHandler metadataHandler = null;
    Token token = null;
    long actionStartTime = System.nanoTime();
    // inputArtifactData is null exactly when we know that the execution result was already
    // computed on a prior run of this SkyFunction. If it is null we don't need to initialize
    // anything -- we will get the result directly from SkyframeActionExecutor's cache.
    if (inputArtifactData != null) {
      // Check action cache to see if we need to execute anything. Checking the action cache only
      // needs to happen on the first run, since a cache hit means we'll return immediately, and
      // there'll be no second run.
      fileAndMetadataCache = new FileAndMetadataCache(
          inputArtifactData,
          expandedMiddlemen,
          skyframeActionExecutor.getExecRoot(),
          action.getOutputs(),
          // Only give the metadata cache the ability to look up Skyframe values if the action
          // might have undeclared inputs. If those undeclared inputs are generated, they are
          // present in Skyframe, so we can save a stat by looking them up directly.
          action.discoversInputs() ? env : null,
          tsgm);
      metadataHandler =
          skyframeActionExecutor.constructMetadataHandler(fileAndMetadataCache);
      token = skyframeActionExecutor.checkActionCache(action, metadataHandler,  
          new PackageRootResolverWithEnvironment(env), actionStartTime);
      if (token == Token.NEED_TO_RERUN) {
        return null;
      }
    }
    if (token == null && inputArtifactData != null) {
      // We got a hit from the action cache -- no need to execute.
      return new ActionExecutionValue(
          fileAndMetadataCache.getOutputData(),
          fileAndMetadataCache.getAdditionalOutputData());
    } else {
      ActionExecutionContext actionExecutionContext = null;
      if (inputArtifactData != null) {
        actionExecutionContext = skyframeActionExecutor.constructActionExecutionContext(
            fileAndMetadataCache,
            metadataHandler);
        if (action.discoversInputs()) {
          skyframeActionExecutor.discoverInputs(action, actionExecutionContext);
        }
      }
      // If this is the second time we are here (because the action discovers inputs, and we had
      // to restart the value builder after declaring our dependence on newly discovered inputs),
      // the result returned here is the already-computed result from the first run.
      // Similarly, if this is a shared action and the other action is the one that executed, we
      // must use that other action's value, provided here, since it is populated with metadata
      // for the outputs.
      // If this action was not shared and this is the first run of the action, this returned
      // result was computed during the call.
      return skyframeActionExecutor.executeAction(action, fileAndMetadataCache, token,
          actionStartTime, actionExecutionContext);
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

      // In case the action violates the invariant that getInputs() is a superset of
      // getMandatoryInputs(), explicitly add the mandatory inputs. See bug about an
      // "action not in canonical form" error message. Also note that we may add Skyframe edges on
      // these potentially stale deps due to the way loading inputs from the action cache functions.
      // In practice, this is safe since C++ actions (the only ones which discover inputs) only add
      // possibly stale inputs on source artifacts, which we treat as non-mandatory.
      for (Artifact artifact : mandatory) {
        discoveredArtifacts.add(ArtifactValue.key(artifact, true));
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
    // some deps are stilling missing.
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
