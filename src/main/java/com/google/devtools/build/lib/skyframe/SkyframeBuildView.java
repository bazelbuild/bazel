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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ActionLookupValue.ActionLookupKey;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionValue.BuildInfoKeyAndConfig;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ConflictException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Skyframe-based driver of analysis.
 *
 * <p>Covers enough functionality to work as a substitute for {@code BuildView#configureTargets}.
 */
public final class SkyframeBuildView {

  private final ConfiguredTargetFactory factory;
  private final ArtifactFactory artifactFactory;
  @Nullable private EventHandler warningListener;
  private final SkyframeExecutor skyframeExecutor;
  private final Runnable legacyDataCleaner;
  private final BinTools binTools;
  private boolean enableAnalysis = false;

  // This hack allows us to see when a configured target has been invalidated, and thus when the set
  // of artifact conflicts needs to be recomputed (whenever a configured target has been invalidated
  // or newly evaluated).
  private final EvaluationProgressReceiver invalidationReceiver =
      new ConfiguredTargetValueInvalidationReceiver();
  private final Set<SkyKey> evaluatedConfiguredTargets = Sets.newConcurrentHashSet();
  // Used to see if checks of graph consistency need to be done after analysis.
  private volatile boolean someConfiguredTargetEvaluated = false;

  // We keep the set of invalidated configuration targets so that we can know if something
  // has been invalidated after graph pruning has been executed.
  private Set<ConfiguredTargetValue> dirtyConfiguredTargets = Sets.newConcurrentHashSet();
  private volatile boolean anyConfiguredTargetDeleted = false;
  private SkyKey configurationKey = null;

  public SkyframeBuildView(ConfiguredTargetFactory factory,
      ArtifactFactory artifactFactory,
      SkyframeExecutor skyframeExecutor, Runnable legacyDataCleaner,  BinTools binTools) {
    this.factory = factory;
    this.artifactFactory = artifactFactory;
    this.skyframeExecutor = skyframeExecutor;
    this.legacyDataCleaner = legacyDataCleaner;
    this.binTools = binTools;
    skyframeExecutor.setArtifactFactoryAndBinTools(artifactFactory, binTools);
  }

  public void setWarningListener(@Nullable EventHandler warningListener) {
    this.warningListener = warningListener;
  }

  public void setConfigurationSkyKey(SkyKey skyKey) {
    this.configurationKey = skyKey;
  }

  public void resetEvaluatedConfiguredTargetKeysSet() {
    evaluatedConfiguredTargets.clear();
  }

  public Set<SkyKey> getEvaluatedTargetKeys() {
    return ImmutableSet.copyOf(evaluatedConfiguredTargets);
  }

  private void setDeserializedArtifactOwners() throws ViewCreationFailedException {
    Map<PathFragment, Artifact> deserializedArtifactMap =
        artifactFactory.getDeserializedArtifacts();
    Set<Artifact> deserializedArtifacts = new HashSet<>();
    for (Artifact artifact : deserializedArtifactMap.values()) {
      if (!artifact.getExecPath().getBaseName().endsWith(".gcda")) {
        // gcda files are classified as generated artifacts, but are not actually generated. All
        // others need owners.
        deserializedArtifacts.add(artifact);
      }
    }
    if (deserializedArtifacts.isEmpty()) {
      // If there are no deserialized artifacts to process, don't pay the price of iterating over
      // the graph.
      return;
    }
    for (Map.Entry<SkyKey, ActionLookupValue> entry :
      skyframeExecutor.getActionLookupValueMap().entrySet()) {
      for (Action action : entry.getValue().getActionsForFindingArtifactOwners()) {
        for (Artifact output : action.getOutputs()) {
          Artifact deserializedArtifact = deserializedArtifactMap.get(output.getExecPath());
          if (deserializedArtifact != null) {
            deserializedArtifact.setArtifactOwner((ActionLookupKey) entry.getKey().argument());
            deserializedArtifacts.remove(deserializedArtifact);
          }
        }
      }
    }
    if (!deserializedArtifacts.isEmpty()) {
      throw new ViewCreationFailedException("These artifacts were read in from the FDO profile but"
      + " have no generating action that could be found. If you are confident that your profile was"
      + " collected from the same source state at which you're building, please report this:\n"
      + Artifact.asExecPaths(deserializedArtifacts));
    }
    artifactFactory.clearDeserializedArtifacts();
  }

  /**
   * Analyzes the specified targets using Skyframe as the driving framework.
   *
   * @return the configured targets that should be built
   */
  public Collection<ConfiguredTarget> configureTargets(List<ConfiguredTargetKey> values,
      EventBus eventBus, boolean keepGoing)
          throws InterruptedException, ViewCreationFailedException {
    enableAnalysis(true);
    EvaluationResult<ConfiguredTargetValue> result;
    try {
      result = skyframeExecutor.configureTargets(values, keepGoing);
    } finally {
      enableAnalysis(false);
    }
    // For Skyframe m1, note that we already reported action conflicts during action registration
    // in the legacy action graph.
    ImmutableMap<Action, ConflictException> badActions = skyframeExecutor.findArtifactConflicts();

    // Filter out all CTs that have a bad action and convert to a list of configured targets. This
    // code ensures that the resulting list of configured targets has the same order as the incoming
    // list of values, i.e., that the order is deterministic.
    Collection<ConfiguredTarget> goodCts = Lists.newArrayListWithCapacity(values.size());
    for (ConfiguredTargetKey value : values) {
      ConfiguredTargetValue ctValue = result.get(ConfiguredTargetValue.key(value));
      if (ctValue == null) {
        continue;
      }
      goodCts.add(ctValue.getConfiguredTarget());
    }

    if (!result.hasError() && badActions.isEmpty()) {
      setDeserializedArtifactOwners();
      return goodCts;
    }

    // --nokeep_going so we fail with an exception for the first error.
    // TODO(bazel-team): We might want to report the other errors through the event bus but
    // for keeping this code in parity with legacy we just report the first error for now.
    if (!keepGoing) {
      for (Map.Entry<Action, ConflictException> bad : badActions.entrySet()) {
        ConflictException ex = bad.getValue();
        try {
          ex.rethrowTyped();
        } catch (MutableActionGraph.ActionConflictException ace) {
          ace.reportTo(skyframeExecutor.getReporter());
          String errorMsg = "Analysis of target '" + bad.getKey().getOwner().getLabel()
              + "' failed; build aborted";
          throw new ViewCreationFailedException(errorMsg);
        } catch (ArtifactPrefixConflictException apce) {
          skyframeExecutor.getReporter().handle(Event.error(apce.getMessage()));
        }
        throw new ViewCreationFailedException(ex.getMessage());
      }

      Map.Entry<SkyKey, ErrorInfo> error = result.errorMap().entrySet().iterator().next();
      SkyKey topLevel = error.getKey();
      ErrorInfo errorInfo = error.getValue();
      assertSaneAnalysisError(errorInfo, topLevel);
      skyframeExecutor.getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), topLevel,
          skyframeExecutor.getReporter());
      Throwable cause = errorInfo.getException();
      Preconditions.checkState(cause != null || !Iterables.isEmpty(errorInfo.getCycleInfo()),
          errorInfo);
      String errorMsg = "Analysis of target '" + ConfiguredTargetValue.extractLabel(topLevel)
          + "' failed; build aborted";
      throw new ViewCreationFailedException(errorMsg);
    }

    // --keep_going : We notify the error and return a ConfiguredTargetValue
    for (Map.Entry<SkyKey, ErrorInfo> errorEntry : result.errorMap().entrySet()) {
      if (values.contains(errorEntry.getKey().argument())) {
        SkyKey errorKey = errorEntry.getKey();
        ConfiguredTargetKey label = (ConfiguredTargetKey) errorKey.argument();
        ErrorInfo errorInfo = errorEntry.getValue();
        assertSaneAnalysisError(errorInfo, errorKey);

        skyframeExecutor.getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), errorKey,
            skyframeExecutor.getReporter());
        // We try to get the root cause key first from ErrorInfo rootCauses. If we don't have one
        // we try to use the cycle culprit if the error is a cycle. Otherwise we use the top-level
        // error key.
        Label root;
        if (!Iterables.isEmpty(errorEntry.getValue().getRootCauses())) {
          SkyKey culprit = Preconditions.checkNotNull(Iterables.getFirst(
              errorEntry.getValue().getRootCauses(), null));
          root = ((ConfiguredTargetKey) culprit.argument()).getLabel();
        } else {
          root = maybeGetConfiguredTargetCycleCulprit(errorInfo.getCycleInfo());
        }
        if (warningListener != null) {
          warningListener.handle(Event.warn("errors encountered while analyzing target '"
              + label + "': it will not be built"));
        }
        eventBus.post(new AnalysisFailureEvent(
            LabelAndConfiguration.of(label.getLabel(), label.getConfiguration()), root));
      }
    }

    Collection<Exception> reportedExceptions = Sets.newHashSet();
    for (Map.Entry<Action, ConflictException> bad : badActions.entrySet()) {
      ConflictException ex = bad.getValue();
      try {
        ex.rethrowTyped();
      } catch (MutableActionGraph.ActionConflictException ace) {
        ace.reportTo(skyframeExecutor.getReporter());
        if (warningListener != null) {
          warningListener.handle(Event.warn("errors encountered while analyzing target '"
              + bad.getKey().getOwner().getLabel() + "': it will not be built"));
        }
      } catch (ArtifactPrefixConflictException apce) {
        if (reportedExceptions.add(apce)) {
          skyframeExecutor.getReporter().handle(Event.error(apce.getMessage()));
        }
      }
    }

    if (!badActions.isEmpty()) {
      // In order to determine the set of configured targets transitively error free from action
      // conflict issues, we run a post-processing update() that uses the bad action map.
      EvaluationResult<PostConfiguredTargetValue> actionConflictResult =
          skyframeExecutor.postConfigureTargets(values, keepGoing, badActions);

      goodCts = Lists.newArrayListWithCapacity(values.size());
      for (ConfiguredTargetKey value : values) {
        PostConfiguredTargetValue postCt =
            actionConflictResult.get(PostConfiguredTargetValue.key(value));
        if (postCt != null) {
          goodCts.add(postCt.getCt());
        }
      }
    }
    setDeserializedArtifactOwners();
    return goodCts;
  }

  @Nullable
  Label maybeGetConfiguredTargetCycleCulprit(Iterable<CycleInfo> cycleInfos) {
    for (CycleInfo cycleInfo : cycleInfos) {
      SkyKey culprit = Iterables.getFirst(cycleInfo.getCycle(), null);
      if (culprit == null) {
        continue;
      }
      if (culprit.functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
        return ((LabelAndConfiguration) culprit.argument()).getLabel();
      }
    }
    return null;
  }

  private static void assertSaneAnalysisError(ErrorInfo errorInfo, SkyKey key) {
    Throwable cause = errorInfo.getException();
    if (cause != null) {
      // We should only be trying to configure targets when the loading phase succeeds, meaning
      // that the only errors should be analysis errors.
      Preconditions.checkState(cause instanceof ConfiguredValueCreationException,
          "%s -> %s", key, errorInfo);
    }
  }

  ArtifactFactory getArtifactFactory() {
    return artifactFactory;
  }

  @Nullable
  EventHandler getWarningListener() {
    return warningListener;
  }

  /**
   * Because we don't know what build-info artifacts this configured target may request, we
   * conservatively register a dep on all of them.
   */
  // TODO(bazel-team): Allow analysis to return null so the value builder can exit and wait for a
  // restart deps are not present.
  private boolean getWorkspaceStatusValues(Environment env) {
    env.getValue(WorkspaceStatusValue.SKY_KEY);
    Map<BuildInfoKey, BuildInfoFactory> buildInfoFactories =
        PrecomputedValue.BUILD_INFO_FACTORIES.get(env);
    if (buildInfoFactories == null) {
      return false;
    }
    BuildConfigurationCollection configurations = getBuildConfigurationCollection(env);
    if (configurations == null) {
      return false;
    }
    // These factories may each create their own build info artifacts, all depending on the basic
    // build-info.txt and build-changelist.txt.
    List<SkyKey> depKeys = Lists.newArrayList();
    for (BuildInfoKey key : buildInfoFactories.keySet()) {
      for (BuildConfiguration config : configurations.getAllConfigurations()) {
        if (buildInfoFactories.get(key).isEnabled(config)) {
          depKeys.add(BuildInfoCollectionValue.key(new BuildInfoKeyAndConfig(key, config)));
        }
      }
    }
    env.getValues(depKeys);
    return !env.valuesMissing();
  }

  /** Returns null if any build-info values are not ready. */
  @Nullable
  CachingAnalysisEnvironment createAnalysisEnvironment(ArtifactOwner owner,
      boolean isSystemEnv, boolean extendedSanityChecks, EventHandler eventHandler,
      Environment env, boolean allowRegisteringActions) {
    if (!getWorkspaceStatusValues(env)) {
      return null;
    }
    return new CachingAnalysisEnvironment(
        artifactFactory, owner, isSystemEnv, extendedSanityChecks, eventHandler, env,
        allowRegisteringActions, binTools);
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   *
   * <p>For use in {@code ConfiguredTargetFunction}.
   *
   * <p>Returns null if Skyframe deps are missing or upon certain errors.
   */
  @Nullable
  ConfiguredTarget createConfiguredTarget(Target target, BuildConfiguration configuration,
      CachingAnalysisEnvironment analysisEnvironment,
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      Set<ConfigMatchingProvider> configConditions)
      throws InterruptedException {
    Preconditions.checkState(enableAnalysis,
        "Already in execution phase %s %s", target, configuration);
    return factory.createConfiguredTarget(analysisEnvironment, artifactFactory, target,
        configuration, prerequisiteMap, configConditions);
  }

  @Nullable
  public Aspect createAspect(
      AnalysisEnvironment env, RuleConfiguredTarget associatedTarget,
      ConfiguredAspectFactory aspectFactory,
      ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap,
      Set<ConfigMatchingProvider> configConditions) {
    return factory.createAspect(
        env, associatedTarget, aspectFactory, prerequisiteMap, configConditions);
  }

  @Nullable
  private BuildConfigurationCollection getBuildConfigurationCollection(Environment env) {
    ConfigurationCollectionValue configurationsValue =
        (ConfigurationCollectionValue) env.getValue(configurationKey);
    return configurationsValue == null ? null : configurationsValue.getConfigurationCollection();
  }

  @Nullable
  SkyframeDependencyResolver createDependencyResolver(Environment env) {
    BuildConfigurationCollection configurations = getBuildConfigurationCollection(env);
    return configurations == null ? null : new SkyframeDependencyResolver(env);
  }

  /**
   * Workaround to clear all legacy data, like the action graph and the artifact factory. We need
   * to clear them to avoid conflicts.
   * TODO(bazel-team): Remove this workaround. [skyframe-execution]
   */
  void clearLegacyData() {
    legacyDataCleaner.run();
  }

  /**
   * Hack to invalidate actions in legacy action graph when their values are invalidated in
   * skyframe.
   */
  EvaluationProgressReceiver getInvalidationReceiver() {
    return invalidationReceiver;
  }

  /** Clear the invalidated configured targets detected during loading and analysis phases. */
  public void clearInvalidatedConfiguredTargets() {
    dirtyConfiguredTargets = Sets.newConcurrentHashSet();
    anyConfiguredTargetDeleted = false;
  }

  public boolean isSomeConfiguredTargetInvalidated() {
    return anyConfiguredTargetDeleted || !dirtyConfiguredTargets.isEmpty();
  }

  /**
   * Called from SkyframeExecutor to see whether the graph needs to be checked for artifact
   * conflicts. Returns true if some configured target has been evaluated since the last time the
   * graph was checked for artifact conflicts (with that last time marked by a call to
   * {@link #resetEvaluatedConfiguredTargetFlag()}).
   */
  boolean isSomeConfiguredTargetEvaluated() {
    Preconditions.checkState(!enableAnalysis);
    return someConfiguredTargetEvaluated;
  }

  /**
   * Called from SkyframeExecutor after the graph is checked for artifact conflicts so that
   * the next time {@link #isSomeConfiguredTargetEvaluated} is called, it will return true only if
   * some configured target has been evaluated since the last check for artifact conflicts.
   */
  void resetEvaluatedConfiguredTargetFlag() {
    someConfiguredTargetEvaluated = false;
  }

  /**
   * {@link #createConfiguredTarget} will only create configured targets if this is set to true. It
   * should be set to true before any Skyframe update call that might call into {@link
   * #createConfiguredTarget}, and false immediately after the call. Use it to fail-fast in the case
   * that a target is requested for analysis not during the analysis phase.
   */
  void enableAnalysis(boolean enable) {
    this.enableAnalysis = enable;
  }

  private class ConfiguredTargetValueInvalidationReceiver implements EvaluationProgressReceiver {
    @Override
    public void invalidated(SkyValue value, InvalidationState state) {
      if (value instanceof ConfiguredTargetValue) {
        ConfiguredTargetValue ctValue = (ConfiguredTargetValue) value;
        // If the value was just dirtied and not deleted, then it may not be truly invalid, since
        // it may later get re-validated.
        if (state == InvalidationState.DELETED) {
          anyConfiguredTargetDeleted = true;
        } else {
          dirtyConfiguredTargets.add(ctValue);
        }
      }
    }

    @Override
    public void enqueueing(SkyKey skyKey) {}

    @Override
    public void evaluated(SkyKey skyKey, SkyValue value, EvaluationState state) {
      if (skyKey.functionName() == SkyFunctions.CONFIGURED_TARGET && value != null) {
        if (state == EvaluationState.BUILT) {
          evaluatedConfiguredTargets.add(skyKey);
          // During multithreaded operation, this is only set to true, so no concurrency issues.
          someConfiguredTargetEvaluated = true;
        }
        Preconditions.checkNotNull(value, "%s %s", skyKey, state);
        ConfiguredTargetValue ctValue = (ConfiguredTargetValue) value;
        dirtyConfiguredTargets.remove(ctValue);
      }
    }
  }
}
