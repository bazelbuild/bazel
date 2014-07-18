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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageNotInCacheException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionNode.BuildInfoKeyAndConfig;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetNodeBuilder.ConfiguredNodeCreationException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.AnalysisFailureEvent;
import com.google.devtools.build.lib.view.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.ConfiguredTargetFactory;
import com.google.devtools.build.lib.view.PrerequisiteMap;
import com.google.devtools.build.lib.view.ViewCreationFailedException;
import com.google.devtools.build.lib.view.WorkspaceStatusArtifacts;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.view.config.BinTools;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder.Environment;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeProgressReceiver;
import com.google.devtools.build.skyframe.UpdateResult;

import java.util.Collection;
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

  private final MutableActionGraph actionGraph;
  private final ConfiguredTargetFactory factory;
  private final ArtifactFactory artifactFactory;
  @Nullable private ErrorEventListener warningListener;
  private final SkyframeExecutor skyframeExecutor;
  private final Runnable legacyDataCleaner;
  private final BinTools binTools;
  private boolean enableAnalysis = false;

  /**
   * Because of change pruning, we keep the set of actions to be unregistered instead of eagerly
   * unregistering them (they might later on be re-validated). This avoids re-registering the wrong
   * shared action in the action graph while keeping the original action in the forward graph.
   *
   * <p>We only unregister the actions in the set in the following situations: <li> After
   * Skyframe analysis update has been executed, the remaining actions in the set are
   * unregistered.
   *
   * <li> When we are about to register a new action in {@code ConfiguredTargetNodeBuilder}. At
   * that moment we know that we are going to throw away the forward graph, so we do not care
   * about preserving the registration order of shared actions anymore.
   */
  private Set<Action> pendingInvalidatedActions = Sets.newConcurrentHashSet();
  private final Object registrationLock = new Object();

  // This hack allows us to connect legacy Blaze with Skyframe by listening to events of Skyframe.
  // TODO(bazel-team): Remove this hack. [skyframe-execution]
  private final NodeProgressReceiver invalidationReceiver =
      new ConfiguredTargetNodeInvalidationReceiver();
  private final Set<NodeKey> evaluatedConfiguredTargets = Sets.newConcurrentHashSet();
  // Used to see if checks of graph consistency need to be done after analysis.
  private volatile boolean someConfiguredTargetEvaluated = false;

  private final ImmutableList<OutputFormatter> outputFormatters;

  // We keep the set of invalidated configuration targets so that we can know if something
  // has been invalidated after graph pruning has been executed.
  private Set<ConfiguredTargetNode> dirtyConfiguredTargets = Sets.newConcurrentHashSet();
  private volatile boolean anyConfiguredTargetDeleted = false;
  // This remains null in a skyframe build.
  private WorkspaceStatusArtifacts workspaceStatusArtifacts = null;

  public SkyframeBuildView(MutableActionGraph actionGraph, ConfiguredTargetFactory factory,
      ArtifactFactory artifactFactory, @Nullable ErrorEventListener warningListener,
      SkyframeExecutor skyframeExecutor, Runnable legacyDataCleaner,
      ImmutableList<OutputFormatter> outputFormatters, BinTools binTools) {
    this.actionGraph = actionGraph;
    this.factory = factory;
    this.artifactFactory = artifactFactory;
    // We never reuse derived artifacts in Skyframe, because we never unnecessarily reanalyze an
    // unchanged target, so the same artifact should rarely be requested twice from the factory.
    this.artifactFactory.clear(/*newReuseDerivedArtifacts=*/false);
    this.warningListener = warningListener;
    this.skyframeExecutor = skyframeExecutor;
    this.legacyDataCleaner = legacyDataCleaner;
    this.outputFormatters = outputFormatters;
    this.binTools = binTools;
    skyframeExecutor.setArtifactFactoryAndBinTools(artifactFactory, binTools);
  }

  public void setWorkspaceStatusArtifacts(WorkspaceStatusArtifacts buildInfoArtifacts) {
    Preconditions.checkState(!skyframeExecutor.skyframeBuild());
    this.workspaceStatusArtifacts = buildInfoArtifacts;
  }

  public void setWarningListener(@Nullable ErrorEventListener warningListener) {
    this.warningListener = warningListener;
  }

  public void resetEvaluatedConfiguredTargetKeysSet() {
    evaluatedConfiguredTargets.clear();
  }

  public Set<NodeKey> getEvaluatedTargetKeys() {
    return ImmutableSet.copyOf(evaluatedConfiguredTargets);
  }

  /**
   * Analyzes the specified targets using Skyframe as the driving framework.
   *
   * @return the configured targets that should be built
   */
  public Collection<ConfiguredTarget> configureTargets(List<LabelAndConfiguration> nodes,
      EventBus eventBus, boolean keepGoing)
  throws InterruptedException, ViewCreationFailedException {
    enableAnalysis(true);
    UpdateResult<ConfiguredTargetNode> result;
    try {
      skyframeExecutor.deleteOldNodes();
      result = skyframeExecutor.configureTargets(nodes, keepGoing);
    } finally {
      enableAnalysis(false);
    }
    // For Skyframe m1, note that we already reported action conflicts during action registration
    // in the legacy action graph.
    ImmutableMap<Action, Exception> badActions = skyframeExecutor.skyframeBuild()
        ? skyframeExecutor.findArtifactConflicts()
        : ImmutableMap.<Action, Exception>of();

    // Filter out all CTs that have a bad action and convert to a list of configured targets. This
    // code ensures that the resulting list of configured targets has the same order as the incoming
    // list of nodes, i.e., that the order is deterministic.
    Collection<ConfiguredTarget> goodCts = Lists.newArrayListWithCapacity(nodes.size());
    for (LabelAndConfiguration node : nodes) {
      ConfiguredTargetNode ctNode = result.get(ConfiguredTargetNode.key(node));
      if (ctNode == null) {
        continue;
      }
      goodCts.add(ctNode.getConfiguredTarget());
    }

    if (!result.hasError() && badActions.isEmpty()) {
      return goodCts;
    }

    // --nokeep_going so we fail with an exception for the first error.
    // TODO(bazel-team): We might want to report the other errors through the event bus but
    // for keeping this code in parity with legacy we just report the first error for now.
    if (!keepGoing) {
      for (Map.Entry<Action, Exception> bad : badActions.entrySet()) {
        Exception ex = bad.getValue();
        if (ex instanceof MutableActionGraph.ActionConflictException) {
          MutableActionGraph.ActionConflictException ace =
              (MutableActionGraph.ActionConflictException) ex;
          ace.reportTo(skyframeExecutor.getReporter());
          String errorMsg = "Analysis of target '" + bad.getKey().getOwner().getLabel()
              + "' failed; build aborted";
          throw new ViewCreationFailedException(errorMsg);
        } else {
          skyframeExecutor.getReporter().error(null, ex.getMessage());
        }
        throw new ViewCreationFailedException(ex.getMessage());
      }

      Map.Entry<NodeKey, ErrorInfo> error = result.errorMap().entrySet().iterator().next();
      NodeKey topLevel = error.getKey();
      ErrorInfo errorInfo = error.getValue();
      assertSaneAnalysisError(errorInfo);
      skyframeExecutor.getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), topLevel,
          warningListener);
      Throwable cause = errorInfo.getException();
      Preconditions.checkState(cause != null || !Iterables.isEmpty(errorInfo.getCycleInfo()),
          errorInfo);
      String errorMsg = "Analysis of target '" + ConfiguredTargetNode.extractLabel(topLevel)
          + "' failed; build aborted";
      throw new ViewCreationFailedException(errorMsg);
    }

    // --keep_going : We notify the error and return a ConfiguredTargetNode
    for (Map.Entry<NodeKey, ErrorInfo> errorEntry : result.errorMap().entrySet()) {
      if (nodes.contains(errorEntry.getKey().getNodeName())) {
        NodeKey errorKey = errorEntry.getKey();
        LabelAndConfiguration label = (LabelAndConfiguration) errorKey.getNodeName();
        ErrorInfo errorInfo = errorEntry.getValue();
        assertSaneAnalysisError(errorInfo);

        skyframeExecutor.getCyclesReporter().reportCycles(errorInfo.getCycleInfo(), errorKey,
            warningListener);
        // We try to get the root cause key first from ErrorInfo rootCauses. If we don't have one
        // we try to use the cycle culprit if the error is a cycle. Otherwise we use the top-level
        // error key.
        Label root;
        if (!Iterables.isEmpty(errorEntry.getValue().getRootCauses())) {
          NodeKey culprit = Preconditions.checkNotNull(Iterables.getFirst(
              errorEntry.getValue().getRootCauses(), null));
          root = ((LabelAndConfiguration) culprit.getNodeName()).getLabel();
        } else {
          root = maybeGetConfiguredTargetCycleCulprit(errorInfo.getCycleInfo());
        }
        if (warningListener != null) {
          warningListener.warn(null, "errors encountered while analyzing target '"
              + label + "': it will not be built");
        }
        eventBus.post(new AnalysisFailureEvent(label, root));
      }
    }

    Collection<Exception> reportedExceptions = Sets.newHashSet();
    for (Map.Entry<Action, Exception> bad : badActions.entrySet()) {
      Exception ex = bad.getValue();
      if (ex instanceof MutableActionGraph.ActionConflictException) {
        MutableActionGraph.ActionConflictException ace =
            (MutableActionGraph.ActionConflictException) ex;
        ace.reportTo(skyframeExecutor.getReporter());
        if (warningListener != null) {
          warningListener.warn(null, "errors encountered while analyzing target '"
              + bad.getKey().getOwner().getLabel() + "': it will not be built");
        }
      } else {
        if (reportedExceptions.add(ex)) {
          skyframeExecutor.getReporter().error(null, ex.getMessage());
        }
      }
    }

    if (!badActions.isEmpty()) {
      // In order to determine the set of configured targets transitively error free from action
      // conflict issues, we run a post-processing update() that uses the bad action map.
      UpdateResult<PostConfiguredTargetNode> actionConflictResult =
          skyframeExecutor.postConfigureTargets(nodes, keepGoing, badActions);

      goodCts = Lists.newArrayListWithCapacity(nodes.size());
      for (LabelAndConfiguration node : nodes) {
        PostConfiguredTargetNode postCt =
            actionConflictResult.get(PostConfiguredTargetNode.key(node));
        if (postCt != null) {
          goodCts.add(postCt.getCt());
        }
      }
    }
    return goodCts;
  }

  @Nullable
  Label maybeGetConfiguredTargetCycleCulprit(Iterable<CycleInfo> cycleInfos) {
    for (CycleInfo cycleInfo : cycleInfos) {
      NodeKey culprit = Iterables.getFirst(cycleInfo.getCycle(), null);
      if (culprit == null) {
        continue;
      }
      if (culprit.getNodeType().equals(NodeTypes.CONFIGURED_TARGET)) {
        return ((LabelAndConfiguration) culprit.getNodeName()).getLabel();
      }
    }
    return null;
  }

  private static void assertSaneAnalysisError(ErrorInfo errorInfo) {
    Throwable cause = errorInfo.getException();
    if (cause != null) {
      // We should only be trying to configure targets when the loading phase succeeds, meaning
      // that the only errors should be analysis errors.
      Preconditions.checkState(cause instanceof ConfiguredNodeCreationException, errorInfo);
    }
  }

  MutableActionGraph getActionGraph() {
    return actionGraph;
  }

  ArtifactFactory getArtifactFactory() {
    return artifactFactory;
  }

  @Nullable
  ErrorEventListener getWarningListener() {
    return warningListener;
  }

  /**
   * Because we don't know what build-info artifacts this configured target may request, we
   * conservatively register a dep on all of them.
   */
  // TODO(bazel-team): Allow analysis to return null so the node builder can exit and wait for a
  // restart deps are not present.
  private boolean getWorkspaceStatusNodes(Environment env) {
    env.getDep(WorkspaceStatusNode.NODE_KEY);
    Map<BuildInfoKey, BuildInfoFactory> buildInfoFactories =
        BuildVariableNode.BUILD_INFO_FACTORIES.get(env);
    if (buildInfoFactories == null) {
      return false;
    }
    BuildConfigurationCollection configurations = getBuildConfigurationCollection(env);
    if (configurations == null) {
      return false;
    }
    // These factories may each create their own build info artifacts, all depending on the basic
    // build-info.txt and build-changelist.txt.
    List<NodeKey> depKeys = Lists.newArrayList();
    for (BuildInfoKey key : buildInfoFactories.keySet()) {
      for (BuildConfiguration config : configurations.getAllConfigurations()) {
        depKeys.add(BuildInfoCollectionNode.key(new BuildInfoKeyAndConfig(key, config)));
      }
    }
    env.getDeps(depKeys);
    return !env.depsMissing();
  }

  /** Returns null if any build-info nodes are not ready. */
  @Nullable
  CachingAnalysisEnvironment createAnalysisEnvironment(LabelAndConfiguration owner,
      boolean isSystemEnv, boolean extendedSanityChecks, ErrorEventListener listener,
      Environment env, boolean allowRegisteringActions) {
    if (skyframeExecutor.skyframeBuild() && !getWorkspaceStatusNodes(env)) {
      return null;
    }
    return new CachingAnalysisEnvironment(
        artifactFactory, owner, workspaceStatusArtifacts, isSystemEnv,
        extendedSanityChecks, listener, env, allowRegisteringActions, outputFormatters, binTools);
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   *
   * <p>For use in {@code ConfiguredTargetNodeBuilder}.
   *
   * <p>Returns null if Skyframe deps are missing or upon certain errors.
   */
  @Nullable
  ConfiguredTarget createAndInitialize(Target target, BuildConfiguration configuration,
      CachingAnalysisEnvironment analysisEnvironment, PrerequisiteMap prerequisiteMap,
      ListMultimap<Attribute, Label> labelMap, Environment skyframeEnv)
      throws InterruptedException {
    Preconditions.checkState(enableAnalysis,
        "Already in execution phase %s %s", target, configuration);
    LoadedPackageProvider targetProvider = new TargetProvider(skyframeEnv);
    return factory.createAndInitialize(analysisEnvironment,
        artifactFactory, targetProvider, target, configuration,
        prerequisiteMap, labelMap);
  }

  @Nullable
  private BuildConfigurationCollection getBuildConfigurationCollection(Environment env) {
    ConfigurationCollectionNode configurationsNode =
        (ConfigurationCollectionNode) env.getDep(ConfigurationCollectionNode.CONFIGURATION_KEY);
    return configurationsNode == null ? null : configurationsNode.getConfigurationCollection();
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
   * Hack to invalidate actions in legacy action graph when their nodes are invalidated in skyframe.
   */
  NodeProgressReceiver getInvalidationReceiver() {
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
   * Called from SkyframeExecutor after the graph is checked for artifact conflicts so that the next
   * time {@link #isSomeConfiguredTargetEvaluated} is called, it will return true only if some
   * configured target has been evaluated since the last check for artifact conflicts.
   */
  void resetEvaluatedConfiguredTargetFlag() {
    someConfiguredTargetEvaluated = false;
  }

  /**
   * {@link #createAndInitialize} will only create configured targets if this is set to true. It
   * should be set to true before any Skyframe update call that might call into {@link
   * #createAndInitialize}, and false immediately after the call. Use it to fail-fast in the case
   * that a target is requested for analysis not during the analysis phase.
   */
  void enableAnalysis(boolean enable) {
    this.enableAnalysis = enable;
  }

  private static class TargetProvider implements LoadedPackageProvider {

    private final Environment skyframeEnv;

    public TargetProvider(Environment skyframeEnv) {
      this.skyframeEnv = skyframeEnv;
    }

    @Override
    public Target getLoadedTarget(Label label)
        throws NoSuchPackageException, NoSuchTargetException {
      NodeKey key = PackageNode.key(label.getPackageFragment());
      Node node = skyframeEnv.getDep(key);
      if (node == null) {
        throw new PackageNotInCacheException(label.getPackageName());
      }
      PackageNode packageNode = (PackageNode) node;
      return packageNode.getPackage().getTarget(label.getName());
    }

    @Override
    public Package getLoadedPackage(String packageName) throws NoSuchPackageException {
      // Not used for fetching prerequisites.
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isTargetCurrent(Target target) {
      // Not used for fetching prerequisites.
      throw new UnsupportedOperationException();
    }

    @Override
    public void addDependency(Package pkg, String fileName) {}
  }

  /**
   * Execute the un-registration of all the invalidated actions. It is correct to unregister
   * unrelated actions because on re-validation we register again the action if not found in
   * pendingInvalidatedActions. At this point it does not matter that we register a different
   * shared action since we are going to recreate the forward graph.
   */
  void unregisterPendingActions() {
    synchronized (registrationLock) {
      if (pendingInvalidatedActions.isEmpty()) {
        return;
      }
      if (actionGraph != null) {
        for (Action action : pendingInvalidatedActions) {
          actionGraph.unregisterAction(action);
        }
      }
      pendingInvalidatedActions.clear();
    }
  }

  /**
   * Unregister all pending actions from the action graph and shrink the set of invalidated actions
   * so that further clear() calls are less expensive.
   *
   * <p>Note that this method should not be called when other threads are updating the Skyframe
   * graph since we are creating a new instance of pendingInvalidatedActions and we also
   * synchronize on that object.
   */
  @ThreadHostile
  public void unregisterPendingActionsAndShrink() {
    unregisterPendingActions();
    pendingInvalidatedActions = Sets.newConcurrentHashSet();
  }

  private class ConfiguredTargetNodeInvalidationReceiver implements NodeProgressReceiver {
    @Override
    public void invalidated(Node node, InvalidationState state) {
      if (node instanceof ConfiguredTargetNode) {
        ConfiguredTargetNode ctNode = (ConfiguredTargetNode) node;
        if (state == InvalidationState.DIRTY || state == InvalidationState.DELETED) {
          // If the node was just dirtied and not deleted, then it may not be truly invalid, since
          // it may later get re-validated.
          if (state == InvalidationState.DELETED) {
            anyConfiguredTargetDeleted = true;
          } else {
            dirtyConfiguredTargets.add(ctNode);
          }
          for (Action action : ctNode.getActions()) {
            // Delay the invalidation until a new configured target is created (forward graph will
            // be invalidated) or after the analysis update is executed. This does not need to
            // use registrationLock because all invalidations happen before all the other
            // code paths.
            pendingInvalidatedActions.add(action);
          }
        }
      }
    }

    @Override
    public void enqueueing(NodeKey nodeKey) {}

    @Override
    public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
      if (nodeKey.getNodeType() == NodeTypes.CONFIGURED_TARGET) {
        if (state == EvaluationState.BUILT) {
          evaluatedConfiguredTargets.add(nodeKey);
          // During multithreaded operation, this is only set to true, so no concurrency issues.
          someConfiguredTargetEvaluated = true;
        }
        Preconditions.checkNotNull(node, "%s %s", nodeKey, state);
        ConfiguredTargetNode ctNode = (ConfiguredTargetNode) node;
        dirtyConfiguredTargets.remove(ctNode);
        for (Action action : ctNode.getActions()) {
          // If we are not present in pendingInvalidatedActions that means that we cannot assume
          // that we are already registered in the action graph.
          // We need to synchronize because the removal and registration has to be atomic and
          // exclusive with unregisterPendingActions code.
          boolean removed;
          synchronized (registrationLock) {
            removed = pendingInvalidatedActions.remove(action);
          }
          if (!removed && actionGraph != null) {
            actionGraph.registerAction(action);
          }
        }
      }
    }
  }
}
