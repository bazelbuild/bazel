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
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.events.StoredErrorEventListener;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.LateBoundAttributeHelper;
import com.google.devtools.build.lib.view.PrerequisiteMap;
import com.google.devtools.build.lib.view.TargetAndConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeOrException;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * NodeBuilder for {@link ConfiguredTargetNode}s.
 */
final class ConfiguredTargetNodeBuilder implements NodeBuilder {

  private final BuildViewProvider buildViewProvider;

  ConfiguredTargetNodeBuilder(BuildViewProvider buildViewProvider) {
    this.buildViewProvider = buildViewProvider;
  }

  @Override
  public Node build(NodeKey key, Environment env) throws ConfiguredTargetNodeBuilderException,
      InterruptedException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();

    LabelAndConfiguration lc = (LabelAndConfiguration) key.getNodeName();

    BuildConfiguration configuration = lc.getConfiguration();

    NodeKey packageNodeKey = PackageNode.key(lc.getLabel().getPackageFragment());
    PackageNode packageNode = (PackageNode) env.getDep(packageNodeKey);
    if (packageNode == null) {
      return null;
    }

    Target target;
    try {
      target = packageNode.getPackage().getTarget(lc.getLabel().getName());
    } catch (NoSuchTargetException e1) {
      throw new ConfiguredTargetNodeBuilderException(packageNodeKey,
          new NoSuchTargetException(lc.getLabel(), "No such target"));
    }
    // TODO(bazel-team): This is problematic - we create the right key, but then end up with a node
    // that doesn't match; we can even have the same node multiple times. However, I think it's only
    // triggered in tests (i.e., in normal operation, the configuration passed in is already null).
    if (target instanceof InputFile) {
      // InputFileConfiguredTarget expects its configuration to be null since it's not used.
      configuration = null;
    } else if (target instanceof PackageGroup) {
      // Same for PackageGroupConfiguredTarget.
      configuration = null;
    }
    TargetAndConfiguration ctgNode =
        new TargetAndConfiguration(target, configuration);

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);
    if (resolver == null) {
      return null;
    }

    ListMultimap<Attribute, Label> labelMap = null;
    if (target instanceof Rule) {
      labelMap = new LateBoundAttributeHelper((Rule) target, configuration).createAttributeMap();
    }
    Collection<TargetAndConfiguration> depNodeNames =
        resolver.dependentNodes(ctgNode, labelMap);

    boolean ok = !env.depsMissing();
    List<ConfiguredTargetNode> depNodes = Lists.newArrayListWithCapacity(depNodeNames.size());
    String message = null;
    List<NodeKey> depKeys = Lists.newArrayList();
    for (TargetAndConfiguration depNodeName : depNodeNames) {
      Label depLabel = depNodeName.getLabel();
      depKeys.add(ConfiguredTargetNode.key(depLabel, depNodeName.getConfiguration()));
    }
    // TODO(bazel-team): maybe having a two-exception argument is better than typing a generic
    // Exception here.
    for (Map.Entry<NodeKey, NodeOrException<Exception>> entry :
        env.getDepsOrThrow(depKeys, Exception.class).entrySet()) {
      LabelAndConfiguration depLabelAndConfiguration =
          (LabelAndConfiguration) entry.getKey().getNodeName();
      Label depLabel = depLabelAndConfiguration.getLabel();
      ConfiguredTargetNode depNode = null;
      NoSuchThingException directChildException = null;
      try {
        depNode = (ConfiguredTargetNode) entry.getValue().get();
      } catch (NoSuchTargetException e) {
        if (depLabel.equals(e.getLabel())) {
          directChildException = e;
        }
      } catch (NoSuchPackageException e) {
        if (depLabel.getPackageName().equals(e.getPackageName())) {
          directChildException = e;
        }
      } catch (ConfiguredNodeCreationException e) {
        // Do nothing.
      } catch (Exception e) {
        throw new IllegalStateException("Not NoSuchTargetException or NoSuchPackageException"
            + " or ViewCreationFailedException: " + e.getMessage(), e);
      }
      // If an exception wasn't caused by a direct child target node, we'll treat it the same
      // as any other missing dep by setting ok = false below, and returning null at the end.
      if (directChildException != null) {
        // Only update messages for missing targets we depend on directly.
        message = TargetUtils.formatMissingEdge(target, depLabel, directChildException);
        env.getListener().error(TargetUtils.getLocationMaybe(target), message);
      }

      if (depNode == null) {
        ok = false;
      } else {
        depNodes.add(depNode);
      }
    }
    if (message != null) {
      throw new ConfiguredTargetNodeBuilderException(packageNodeKey,
          new NoSuchTargetException(message));
    }
    if (!ok) {
      return null;
    }

    return createConfiguredTarget(view, env, target, configuration, depNodes, labelMap);
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return Label.print(((LabelAndConfiguration) nodeKey.getNodeName()).getLabel());
  }

  @Nullable
  private ConfiguredTargetNode createConfiguredTarget(SkyframeBuildView view,
      Environment env, Target target, BuildConfiguration configuration,
      List<ConfiguredTargetNode> depNodes, ListMultimap<Attribute, Label> labelMap)
      throws ConfiguredTargetNodeBuilderException,
      InterruptedException {
    boolean extendedSanityChecks = configuration != null && configuration.extendedSanityChecks();

    StoredErrorEventListener events = new StoredErrorEventListener();
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        new LabelAndConfiguration(target.getLabel(), configuration), false,
        extendedSanityChecks, events, env);
    if (env.depsMissing()) {
      return null;
    }
    PrerequisiteMap.Builder prerequisiteMap = new PrerequisiteMap.Builder(extendedSanityChecks);
    for (ConfiguredTargetNode prerequisiteNode : depNodes) {
      ConfiguredTarget prerequisite = prerequisiteNode.getConfiguredTarget();
      prerequisiteMap.add(prerequisite);
    }

    ConfiguredTarget configuredTarget = view.createAndInitialize(
        target, configuration, analysisEnvironment, prerequisiteMap.build(), labelMap, env);
    if (env.depsMissing()) {
      return null;
    }

    Collection<Action> actions = ImmutableList.copyOf(analysisEnvironment.getRegisteredActions());

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(target);
      throw new ConfiguredTargetNodeBuilderException(ConfiguredTargetNode.key(target.getLabel(),
          configuration), new ConfiguredNodeCreationException(
              "Analysis of target '" + target.getLabel() + "' failed; build aborted"));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");
    Preconditions.checkNotNull(configuredTarget);
    analysisEnvironment.disable(target);

    // Record actions and check duplicates.
    // It's a bit awkward that non-ActionOwner configured targets can have actions, but that's
    // how BUILD file analysis works right now.
    Collection<Action> registeredActions = Lists.newArrayListWithCapacity(actions.size());
    for (Action action : actions) {
      // TODO(bazel-team): Drop construction of the legacy action graph. [skyframe-execution]
      try {
        // Invalidate all pending actions before proceeding. This is needed because we could
        // have pending invalidated actions that would conflict with this registration. We delay
        // the unregistration until we try to create a new action to avoid a shared actions issue
        // (see SkyframeBuildView.pendingInvalidatedActions).
        view.unregisterPendingActions();
        view.getActionGraph().registerAction(action);
      } catch (ActionConflictException e) {
        e.reportTo(env.getListener());
        // Unregister all actions registered before to keep the legacy action graph in sync.
        for (Action a : registeredActions) {
          view.getActionGraph().unregisterAction(a);
        }
        throw new ConfiguredTargetNodeBuilderException(ConfiguredTargetNode.key(target.getLabel(),
            configuration), new ConfiguredNodeCreationException(
                "Analysis of target '" + target.getLabel() + "' failed; build aborted"));
      }
      registeredActions.add(action);
    }
    return new ConfiguredTargetNode(configuredTarget, actions);
  }

  /**
   * An exception indicating that there was a problem during the construction of
   * a ConfiguredTargetNode.
   */
  public static final class ConfiguredNodeCreationException extends Exception {

    public ConfiguredNodeCreationException(String message) {
      super(message);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfiguredTargetNodeBuilder#build}.
   */
  private static final class ConfiguredTargetNodeBuilderException extends NodeBuilderException {
    public ConfiguredTargetNodeBuilderException(NodeKey key, NoSuchTargetException e) {
      super(key, e);
    }

    public ConfiguredTargetNodeBuilderException(NodeKey key, ConfiguredNodeCreationException e) {
      super(key, e);
    }
  }
}
