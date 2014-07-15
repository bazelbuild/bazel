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
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.TargetAndConfiguration;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/**
 * Build a post-processed ConfiguredTarget, vetting it for action conflict issues.
 */
public class PostConfiguredTargetNodeBuilder implements NodeBuilder {

  private final SkyframeExecutor.BuildViewProvider buildViewProvider;

  public PostConfiguredTargetNodeBuilder(SkyframeExecutor.BuildViewProvider buildViewProvider) {
    this.buildViewProvider = Preconditions.checkNotNull(buildViewProvider);
  }

  @Nullable
  @Override
  public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException {
    ImmutableMap<Action, Exception> badActions = BuildVariableNode.BAD_ACTIONS.get(env);
    ConfiguredTargetNode ctNode = (ConfiguredTargetNode)
        env.getDep(ConfiguredTargetNode.key((LabelAndConfiguration) nodeKey.getNodeName()));
    SkyframeDependencyResolver resolver =
        buildViewProvider.getSkyframeBuildView().createDependencyResolver(env);
    if (env.depsMissing()) {
      return null;
    }

    for (Action action : ctNode.getActions()) {
      if (badActions.containsKey(action)) {
        throw new ActionConflictNodeBuilderException(nodeKey, badActions.get(action));
      }
    }

    ConfiguredTarget ct = ctNode.getConfiguredTarget();
    TargetAndConfiguration ctgNode =
        new TargetAndConfiguration(ct.getTarget(), ct.getConfiguration());

    for (TargetAndConfiguration dep : resolver.dependentNodes(ctgNode)) {
      env.getDep(PostConfiguredTargetNode.key(
          new LabelAndConfiguration(dep.getLabel(), dep.getConfiguration())));
    }
    if (env.depsMissing()) {
      return null;
    }

    return new PostConfiguredTargetNode(ct);
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return Label.print(((LabelAndConfiguration) nodeKey.getNodeName()).getLabel());
  }

  private static class ActionConflictNodeBuilderException extends NodeBuilderException {
    public ActionConflictNodeBuilderException(NodeKey nodeKey, Throwable cause) {
      super(nodeKey, cause);
    }
  }
}
