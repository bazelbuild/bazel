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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/**
 * A configured target in the context of a Skyframe graph.
 */
@Immutable
@ThreadSafe
final class ConfiguredTargetNode extends ActionLookupNode {

  // These variables are only non-final because they may be clear()ed to save memory. They are null
  // only after they are cleared.
  @Nullable private ConfiguredTarget configuredTarget;
  // We overload this variable to check whether the node has been clear()ed. We don't use a separate
  // variable in order to save memory.
  @Nullable private volatile Iterable<Action> actions;

  ConfiguredTargetNode(ConfiguredTarget configuredTarget, Iterable<Action> actions) {
    super(actions);
    this.configuredTarget = configuredTarget;
    this.actions = actions;
  }

  ConfiguredTarget getConfiguredTarget() {
    Preconditions.checkNotNull(actions, configuredTarget);
    return configuredTarget;
  }

  Iterable<Action> getActions() {
    return Preconditions.checkNotNull(actions, configuredTarget);
  }

  /**
   * Clears configured target data from this node, leaving only the artifact->generating action map.
   * Should only be used when user specifies --discard_analysis_cache. Must be called at most once
   * per node, after which {@link #getConfiguredTarget} and {@link #getActions} cannot be called.
   */
  public void clear() {
    Preconditions.checkNotNull(actions, configuredTarget);
    configuredTarget = null;
    actions = null;
  }

  static NodeKey key(Label label, BuildConfiguration configuration) {
    return key(new LabelAndConfiguration(label, configuration));
  }

  static NodeKey key(ConfiguredTarget ct) {
    return key(ct.getLabel(), ct.getConfiguration());
  }

  static ImmutableList<NodeKey> keys(Iterable<LabelAndConfiguration> lacs) {
    ImmutableList.Builder<NodeKey> keys = ImmutableList.builder();
    for (LabelAndConfiguration lac : lacs) {
      keys.add(key(lac));
    }
    return keys.build();
  }

  /**
   * Returns a label of ConfiguredNodeTarget
   */
  @ThreadSafe
  static Label extractLabel(NodeKey node) {
    Object nodeName = node.getNodeName();
    Preconditions.checkState(nodeName instanceof LabelAndConfiguration, nodeName);
    return ((LabelAndConfiguration) nodeName).getLabel();
  }

  @Override
  public String toString() {
    return "ConfiguredTargetNode: "
        + configuredTarget + ", actions: " + (actions == null ? null : Iterables.toString(actions));
  }
}
