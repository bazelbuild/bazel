// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.actiongraph;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecutionInfoSpecifier;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Encapsulates necessary functionality to dump the current skyframe state of the action graph to
 * proto format.
 */
public class ActionGraphDump {

  private final ActionGraphContainer.Builder actionGraphBuilder = ActionGraphContainer.newBuilder();
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();
  private final Set<String> actionGraphTargets;

  private final KnownRuleClassStrings knownRuleClassStrings;
  private final KnownArtifacts knownArtifacts;
  private final KnownConfigurations knownConfigurations;
  private final KnownNestedSets knownNestedSets;
  private final KnownAspectDescriptors knownAspectDescriptors;
  private final KnownRuleConfiguredTargets knownRuleConfiguredTargets;
  private final boolean includeActionCmdLine;

  public ActionGraphDump(boolean includeActionCmdLine) {
    this(/* actionGraphTargets= */ ImmutableList.of("..."), includeActionCmdLine);
  }

  public ActionGraphDump(List<String> actionGraphTargets, boolean includeActionCmdLine) {
    this.actionGraphTargets = ImmutableSet.copyOf(actionGraphTargets);
    this.includeActionCmdLine = includeActionCmdLine;

    knownRuleClassStrings = new KnownRuleClassStrings(actionGraphBuilder);
    knownArtifacts = new KnownArtifacts(actionGraphBuilder);
    knownConfigurations = new KnownConfigurations(actionGraphBuilder);
    knownNestedSets = new KnownNestedSets(actionGraphBuilder, knownArtifacts);
    knownAspectDescriptors = new KnownAspectDescriptors(actionGraphBuilder);
    knownRuleConfiguredTargets = new KnownRuleConfiguredTargets(actionGraphBuilder,
        knownRuleClassStrings);
  }

  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
  }

  private boolean includeInActionGraph(String labelString) {
    if (actionGraphTargets.size() == 1
        && Iterables.getOnlyElement(actionGraphTargets).equals("...")) {
      return true;
    }
    return actionGraphTargets.contains(labelString);
  }

  private void dumpSingleAction(ConfiguredTarget configuredTarget, ActionAnalysisMetadata action)
      throws CommandLineExpansionException {
    Preconditions.checkState(configuredTarget instanceof RuleConfiguredTarget);
    RuleConfiguredTarget ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
    AnalysisProtos.Action.Builder actionBuilder =
        AnalysisProtos.Action.newBuilder()
            .setMnemonic(action.getMnemonic())
            .setTargetId(knownRuleConfiguredTargets.dataToId(ruleConfiguredTarget));

    if (action instanceof ActionExecutionMetadata) {
      ActionExecutionMetadata actionExecutionMetadata = (ActionExecutionMetadata) action;
      actionBuilder
          .setActionKey(actionExecutionMetadata.getKey(getActionKeyContext()))
          .setDiscoversInputs(actionExecutionMetadata.discoversInputs());
    }

    // store environment
    if (action instanceof SpawnAction) {
      SpawnAction spawnAction = (SpawnAction) action;
      // TODO(twerth): This handles the fixed environment. We probably want to output the inherited
      // environment as well.
      Map<String, String> fixedEnvironment = spawnAction.getEnvironment().getFixedEnv().toMap();
      for (Map.Entry<String, String> environmentVariable : fixedEnvironment.entrySet()) {
        AnalysisProtos.KeyValuePair.Builder keyValuePairBuilder =
            AnalysisProtos.KeyValuePair.newBuilder();
        keyValuePairBuilder
            .setKey(environmentVariable.getKey())
            .setValue(environmentVariable.getValue());
        actionBuilder.addEnvironmentVariables(keyValuePairBuilder.build());
      }

      if (includeActionCmdLine) {
        actionBuilder.addAllArguments(spawnAction.getArguments());
      }
    }

    if (action instanceof ExecutionInfoSpecifier) {
      ExecutionInfoSpecifier executionInfoSpecifier = (ExecutionInfoSpecifier) action;
      for (Map.Entry<String, String> info : executionInfoSpecifier.getExecutionInfo().entrySet()) {
        actionBuilder.addExecutionInfo(
            AnalysisProtos.KeyValuePair.newBuilder()
                .setKey(info.getKey())
                .setValue(info.getValue()));
      }
    }

    ActionOwner actionOwner = action.getOwner();
    if (actionOwner != null) {
      BuildEvent event = actionOwner.getConfiguration();
      actionBuilder.setConfigurationId(knownConfigurations.dataToId(event));

      // store aspect
      for (AspectDescriptor aspectDescriptor : actionOwner.getAspectDescriptors()) {
        actionBuilder.addAspectDescriptorIds(knownAspectDescriptors.dataToId(aspectDescriptor));
      }
    }

    // store inputs
    Iterable<Artifact> inputs = action.getInputs();
    if (!(inputs instanceof NestedSet)) {
      inputs = NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs);
    }
    NestedSetView<Artifact> nestedSetView = new NestedSetView<>((NestedSet<Artifact>) inputs);
    if (nestedSetView.directs().size() > 0 || nestedSetView.transitives().size() > 0) {
      actionBuilder.addInputDepSetIds(knownNestedSets.dataToId(nestedSetView));
    }

    // store outputs
    for (Artifact artifact : action.getOutputs()) {
      actionBuilder.addOutputIds(knownArtifacts.dataToId(artifact));
    }

    actionGraphBuilder.addActions(actionBuilder.build());
  }

  public void dumpAspect(AspectValue aspectValue, ConfiguredTargetValue configuredTargetValue)
      throws CommandLineExpansionException {
    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    if (!includeInActionGraph(configuredTarget.getLabel().toString())) {
      return;
    }
    for (int i = 0; i < aspectValue.getNumActions(); i++) {
      Action action = aspectValue.getAction(i);
      dumpSingleAction(configuredTarget, action);
    }
  }

  public void dumpConfiguredTarget(ConfiguredTargetValue configuredTargetValue)
      throws CommandLineExpansionException {
    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    if (!includeInActionGraph(configuredTarget.getLabel().toString())) {
      return;
    }
    List<ActionAnalysisMetadata> actions = configuredTargetValue.getActions();
    for (ActionAnalysisMetadata action : actions) {
      dumpSingleAction(configuredTarget, action);
    }
  }

  public ActionGraphContainer build() {
    return actionGraphBuilder.build();
  }
}
