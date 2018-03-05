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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtos.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtos.KeyValuePair;
import com.google.devtools.build.lib.analysis.AnalysisProtos.RuleClass;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Target;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Encapsulates necessary functionality to dump the current skyframe state of the action graph to
 * proto format.
 */
public class ActionGraphDump {

  private final Map<Artifact, String> knownArtifacts = new HashMap<>();
  private final Map<BuildConfiguration, String> knownConfigurations = new HashMap<>();
  private final Map<Label, String> knownTargets = new HashMap<>();
  private final Map<AspectDescriptor, String> knownAspectDescriptors = new HashMap<>();
  private final Map<String, String> knownRuleClassStrings = new HashMap<>();
  // The NestedSet is identified by their raw 'children' object since multiple NestedSetViews
  // can point to the same object.
  private final Map<Object, String> knownNestedSets = new HashMap<>();

  private final ActionGraphContainer.Builder actionGraphBuilder = ActionGraphContainer.newBuilder();
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();
  private final Set<String> actionGraphTargets;

  public ActionGraphDump(List<String> actionGraphTargets) {
    this.actionGraphTargets = ImmutableSet.copyOf(actionGraphTargets);
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

  private String ruleClassStringToId(String ruleClassString) {
    if (!knownRuleClassStrings.containsKey(ruleClassString)) {
      String targetId = String.valueOf(knownRuleClassStrings.size());
      knownRuleClassStrings.put(ruleClassString, targetId);
      RuleClass.Builder ruleClassBuilder =
          RuleClass.newBuilder().setId(targetId).setName(ruleClassString);
      actionGraphBuilder.addRuleClasses(ruleClassBuilder.build());
    }
    return knownRuleClassStrings.get(ruleClassString);
  }

  private String targetToId(Label label, String ruleClassString) {
    if (!knownTargets.containsKey(label)) {
      String targetId = String.valueOf(knownTargets.size());
      knownTargets.put(label, targetId);
      Target.Builder targetBuilder = Target.newBuilder();
      targetBuilder.setId(targetId).setLabel(label.toString());
      if (ruleClassString != null) {
        targetBuilder.setRuleClassId(ruleClassStringToId(ruleClassString));
      }
      actionGraphBuilder.addTargets(targetBuilder.build());
    }
    return knownTargets.get(label);
  }

  private String configurationToId(BuildConfiguration buildConfiguration) {
    if (!knownConfigurations.containsKey(buildConfiguration)) {
      String configurationId = String.valueOf(knownConfigurations.size());
      knownConfigurations.put(buildConfiguration, configurationId);
      Configuration configurationProto =
          Configuration.newBuilder()
              .setMnemonic(buildConfiguration.getMnemonic())
              .setPlatformName(buildConfiguration.getPlatformName())
              .setId(configurationId)
              .build();
      actionGraphBuilder.addConfiguration(configurationProto);
    }
    return knownConfigurations.get(buildConfiguration);
  }

  private String artifactToId(Artifact artifact) {
    if (!knownArtifacts.containsKey(artifact)) {
      String artifactId = String.valueOf(knownArtifacts.size());
      knownArtifacts.put(artifact, artifactId);
      AnalysisProtos.Artifact artifactProto =
          AnalysisProtos.Artifact.newBuilder()
              .setId(artifactId)
              .setExecPath(artifact.getExecPathString())
              .setIsTreeArtifact(artifact.isTreeArtifact())
              .build();
      actionGraphBuilder.addArtifacts(artifactProto);
    }
    return knownArtifacts.get(artifact);
  }

  private String depSetToId(NestedSetView<Artifact> nestedSetView) {
    if (!knownNestedSets.containsKey(nestedSetView.identifier())) {
      String nestedSetId = String.valueOf(knownNestedSets.size());
      knownNestedSets.put(nestedSetView.identifier(), nestedSetId);
      DepSetOfFiles.Builder depSetBuilder = DepSetOfFiles.newBuilder().setId(nestedSetId);
      for (NestedSetView<Artifact> transitiveNestedSet : nestedSetView.transitives()) {
        depSetBuilder.addTransitiveDepSetIds(depSetToId(transitiveNestedSet));
      }
      for (Artifact directArtifact : nestedSetView.directs()) {
        depSetBuilder.addDirectArtifactIds(artifactToId(directArtifact));
      }
      actionGraphBuilder.addDepSetOfFiles(depSetBuilder.build());
    }
    return knownNestedSets.get(nestedSetView.identifier());
  }

  private String aspectDescriptorToId(AspectDescriptor aspectDescriptor) {
    if (!knownAspectDescriptors.containsKey(aspectDescriptor)) {
      String aspectDescriptorId = String.valueOf(knownAspectDescriptors.size());
      knownAspectDescriptors.put(aspectDescriptor, aspectDescriptorId);
      AnalysisProtos.AspectDescriptor.Builder aspectDescriptorBuilder =
          AnalysisProtos.AspectDescriptor.newBuilder()
              .setId(aspectDescriptorId)
              .setName(aspectDescriptor.getAspectClass().getName());
      for (Entry<String, String> parameter :
          aspectDescriptor.getParameters().getAttributes().entries()) {
        KeyValuePair.Builder keyValuePairBuilder = KeyValuePair.newBuilder();
        keyValuePairBuilder.setKey(parameter.getKey()).setValue(parameter.getValue());
        aspectDescriptorBuilder.addParameters(keyValuePairBuilder.build());
      }
      actionGraphBuilder.addAspectDescriptors(aspectDescriptorBuilder.build());
    }
    return knownAspectDescriptors.get(aspectDescriptor);
  }

  private void dumpSingleAction(ConfiguredTarget configuredTarget, ActionAnalysisMetadata action) {
    Preconditions.checkState(configuredTarget instanceof RuleConfiguredTarget);
    Label label = configuredTarget.getLabel();
    String ruleClassString = ((RuleConfiguredTarget) configuredTarget).getRuleClassString();
    AnalysisProtos.Action.Builder actionBuilder =
        AnalysisProtos.Action.newBuilder()
            .setMnemonic(action.getMnemonic())
            .setTargetId(targetToId(label, ruleClassString));

    if (action instanceof ActionExecutionMetadata) {
      ActionExecutionMetadata actionExecutionMetadata = (ActionExecutionMetadata) action;
      actionBuilder
          .setActionKey(actionExecutionMetadata.getKey(getActionKeyContext()))
          .setDiscoversInputs(actionExecutionMetadata.discoversInputs());
    }

    // store environment
    if (action instanceof SpawnAction) {
      SpawnAction spawnAction = (SpawnAction) action;
      // TODO(twerth): This handles the fixed environemnt. We probably want to output the inherited
      // environment as well.
      ImmutableMap<String, String> fixedEnvironment = spawnAction.getEnvironment();
      for (Entry<String, String> environmentVariable : fixedEnvironment.entrySet()) {
        AnalysisProtos.KeyValuePair.Builder keyValuePairBuilder =
            AnalysisProtos.KeyValuePair.newBuilder();
        keyValuePairBuilder
            .setKey(environmentVariable.getKey())
            .setValue(environmentVariable.getValue());
        actionBuilder.addEnvironmentVariables(keyValuePairBuilder.build());
      }
    }

    ActionOwner actionOwner = action.getOwner();
    if (actionOwner != null) {
      BuildConfiguration buildConfiguration = (BuildConfiguration) actionOwner.getConfiguration();
      actionBuilder.setConfigurationId(configurationToId(buildConfiguration));

      // store aspect
      for (AspectDescriptor aspectDescriptor : actionOwner.getAspectDescriptors()) {
        actionBuilder.addAspectDescriptorIds(aspectDescriptorToId(aspectDescriptor));
      }
    }

    // store inputs
    Iterable<Artifact> inputs = action.getInputs();
    if (!(inputs instanceof NestedSet)) {
      inputs = NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs);
    }
    NestedSetView<Artifact> nestedSetView = new NestedSetView<>((NestedSet<Artifact>) inputs);
    if (nestedSetView.directs().size() > 0 || nestedSetView.transitives().size() > 0) {
      actionBuilder.addInputDepSetIds(depSetToId(nestedSetView));
    }

    // store outputs
    for (Artifact artifact : action.getOutputs()) {
      actionBuilder.addOutputIds(artifactToId(artifact));
    }

    actionGraphBuilder.addActions(actionBuilder.build());
  }

  public void dumpAspect(AspectValue aspectValue, ConfiguredTargetValue configuredTargetValue) {
    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    if (!includeInActionGraph(configuredTarget.getLabel().toString())) {
      return;
    }
    for (int i = 0; i < aspectValue.getNumActions(); i++) {
      Action action = aspectValue.getAction(i);
      dumpSingleAction(configuredTarget, action);
    }
  }

  public void dumpConfiguredTarget(ConfiguredTargetValue configuredTargetValue) {
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
