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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

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
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.query2.aquery.AqueryActionFilter;
import com.google.devtools.build.lib.query2.aquery.AqueryUtils;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Encapsulates necessary functionality to dump the current skyframe state of the action graph to
 * proto format.
 */
public class ActionGraphDump {

  private final ActionKeyContext actionKeyContext = new ActionKeyContext();
  private final Set<String> actionGraphTargets;

  private final KnownRuleClassStrings knownRuleClassStrings;
  private final KnownArtifacts knownArtifacts;
  private final KnownConfigurations knownConfigurations;
  private final KnownNestedSets knownNestedSets;
  private final KnownAspectDescriptors knownAspectDescriptors;
  private final KnownTargets knownTargets;
  private final AqueryActionFilter actionFilters;
  private final boolean includeActionCmdLine;
  private final boolean includeArtifacts;
  private final boolean includeParamFiles;
  private final AqueryOutputHandler aqueryOutputHandler;

  private Map<String, Iterable<String>> paramFileNameToContentMap;

  public ActionGraphDump(
      boolean includeActionCmdLine,
      boolean includeArtifacts,
      AqueryActionFilter actionFilters,
      boolean includeParamFiles,
      AqueryOutputHandler aqueryOutputHandler) {
    this(
        /* actionGraphTargets= */ ImmutableList.of("..."),
        includeActionCmdLine,
        includeArtifacts,
        actionFilters,
        includeParamFiles,
        aqueryOutputHandler);
  }

  public ActionGraphDump(
      List<String> actionGraphTargets,
      boolean includeActionCmdLine,
      boolean includeArtifacts,
      AqueryActionFilter actionFilters,
      boolean includeParamFiles,
      AqueryOutputHandler aqueryOutputHandler) {
    this.actionGraphTargets = ImmutableSet.copyOf(actionGraphTargets);
    this.includeActionCmdLine = includeActionCmdLine;
    this.includeArtifacts = includeArtifacts;
    this.actionFilters = actionFilters;
    this.includeParamFiles = includeParamFiles;
    this.aqueryOutputHandler = aqueryOutputHandler;

    knownRuleClassStrings = new KnownRuleClassStrings(aqueryOutputHandler);
    knownArtifacts = new KnownArtifacts(aqueryOutputHandler);
    knownConfigurations = new KnownConfigurations(aqueryOutputHandler);
    knownNestedSets = new KnownNestedSets(aqueryOutputHandler, knownArtifacts);
    knownAspectDescriptors = new KnownAspectDescriptors(aqueryOutputHandler);
    knownTargets = new KnownTargets(aqueryOutputHandler, knownRuleClassStrings);
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
      throws CommandLineExpansionException, IOException {

    // Store the content of param files.
    if (includeParamFiles && (action instanceof ParameterFileWriteAction)) {
      ParameterFileWriteAction parameterFileWriteAction = (ParameterFileWriteAction) action;

      Iterable<String> fileContent = parameterFileWriteAction.getArguments();
      String paramFileExecPath = action.getPrimaryOutput().getExecPathString();
      getParamFileNameToContentMap().put(paramFileExecPath, fileContent);
    }

    if (!AqueryUtils.matchesAqueryFilters(action, actionFilters)) {
      return;
    }

    // Dereference any aliases that might be present.
    configuredTarget = configuredTarget.getActual();

    Preconditions.checkState(configuredTarget instanceof RuleConfiguredTarget);
    Pair<String, String> targetIdentifier =
        new Pair<>(
            configuredTarget.getLabel().toString(),
            ((RuleConfiguredTarget) configuredTarget).getRuleClassString());
    AnalysisProtosV2.Action.Builder actionBuilder =
        AnalysisProtosV2.Action.newBuilder()
            .setMnemonic(action.getMnemonic())
            .setTargetId(knownTargets.dataToIdAndStreamOutputProto(targetIdentifier));

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
        actionBuilder.addEnvironmentVariables(
            AnalysisProtosV2.KeyValuePair.newBuilder()
                .setKey(environmentVariable.getKey())
                .setValue(environmentVariable.getValue())
                .build());
      }
    }

    if (includeActionCmdLine && action instanceof CommandAction) {
      CommandAction commandAction = (CommandAction) action;
      actionBuilder.addAllArguments(commandAction.getArguments());
    }

    // Include the content of param files in output.
    if (includeParamFiles) {
      // Assumption: if an Action takes a params file as an input, it will be used
      // to provide params to the command.
      for (Artifact input : action.getInputs().toList()) {
        String inputFileExecPath = input.getExecPathString();
        if (getParamFileNameToContentMap().containsKey(inputFileExecPath)) {
          AnalysisProtosV2.ParamFile paramFile =
              AnalysisProtosV2.ParamFile.newBuilder()
                  .setExecPath(inputFileExecPath)
                  .addAllArguments(getParamFileNameToContentMap().get(inputFileExecPath))
                  .build();
          actionBuilder.addParamFiles(paramFile);
        }
      }
    }
    Map<String, String> executionInfo = action.getExecutionInfo();
    if (executionInfo != null) {
      for (Map.Entry<String, String> info : executionInfo.entrySet()) {
        actionBuilder.addExecutionInfo(
            AnalysisProtosV2.KeyValuePair.newBuilder()
                .setKey(info.getKey())
                .setValue(info.getValue()));
      }
    }

    ActionOwner actionOwner = action.getOwner();
    if (actionOwner != null) {
      BuildEvent event = actionOwner.getConfiguration();
      actionBuilder.setConfigurationId(knownConfigurations.dataToIdAndStreamOutputProto(event));

      // Store aspects.
      // Iterate through the aspect path and dump the aspect descriptors.
      // In the case of aspect-on-aspect, AspectDescriptors are listed in topological order
      // of the configured target graph.
      // e.g. [A, B] would imply that aspect A is applied on top of aspect B.
      for (AspectDescriptor aspectDescriptor : actionOwner.getAspectDescriptors().reverse()) {
        actionBuilder.addAspectDescriptorIds(
            knownAspectDescriptors.dataToIdAndStreamOutputProto(aspectDescriptor));
      }
    }

    if (includeArtifacts) {
      // Store inputs
      NestedSet<Artifact> inputs = action.getInputs();
      NestedSetView<Artifact> nestedSetView = new NestedSetView<>(inputs);

      if (!nestedSetView.directs().isEmpty() || !nestedSetView.transitives().isEmpty()) {
        actionBuilder.addInputDepSetIds(
            knownNestedSets.dataToIdAndStreamOutputProto(nestedSetView));
      }

      // store outputs
      for (Artifact artifact : action.getOutputs()) {
        actionBuilder.addOutputIds(knownArtifacts.dataToIdAndStreamOutputProto(artifact));
      }

      actionBuilder.setPrimaryOutputId(
          knownArtifacts.dataToIdAndStreamOutputProto(action.getPrimaryOutput()));
    }

    aqueryOutputHandler.outputAction(actionBuilder.build());
  }

  public void dumpAspect(AspectValue aspectValue, ConfiguredTargetValue configuredTargetValue)
      throws CommandLineExpansionException, IOException {
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
      throws CommandLineExpansionException, IOException {
    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    if (!includeInActionGraph(configuredTarget.getLabel().toString())) {
      return;
    }
    List<ActionAnalysisMetadata> actions = configuredTargetValue.getActions();
    for (ActionAnalysisMetadata action : actions) {
      dumpSingleAction(configuredTarget, action);
    }
  }

  /** Lazy initialization of paramFileNameToContentMap. */
  private Map<String, Iterable<String>> getParamFileNameToContentMap() {
    if (paramFileNameToContentMap == null) {
      paramFileNameToContentMap = new HashMap<>();
    }
    return paramFileNameToContentMap;
  }
}
