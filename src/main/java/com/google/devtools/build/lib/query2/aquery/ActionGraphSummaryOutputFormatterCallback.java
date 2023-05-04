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
package com.google.devtools.build.lib.query2.aquery;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/** Output callback for aquery, prints a human readable summary. */
class ActionGraphSummaryOutputFormatterCallback extends AqueryThreadsafeCallback {

  private final AqueryActionFilter actionFilters;
  private final Map<String, Integer> mnemonicToCount = new HashMap<>();
  private final Map<String, Integer> configurationToCount = new HashMap<>();
  private final Map<String, Integer> execPlatformToCount = new HashMap<>();
  private final Map<String, Integer> aspectToCount = new HashMap<>();

  ActionGraphSummaryOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      TargetAccessor<KeyedConfiguredTargetValue> accessor,
      AqueryActionFilter actionFilters) {
    super(eventHandler, options, out, accessor);
    this.actionFilters = actionFilters;
  }

  @Override
  public String getName() {
    return "summary";
  }

  @Override
  public void processOutput(Iterable<KeyedConfiguredTargetValue> partialResult)
      throws IOException, InterruptedException {
    // Enabling includeParamFiles should enable includeCommandline by default.
    options.includeCommandline |= options.includeParamFiles;

    for (KeyedConfiguredTargetValue keyedConfiguredTargetValue : partialResult) {
      ConfiguredTargetValue configuredTargetValue =
          keyedConfiguredTargetValue.getConfiguredTargetValue();
      if (!(configuredTargetValue instanceof RuleConfiguredTargetValue)) {
        // We have to include non-rule values in the graph to visit their dependencies, but they
        // don't have any actions to print out.
        continue;
      }
      for (ActionAnalysisMetadata action :
          ((RuleConfiguredTargetValue) configuredTargetValue).getActions()) {
        processAction(action);
      }
      if (options.useAspects) {
        for (AspectValue aspectValue : accessor.getAspectValues(keyedConfiguredTargetValue)) {
          for (ActionAnalysisMetadata action : aspectValue.getActions()) {
            processAction(action);
          }
        }
      }
    }
  }

  private void processAction(ActionAnalysisMetadata action) throws InterruptedException {
    if (!AqueryUtils.matchesAqueryFilters(action, actionFilters)) {
      return;
    }

    mnemonicToCount.merge(action.getMnemonic(), 1, Integer::sum);
    ActionOwner actionOwner = action.getOwner();
    if (actionOwner != null) {
      BuildEvent configuration = actionOwner.getBuildConfigurationEvent();
      BuildEventStreamProtos.Configuration configProto =
          configuration.asStreamProto(/*context=*/ null).getConfiguration();
      configurationToCount.merge(configProto.getMnemonic(), 1, Integer::sum);

      if (actionOwner.getExecutionPlatform() != null) {
        execPlatformToCount.merge(
            actionOwner.getExecutionPlatform().label().toString(), 1, Integer::sum);
      }

      // In the case of aspect-on-aspect, AspectDescriptors are listed in
      // topological order of the dependency graph.
      // e.g. [A -> B] would imply that aspect A is applied on top of aspect B.
      ImmutableList<AspectDescriptor> aspectDescriptors =
          actionOwner.getAspectDescriptors().reverse();
      if (!aspectDescriptors.isEmpty()) {
        aspectDescriptors.forEach(
            aspectDescriptor ->
                aspectToCount.merge(aspectDescriptor.getAspectClass().getName(), 1, Integer::sum));
      }
    }
  }

  @Override
  public void close(boolean failFast) throws InterruptedException, IOException {
    if (failFast) {
      return;
    }

    int totalActions = mnemonicToCount.values().stream().mapToInt(v -> v).sum();
    if (totalActions == 0) {
      printStream.println("No actions matched.");
    } else {
      printStream.println(totalActions + " total action" + (totalActions == 1 ? "" : "s") + ".");
    }

    printSummary(mnemonicToCount, "Mnemonics:");
    printSummary(configurationToCount, "Configurations:");
    printSummary(execPlatformToCount, "Execution Platforms:");
    printSummary(aspectToCount, "Aspects:");
  }

  private void printSummary(Map<String, Integer> actionsCount, String s) {
    if (!actionsCount.isEmpty()) {
      printStream.println();
      printStream.println(s);
      actionsCount.entrySet().stream()
          .sorted(Comparator.comparingInt(Entry::getValue))
          .forEach(entry -> printStream.println("  " + entry.getKey() + ": " + entry.getValue()));
    }
  }
}
