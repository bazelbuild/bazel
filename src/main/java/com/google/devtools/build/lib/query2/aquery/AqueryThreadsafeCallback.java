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

import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment.TopLevelConfigurations;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import javax.annotation.Nullable;

/** Base class for aquery output callbacks. */
public abstract class AqueryThreadsafeCallback
    extends NamedThreadSafeOutputFormatterCallback<ConfiguredTargetValue> {
  protected final ExtendedEventHandler eventHandler;
  protected final AqueryOptions options;
  protected final PrintStream printStream;
  protected final ConfiguredTargetValueAccessor accessor;
  protected final TopLevelConfigurations topLevelConfigurations;
  @Nullable protected final TopLevelArtifactContext topLevelArtifactContext;

  AqueryThreadsafeCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      TargetAccessor<ConfiguredTargetValue> accessor,
      TopLevelConfigurations topLevelConfigurations,
      @Nullable TopLevelArtifactContext topLevelArtifactContext) {
    this.eventHandler = eventHandler;
    this.options = options;
    this.printStream = out == null ? null : new PrintStream(out);
    this.accessor = (ConfiguredTargetValueAccessor) accessor;
    this.topLevelConfigurations = topLevelConfigurations;
    this.topLevelArtifactContext = topLevelArtifactContext;
  }

  @Nullable
  protected Set<ActionAnalysisMetadata> getReachableActions(
      Iterable<ConfiguredTargetValue> configuredTargetValues) throws InterruptedException {
    if (!options.getPruneUnusedActions()) {
      return null;
    }

    Map<Artifact, ActionAnalysisMetadata> generatingActionMap = new HashMap<>();
    for (ConfiguredTargetValue ctv : configuredTargetValues) {
      if (ctv instanceof RuleConfiguredTargetValue ruleValue) {
        for (ActionAnalysisMetadata action : ruleValue.getActions()) {
          for (Artifact output : action.getOutputs()) {
            generatingActionMap.put(output, action);
          }
        }
      }
      if (options.getUseAspects()) {
        for (AspectValue aspectValue : accessor.getAspectValues(ctv)) {
          if (aspectValue != null) {
            for (ActionAnalysisMetadata action : aspectValue.getActions()) {
              for (Artifact output : action.getOutputs()) {
                generatingActionMap.put(output, action);
              }
            }
          }
        }
      }
    }

    Set<Artifact> visitedArtifacts = new HashSet<>();
    Queue<Artifact> worklist = new ArrayDeque<>();
    for (ConfiguredTargetValue ctv : configuredTargetValues) {
      ConfiguredTarget ct = ctv.getConfiguredTarget();
      if ((topLevelConfigurations == null || topLevelConfigurations.isTopLevelTarget(ct.getLabel()))
          && TopLevelArtifactHelper.shouldConsiderForDisplay(ct)) {

        if (topLevelArtifactContext != null) {
          for (Artifact artifact :
              TopLevelArtifactHelper.getAllArtifactsToBuild(ct, topLevelArtifactContext)
                  .getAllArtifacts()
                  .toList()) {
            if (visitedArtifacts.add(artifact)) {
              worklist.add(artifact);
            }
          }
        } else if (ct.getProvider(FileProvider.class) != null) {
          for (Artifact artifact : ct.getProvider(FileProvider.class).getFilesToBuild().toList()) {
            if (visitedArtifacts.add(artifact)) {
              worklist.add(artifact);
            }
          }
        }
      }
    }

    Set<ActionAnalysisMetadata> reachableActions = new HashSet<>();
    while (!worklist.isEmpty()) {
      Artifact artifact = worklist.poll();
      ActionAnalysisMetadata action = generatingActionMap.get(artifact);
      if (action != null && reachableActions.add(action)) {
        for (Artifact input :
            AqueryUtils.getActionInputs(action, options.getIncludePrunedInputs()).toList()) {
          if (visitedArtifacts.add(input)) {
            worklist.add(input);
          }
        }
      }
    }
    return reachableActions;
  }
}
