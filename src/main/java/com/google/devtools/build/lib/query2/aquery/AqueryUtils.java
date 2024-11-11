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

import static com.google.devtools.build.lib.query2.engine.InputsFunction.INPUTS;
import static com.google.devtools.build.lib.query2.engine.MnemonicFunction.MNEMONIC;
import static com.google.devtools.build.lib.query2.engine.OutputsFunction.OUTPUTS;

import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.io.IOException;

/** Utility class for Aquery */
public class AqueryUtils {

  private AqueryUtils() {}

  /**
   * Returns the set of action inputs according to the --include_pruned_inputs flag.
   *
   * <p>This may differ from {@link ActionAnalysisMetadata#getInputs} for actions that discover
   * inputs.
   *
   * @param action the analysis metadata of an action
   * @param includePrunedInputs the value of the --include_pruned_inputs flag
   */
  public static NestedSet<Artifact> getActionInputs(
      ActionAnalysisMetadata action, boolean includePrunedInputs) {
    if (includePrunedInputs
        || (action instanceof ActionExecutionMetadata actionExecutionMetadata
            && !actionExecutionMetadata.inputsKnown())) {
      // getInputs() is potentially missing inputs that will be added by discovery (if the action
      // hasn't yet executed) and inputs that have been removed by discovery (if the action has
      // already executed). Instead, assemble the inputs from getOriginalInputs() and
      // getSchedulingDependencies(), which also include those added or removed by discovery.
      return NestedSetBuilder.<Artifact>stableOrder()
          .addTransitive(action.getOriginalInputs())
          .addTransitive(action.getSchedulingDependencies())
          .build();
    }
    return action.getInputs();
  }

  /**
   * Return true if the given {@code action} matches the filters specified in {@code actionFilters}.
   *
   * @param action the analysis metadata of an action
   * @param actionFilters the filters parsed from the query expression
   * @param includePrunedInputs the value of the --include_pruned_inputs flag
   * @return whether the action matches the filtering patterns
   */
  public static boolean matchesAqueryFilters(
      ActionAnalysisMetadata action,
      AqueryActionFilter actionFilters,
      boolean includePrunedInputs) {
    NestedSet<Artifact> inputs = getActionInputs(action, includePrunedInputs);
    Iterable<Artifact> outputs = action.getOutputs();
    String mnemonic = action.getMnemonic();

    if (actionFilters.hasFilterForFunction(MNEMONIC)) {
      if (!actionFilters.matchesAllPatternsForFunction(MNEMONIC, mnemonic)) {
        return false;
      }
    }

    if (actionFilters.hasFilterForFunction(INPUTS)) {
      boolean containsFile =
          inputs.toList().stream()
              .anyMatch(
                  artifact ->
                      actionFilters.matchesAllPatternsForFunction(
                          INPUTS, artifact.getExecPathString()));

      if (!containsFile) {
        return false;
      }
    }

    if (actionFilters.hasFilterForFunction(OUTPUTS)) {
      boolean containsFile =
          Streams.stream(outputs)
              .anyMatch(
                  artifact ->
                      actionFilters.matchesAllPatternsForFunction(
                          OUTPUTS, artifact.getExecPathString()));

      return containsFile;
    }

    return true;
  }

  public static String getTemplateContent(TemplateExpansionAction action) throws IOException {
    // If the template artifact is a DerivedArtifact, it is only available during the execution
    // phase. It's therefore not possible to read its content from the FileSystem at this moment.
    if (action.getTemplate().getTemplateArtifact() instanceof DerivedArtifact) {
      return action.getTemplate().toString();
    }
    return action.getTemplate().getContent(ArtifactPathResolver.IDENTITY);
  }
}
