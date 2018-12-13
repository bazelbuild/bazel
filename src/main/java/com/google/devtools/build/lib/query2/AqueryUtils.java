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
package com.google.devtools.build.lib.query2;

import static com.google.devtools.build.lib.query2.engine.InputsFunction.INPUTS;
import static com.google.devtools.build.lib.query2.engine.MnemonicFunction.MNEMONIC;
import static com.google.devtools.build.lib.query2.engine.OutputsFunction.OUTPUTS;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import java.util.regex.Pattern;

/** Utility class for Aquery */
public class AqueryUtils {

  /**
   * Return true if the given {@code action} matches the filters specified in {@code actionFilters}.
   *
   * @param action the analysis metadata of an action
   * @param actionFilters the filters parsed from the query expression
   * @return whether the action matches the filtering patterns
   */
  public static boolean matchesAqueryFilters(
      ActionAnalysisMetadata action, ImmutableMap<String, Pattern> actionFilters) {
    Iterable<Artifact> inputs = action.getInputs();
    Iterable<Artifact> outputs = action.getOutputs();
    String mnemonic = action.getMnemonic();

    if (actionFilters.containsKey(MNEMONIC)) {
      if (!actionFilters.get(MNEMONIC).matcher(mnemonic).matches()) {
        return false;
      }
    }

    if (actionFilters.containsKey(INPUTS)) {
      Pattern inputsPattern = actionFilters.get(INPUTS);
      Boolean containsFile =
          Streams.stream(inputs)
              .map(artifact -> inputsPattern.matcher(artifact.getExecPathString()).matches())
              .reduce(false, Boolean::logicalOr);

      if (!containsFile) {
        return false;
      }
    }

    if (actionFilters.containsKey(OUTPUTS)) {
      Pattern outputsPattern = actionFilters.get(OUTPUTS);
      Boolean containsFile =
          Streams.stream(outputs)
              .map(artifact -> outputsPattern.matcher(artifact.getExecPathString()).matches())
              .reduce(false, Boolean::logicalOr);

      return containsFile;
    }

    return true;
  }
}
