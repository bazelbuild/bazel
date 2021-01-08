// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.VerifyException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Class used to build VariablesExtension from a Starlark dictionary. */
public class UserVariablesExtension implements VariablesExtension {
  public Map<String, Object> variablesExtension;

  /**
   * Converts a Starlark dictionary whose values can either be strings, string lists or string
   * depsets to a regular Java map of objects.
   */
  public UserVariablesExtension(Dict<?, ?> variablesExtension) throws EvalException {
    this.variablesExtension =
        Dict.cast(variablesExtension, String.class, Object.class, "variables_extension");
    // Check values are either string, string sequence or string depset
    for (Map.Entry<String, Object> entry : this.variablesExtension.entrySet()) {
      if (entry.getValue() instanceof Sequence) {
        Sequence<?> sequence = (Sequence) entry.getValue();
        if (sequence.isEmpty()) {
          continue;
        }
        if (!(sequence.get(0) instanceof String)) {
          throw new EvalException(
              "Trying to build UserVariableExtension, found non-string type in sequence.");
        }
      } else if (entry.getValue() instanceof Depset) {
        Depset depset = (Depset) entry.getValue();
        if (depset.isEmpty()) {
          continue;
        }
        if (!depset.getElementType().toString().equals("string")) {
          throw new EvalException(
              "Trying to build UserVariableExtension, found non-string type in depset.");
        }
      } else if (!(entry.getValue() instanceof String)) {
        throw new EvalException(
            "Trying to build UserVariableExtension, the value in the UserVariablesExtension dict"
                + " must be a string, string sequence or depset.");
      }
    }
  }

  @Override
  public void addVariables(CcToolchainVariables.Builder builder) {
    for (Map.Entry<String, Object> entry : variablesExtension.entrySet()) {
      if (entry.getValue() instanceof Sequence) {
        try {
          List<String> sequence =
              Sequence.cast(entry.getValue(), String.class, "string_sequence_variables_extension");
          builder.addStringSequenceVariable(entry.getKey(), sequence);
        } catch (EvalException e) {
          // Cannot throw, cast already checked in constructor.
        }
      } else if (entry.getValue() instanceof Depset) {
        try {
          NestedSet<String> nestedSet =
              Depset.cast(entry.getValue(), String.class, "string_sequence_variables_extension");
          builder.addStringSequenceVariable(entry.getKey(), nestedSet);
        } catch (EvalException e) {
          // Cannot throw, cast already checked in constructor.
        }
      } else if (entry.getValue() instanceof String) {
        String value = (String) entry.getValue();
        builder.addStringVariable(entry.getKey(), value);
      } else {
        // If it's any other type we should have thrown an EvalException in the constructor already.
        throw new VerifyException();
      }
    }
  }
}
