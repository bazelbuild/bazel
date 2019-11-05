// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.util.Pair;
import java.util.Objects;

/**
 * Ninja variable value.
 *
 * Can contain references to the other variables, defined earlier in the scope (or parent scope).
 * It is expected that those references can be replaced in one step, as all the variables
 * are parsed, so this particular structure is only needed to keep the intermediate state.
 */
public class NinjaVariableValue {
  private final String text;
  private final ImmutableSortedMap<String, Pair<Integer, Integer>> variables;

  public NinjaVariableValue(
      String text,
      ImmutableSortedMap<String, Pair<Integer, Integer>> variables) {
    this.text = text;
    this.variables = variables;
  }

  public String getText() {
    return text;
  }

  public ImmutableSortedMap<String, Pair<Integer, Integer>> getVariables() {
    return variables;
  }

  @Override
  public String toString() {
    return "NinjaVariableValue{" +
        "text='" + text + '\'' +
        ", variables=" + variables +
        '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof NinjaVariableValue)) {
      return false;
    }
    NinjaVariableValue that = (NinjaVariableValue) o;
    return Objects.equals(text, that.text) &&
        Objects.equals(variables, that.variables);
  }

  @Override
  public int hashCode() {
    return Objects.hash(text, variables);
  }
}
