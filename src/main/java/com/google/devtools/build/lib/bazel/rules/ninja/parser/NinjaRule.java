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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import java.util.Objects;

/**
 * Ninja rule representation.
 *
 * {@link NinjaVariableValue} to be replaced for each target according to the scope rules.
 */
public class NinjaRule {
  private final ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables;
  private final ImmutableSortedMap<String, NinjaVariableValue> customVariables;

  public NinjaRule(ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables,
      ImmutableSortedMap<String, NinjaVariableValue> customVariables) {
    this.variables = variables;
    this.customVariables = customVariables;
  }

  public ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> getVariables() {
    return variables;
  }

  public ImmutableSortedMap<String, NinjaVariableValue> getCustomVariables() {
    return customVariables;
  }

  public String getName() {
    NinjaVariableValue value = Preconditions.checkNotNull(variables.get(NinjaRuleVariable.NAME));
    Preconditions.checkState(value.getVariables().isEmpty());
    return value.getText();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (! (o instanceof NinjaRule)) {
      return false;
    }
    NinjaRule ninjaRule = (NinjaRule) o;
    return variables.equals(ninjaRule.variables);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(variables);
  }
}
