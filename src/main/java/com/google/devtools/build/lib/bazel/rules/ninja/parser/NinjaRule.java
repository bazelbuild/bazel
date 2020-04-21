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
import javax.annotation.concurrent.Immutable;

/**
 * Ninja rule representation.
 *
 * <p>{@link NinjaVariableValue} to be replaced for each target according to the scope rules.
 */
@Immutable
public final class NinjaRule {
  private final ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables;
  private final String name;

  public NinjaRule(
      String name, ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> variables) {
    this.name = name;
    this.variables = variables;
  }

  public ImmutableSortedMap<NinjaRuleVariable, NinjaVariableValue> getVariables() {
    return variables;
  }

  public String getName() {
    return name;
  }
}
