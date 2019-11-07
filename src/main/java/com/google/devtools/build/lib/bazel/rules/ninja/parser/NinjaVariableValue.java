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

import com.google.common.collect.Range;
import com.google.devtools.build.lib.collect.ImmutableSortedKeyListMultimap;
import javax.annotation.concurrent.Immutable;

/**
 * Ninja variable value.
 *
 * <p>Can contain references to the other variables, defined earlier in the scope (or parent scope).
 * It is expected that those references can be replaced in one step, as all the variables are
 * parsed, so this particular structure is only needed to keep the intermediate state.
 */
@Immutable
public final class NinjaVariableValue {

  /** Variable value text. */
  private final String text;
  /** Map of variable names to the list of ranges of their usage in the {@link #text}. */
  private final ImmutableSortedKeyListMultimap<String, Range<Integer>> variables;

  public NinjaVariableValue(
      String text, ImmutableSortedKeyListMultimap<String, Range<Integer>> variables) {
    this.text = text;
    this.variables = variables;
  }

  public String getText() {
    return text;
  }

  public ImmutableSortedKeyListMultimap<String, Range<Integer>> getVariables() {
    return variables;
  }

  @Override
  public String toString() {
    return "NinjaVariableValue{" + "text='" + text + '\'' + ", variables=" + variables + '}';
  }
}
