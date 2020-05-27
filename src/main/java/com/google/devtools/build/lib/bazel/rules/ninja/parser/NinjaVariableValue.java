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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Ninja variable value.
 *
 * <p>Can contain references to the other variables, defined earlier in the scope (or parent scope).
 * It is expected that those references can be replaced in one step, as all the variables are
 * parsed, so this particular structure is only needed to keep the intermediate state.
 */
public class NinjaVariableValue {
  /**
   * List of functions for computing parts of the variable value. For the case of text part, the
   * function just returns the text literal. For the case of variable reference, function calls the
   * value expander, passed to it as an argument, to get the variable value, and returns it.
   *
   * <p>{@link NinjaVariableValue.Builder#addVariable(String)}
   */
  private final ImmutableList<Function<Function<String, String>, String>> parts;
  /** Indicates that this value contain no variable references, i.e. contain only plain text. */
  private final boolean isPlainText;

  private NinjaVariableValue(
      ImmutableList<Function<Function<String, String>, String>> parts, boolean isPlainText) {
    this.parts = parts;
    this.isPlainText = isPlainText;
  }

  /** Created the value wrapping some plain text. */
  public static NinjaVariableValue createPlainText(String text) {
    return builder().addText(text).build();
  }

  public boolean isPlainText() {
    return isPlainText;
  }

  /** Compute the expanded value, using the passed <code>expander</code> function. */
  @VisibleForTesting
  public String getExpandedValue(Function<String, String> expander) {
    return parts.stream().map(fun -> fun.apply(expander)).collect(Collectors.joining(""));
  }

  /**
   * Returns a "reduced" representation of this value by expanding all variables for which an
   * expanded value is available. If the given expansion function returns null for any part of the
   * value, that part will remain unexpanded in the returned value.
   */
  public NinjaVariableValue reduce(Function<String, String> expander) {
    NinjaVariableValue.Builder builder = builder();
    for (Function<Function<String, String>, String> part : parts) {
      String result = part.apply(expander);
      if (result != null) {
        builder.addText(result);
      } else {
        builder.addPart(part);
      }
    }
    return builder.build();
  }

  /**
   * Returns the fully expanded string that this value represents, given a map from variable name to
   * expanded value.
   *
   * <p>If this value contains an unexpanded variable for which no value is present in the given
   * map, the empty string is used as the value of that variable.
   */
  public String expandValue(ImmutableMap<String, String> map) {
    return getExpandedValue(s -> Strings.nullToEmpty(map.get(s)));
  }

  /**
   * Compute the presentation of this value, replacing the variable references with ${reference}.
   */
  public String getRawText() {
    return getExpandedValue(s -> String.format("${%s}", s));
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder class for {@link NinjaVariableValue}. */
  public static final class Builder {
    private final ImmutableList.Builder<Function<Function<String, String>, String>> builder;
    private boolean isPrimitive = true;

    private Builder() {
      this.builder = ImmutableList.builder();
    }

    /** Add plain text fragment. */
    public Builder addText(String text) {
      builder.add(expander -> text);
      return this;
    }

    /** Add reference to variable <code>name</code>. */
    public Builder addVariable(String name) {
      builder.add(expander -> expander.apply(name));
      isPrimitive = false;
      return this;
    }

    private Builder addPart(Function<Function<String, String>, String> part) {
      builder.add(part);
      isPrimitive = false;
      return this;
    }

    public NinjaVariableValue build() {
      return new NinjaVariableValue(builder.build(), isPrimitive);
    }
  }
}
