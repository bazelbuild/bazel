// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An immutable context, including variable bindings for variables introduced by {@link
 * LetExpression}s.
 */
@Immutable
@ThreadSafe
public class QueryExpressionContext<T> {
  protected final ImmutableMap<String, Set<T>> context;

  protected QueryExpressionContext(ImmutableMap<String, Set<T>> context) {
    this.context = context;
  }

  /**
   * Returns the value bound to the specified variable given by {@code name}, or {@code null} if
   * there is no such binding.
   */
  @Nullable
  Set<T> get(String name) {
    return context.get(name);
  }

  /** Returns a {@link QueryExpressionContext} with no variables defined. */
  public static <T> QueryExpressionContext<T> empty() {
    return new QueryExpressionContext<>(ImmutableMap.<String, Set<T>>of());
  }
  /**
   * Returns a {@link QueryExpressionContext} that has all the same bindings as the given {@code
   * variableContext} and also the binding of {@code name} to {@code value}.
   */
  protected QueryExpressionContext<T> with(String name, Set<T> value) {
    return new QueryExpressionContext<>(withNewVariable(name, value));
  }

  protected final ImmutableMap<String, Set<T>> withNewVariable(String name, Set<T> value) {
    ImmutableMap.Builder<String, Set<T>> newContextBuilder = ImmutableMap.builder();
    for (Map.Entry<String, Set<T>> entry : context.entrySet()) {
      if (!entry.getKey().equals(name)) {
        // The binding of 'name' to 'value' should override any existing binding of name in
        // 'variableContext'. These are the semantics we want in order for nested let-expressions
        // to have the semantics we want.
        newContextBuilder.put(entry);
      }
    }
    newContextBuilder.put(name, value);
    return newContextBuilder.buildOrThrow();
  }

  /**
   * A globally defined map of extra dependency edges. If `//a -> //b` is an entry in this map, then
   * any dependency evaluation of the graph should behave as if `//a` depends on `//b`.
   */
  public ImmutableSetMultimap<SkyKey, SkyKey> extraGlobalDeps() {
    // Only subclasses of this class support extra global deps.
    return ImmutableSetMultimap.<SkyKey, SkyKey>of();
  }

  @Override
  public String toString() {
    return "QueryExpressionContext: " + context;
  }
}

