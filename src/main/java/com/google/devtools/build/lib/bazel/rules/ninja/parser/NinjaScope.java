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
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Pair;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.SortedMap;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Ninja file scope to keep all defined variables and rules according
 * to the order of their definition (and redefinition).
 */
public class NinjaScope {
  /**
   * Parent scope for the case of subninja/include command
   */
  @Nullable
  private final NinjaScope parentScope;
  /**
   * If include command was used for the current scope, the offset of that include command
   */
  @Nullable
  private final Integer includePoint;
  private final SortedMap<Integer, NinjaScope> includedScopes;
  private final Map<String, List<Pair<Integer, NinjaVariableValue>>> variables;
  private final Map<String, List<Pair<Integer, NinjaRule>>> rules;

  public NinjaScope() {
    this(null, null);
  }

  public NinjaScope(@Nullable NinjaScope parentScope, @Nullable Integer includePoint) {
    this.parentScope = parentScope;
    this.includePoint = includePoint;
    includedScopes = Maps.newTreeMap();
    variables = Maps.newHashMap();
    rules = Maps.newHashMap();
  }

  public NinjaScope createIncludeScope(int offset) {
    NinjaScope includeScope = new NinjaScope(this, offset);
    includedScopes.put(offset, includeScope);
    return includeScope;
  }

  public void addVariable(String name, int offset, NinjaVariableValue value) {
    variables.computeIfAbsent(name, k -> Lists.newArrayList()).add(Pair.of(offset, value));
  }

  public void addRule(int offset, NinjaRule rule) {
    rules.computeIfAbsent(rule.getName(), k -> Lists.newArrayList()).add(Pair.of(offset, rule));
  }

  @VisibleForTesting
  public Map<String, List<Pair<Integer, NinjaVariableValue>>> getVariables() {
    return variables;
  }

  @VisibleForTesting
  public Map<String, List<Pair<Integer, NinjaRule>>> getRules() {
    return rules;
  }

  @VisibleForTesting
  public void sortResults() {
    for (List<Pair<Integer, NinjaVariableValue>> list : variables.values()) {
      list.sort(Comparator.comparing(Pair::getFirst));
    }
    for (List<Pair<Integer, NinjaRule>> list : rules.values()) {
      list.sort(Comparator.comparing(Pair::getFirst));
    }
  }

  @Nullable
  public NinjaVariableValue findVariable(int offset, String name) {
    return findByNameAndOffsetRecursively(offset, name, scope -> scope.variables);
  }

  @Nullable
  public NinjaRule findRule(int offset, String name) {
    return findByNameAndOffsetRecursively(offset, name, scope -> scope.rules);
  }

  @Nullable
  private <T> T findByNameAndOffsetRecursively(int offset, String name,
      Function<NinjaScope, Map<String, List<Pair<Integer, T>>>> mapSupplier) {
    Pair<Integer, T> currentScopeValue = findByNameAndOffset(offset, name, this, mapSupplier);
    T result = currentScopeValue != null ? currentScopeValue.getSecond() : null;

    if (!includedScopes.isEmpty()) {
      NavigableMap<Integer, T> variants = Maps.newTreeMap();
      if (currentScopeValue != null) {
        variants.put(currentScopeValue.getFirst(), currentScopeValue.getSecond());
      }
      for (Map.Entry<Integer, NinjaScope> entry : includedScopes.entrySet()) {
        // Only if the file was included before the reference.
        Integer includeOffset = entry.getKey();
        if (includeOffset < offset) {
          NinjaScope includedScope = entry.getValue();
          Pair<Integer, T> includedValue = findByNameAndOffset(Integer.MAX_VALUE,
              name, includedScope, mapSupplier);
          if (includedValue != null) {
            // Put at include statement offset.
            variants.put(includeOffset, includedValue.getSecond());
          }
        }
      }
      if (! variants.isEmpty()) {
        result = variants.lastEntry().getValue();
      }
    }
    if (result != null) {
      return result;
    }
    if (parentScope != null) {
      Preconditions.checkNotNull(includePoint);
      // -1 is used to do not conflict with the current scope.
      return parentScope.findByNameAndOffsetRecursively(includePoint - 1, name, mapSupplier);
    }
    return null;
  }

  @Nullable
  private static <T> Pair<Integer, T> findByNameAndOffset(int offset, String name,
      NinjaScope scope,
      Function<NinjaScope, Map<String, List<Pair<Integer, T>>>> mapFunction) {
    List<Pair<Integer, T>> pairs = Preconditions.checkNotNull(mapFunction.apply(scope)).get(name);
    if (pairs == null) {
      // We may want to search in the parent scope.
      return null;
    }
    int insertionPoint = Collections
        .binarySearch(pairs, Pair.of(offset, null), Comparator.comparing(Pair::getFirst));
    if (insertionPoint >= 0) {
      // Can not be, variable can not be defined in exactly same place.
      throw new IllegalStateException("Trying to interpret declaration as reference.");
    }
    // We need to access the previous element, before the insertion point.
    int idx = -insertionPoint - 2;
    if (idx < 0) {
      // Check the parent scope.
      return null;
    }
    Pair<Integer, T> pair = pairs.get(idx);
    return Pair.of(pair.getFirst(), pair.getSecond());
  }

  public static NinjaScope mergeScopeParts(Collection<NinjaScope> parts) {
    Preconditions.checkState(!parts.isEmpty());
    NinjaScope first = Preconditions.checkNotNull(Iterables.getFirst(parts, null));
    NinjaScope result = new NinjaScope(first.parentScope, first.includePoint);
    for (NinjaScope part : parts) {
      for (String name : part.variables.keySet()) {
        result.variables
            .computeIfAbsent(name, k -> Lists.newArrayList())
            .addAll(part.variables.get(name));
      }
      for (String name : part.rules.keySet()) {
        result.rules
            .computeIfAbsent(name, k -> Lists.newArrayList())
            .addAll(part.rules.get(name));
      }
    }
    result.sortResults();
    return result;
  }
}
