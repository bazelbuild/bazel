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
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Pair;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Ninja file scope to keep all defined variables and rules according to the order of their
 * definition (and redefinition).
 */
public class NinjaScope {
  private final Map<String, List<Pair<Integer, NinjaVariableValue>>> variables;
  private final Map<String, List<Pair<Integer, NinjaRule>>> rules;

  public NinjaScope() {
    variables = Maps.newHashMap();
    rules = Maps.newHashMap();
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
    return findByNameAndOffset(offset, name, variables);
  }

  @Nullable
  public NinjaRule findRule(int offset, String name) {
    return findByNameAndOffset(offset, name, rules);
  }

  @Nullable
  private static <T> T findByNameAndOffset(
      int offset, String name, Map<String, List<Pair<Integer, T>>> map) {
    List<Pair<Integer, T>> pairs = map.get(name);
    if (pairs == null) {
      // We may want to search in the parent scope.
      return null;
    }
    int insertionPoint =
        Collections.binarySearch(
            pairs, Pair.of(offset, null), Comparator.comparing(Pair::getFirst));
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
    return pairs.get(idx).getSecond();
  }

  public static NinjaScope mergeScopeParts(Collection<NinjaScope> parts) {
    NinjaScope result = new NinjaScope();
    for (NinjaScope part : parts) {
      for (String name : part.variables.keySet()) {
        result
            .variables
            .computeIfAbsent(name, k -> Lists.newArrayList())
            .addAll(part.variables.get(name));
      }
      for (String name : part.rules.keySet()) {
        result.rules.computeIfAbsent(name, k -> Lists.newArrayList()).addAll(part.rules.get(name));
      }
    }
    result.sortResults();
    return result;
  }
}
