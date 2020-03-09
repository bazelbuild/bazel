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
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bazel.rules.ninja.file.ByteFragmentAtOffset;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/**
 * Class to hold information about different declarations in Ninja file during parsing.
 *
 * <p>Included files (with include or subninja statements) are kept in form of promises: {@link
 * NinjaPromise<NinjaFileParseResult>}, since for their parsing we may need to first resolve the
 * variables in the current file.
 */
public class NinjaFileParseResult {
  /**
   * Interface for getting the result of lazy parsing of Ninja file in the context of {@link
   * NinjaScope}.
   *
   * @param <T> result of parsing.
   */
  public interface NinjaPromise<T> {
    T compute(NinjaScope scope) throws GenericParsingException, InterruptedException, IOException;
  }

  /** Interface for getting result of lazy parsing of Ninja declaration. */
  public interface NinjaCallable {
    void call() throws GenericParsingException, InterruptedException, IOException;
  }

  private final NavigableMap<String, List<Pair<Integer, NinjaVariableValue>>> variables;
  private final NavigableMap<String, List<Pair<Integer, NinjaRule>>> rules;
  private final NavigableMap<String, List<Pair<Integer, NinjaPool>>> pools;
  private final List<ByteFragmentAtOffset> targets;
  private final NavigableMap<Integer, NinjaPromise<NinjaFileParseResult>> includedFilesFutures;
  private final NavigableMap<Integer, NinjaPromise<NinjaFileParseResult>> subNinjaFilesFutures;

  public NinjaFileParseResult() {
    variables = Maps.newTreeMap();
    rules = Maps.newTreeMap();
    pools = Maps.newTreeMap();
    targets = Lists.newArrayList();
    includedFilesFutures = Maps.newTreeMap();
    subNinjaFilesFutures = Maps.newTreeMap();
  }

  public void addIncludeScope(int offset, NinjaPromise<NinjaFileParseResult> promise) {
    includedFilesFutures.put(offset, promise);
  }

  public void addSubNinjaScope(int offset, NinjaPromise<NinjaFileParseResult> promise) {
    subNinjaFilesFutures.put(offset, promise);
  }

  public void addTarget(ByteFragmentAtOffset fragment) {
    targets.add(fragment);
  }

  public void addVariable(String name, int offset, NinjaVariableValue value) {
    variables.computeIfAbsent(name, k -> Lists.newArrayList()).add(Pair.of(offset, value));
  }

  public void addRule(int offset, NinjaRule rule) {
    rules.computeIfAbsent(rule.getName(), k -> Lists.newArrayList()).add(Pair.of(offset, rule));
  }

  public void addPool(int offset, NinjaPool pool) {
    pools.computeIfAbsent(pool.getName(), k -> Lists.newArrayList()).add(Pair.of(offset, pool));
  }

  @VisibleForTesting
  public Map<String, List<Pair<Integer, NinjaVariableValue>>> getVariables() {
    return variables;
  }

  @VisibleForTesting
  public Map<String, List<Pair<Integer, NinjaRule>>> getRules() {
    return rules;
  }

  public List<ByteFragmentAtOffset> getTargets() {
    return targets;
  }

  @VisibleForTesting
  public void sortResults() {
    for (List<Pair<Integer, NinjaVariableValue>> list : variables.values()) {
      list.sort(Comparator.comparing(Pair::getFirst));
    }
    for (List<Pair<Integer, NinjaRule>> list : rules.values()) {
      list.sort(Comparator.comparing(Pair::getFirst));
    }
    for (List<Pair<Integer, NinjaPool>> list : pools.values()) {
      list.sort(Comparator.comparing(Pair::getFirst));
    }
  }

  public static NinjaFileParseResult merge(Collection<NinjaFileParseResult> parts) {
    NinjaFileParseResult result = new NinjaFileParseResult();
    if (parts.isEmpty()) {
      return result;
    }
    for (NinjaFileParseResult part : parts) {
      for (Map.Entry<String, List<Pair<Integer, NinjaVariableValue>>> entry :
          part.variables.entrySet()) {
        String name = entry.getKey();
        result.variables.computeIfAbsent(name, k -> Lists.newArrayList()).addAll(entry.getValue());
      }
      for (Map.Entry<String, List<Pair<Integer, NinjaRule>>> entry : part.rules.entrySet()) {
        String name = entry.getKey();
        result.rules.computeIfAbsent(name, k -> Lists.newArrayList()).addAll(entry.getValue());
      }
      for (Map.Entry<String, List<Pair<Integer, NinjaPool>>> entry : part.pools.entrySet()) {
        String name = entry.getKey();
        result.pools.computeIfAbsent(name, k -> Lists.newArrayList()).addAll(entry.getValue());
      }
      result.targets.addAll(part.targets);
      result.includedFilesFutures.putAll(part.includedFilesFutures);
      result.subNinjaFilesFutures.putAll(part.subNinjaFilesFutures);
    }
    result.sortResults();
    return result;
  }

  /**
   * Recursively expands variables in the Ninja file and all files it includes (and subninja's).
   * Fills in passed {@link NinjaScope} with the expanded variables and rules, and <code>rawTargets
   * </code> - map of NinjaScope to list of fragments with unparsed Ninja targets.
   */
  public void expandIntoScope(
      NinjaScope scope, Map<NinjaScope, List<ByteFragmentAtOffset>> rawTargets)
      throws InterruptedException, GenericParsingException, IOException {
    scope.setRules(rules);
    scope.setPools(pools);
    rawTargets.put(scope, targets);

    TreeMap<Integer, NinjaCallable> resolvables = Maps.newTreeMap();
    for (Map.Entry<String, List<Pair<Integer, NinjaVariableValue>>> entry : variables.entrySet()) {
      String name = entry.getKey();
      for (Pair<Integer, NinjaVariableValue> pair : entry.getValue()) {
        int offset = Preconditions.checkNotNull(pair.getFirst());
        NinjaVariableValue variableValue = Preconditions.checkNotNull(pair.getSecond());
        resolvables.put(
            offset,
            () ->
                scope.addExpandedVariable(
                    offset, name, scope.getExpandedValue(offset, variableValue)));
      }
    }
    for (Map.Entry<Integer, NinjaPromise<NinjaFileParseResult>> entry :
        includedFilesFutures.entrySet()) {
      Integer offset = entry.getKey();
      resolvables.put(
          offset,
          () -> {
            NinjaFileParseResult fileParseResult = entry.getValue().compute(scope);
            NinjaScope includedScope = scope.addIncluded(offset);
            fileParseResult.expandIntoScope(includedScope, rawTargets);
          });
    }
    for (Map.Entry<Integer, NinjaPromise<NinjaFileParseResult>> entry :
        subNinjaFilesFutures.entrySet()) {
      Integer offset = entry.getKey();
      resolvables.put(
          offset,
          () -> {
            NinjaFileParseResult fileParseResult = entry.getValue().compute(scope);
            NinjaScope subNinjaScope = scope.addSubNinja(entry.getKey());
            fileParseResult.expandIntoScope(subNinjaScope, rawTargets);
          });
    }

    for (NinjaCallable ninjaCallable : resolvables.values()) {
      ninjaCallable.call();
    }
  }
}
