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

public class NinjaFileParseResult {
  private final NavigableMap<String, List<Pair<Integer, NinjaVariableValue>>> variables;
  private final NavigableMap<String, List<Pair<Integer, NinjaRule>>> rules;
  private final List<ByteFragmentAtOffset> targets;
  private final NavigableMap<Integer, NinjaPromise<NinjaFileParseResult>> includedFilesFutures;
  private final NavigableMap<Integer, NinjaPromise<NinjaFileParseResult>> subNinjaFilesFutures;

  public NinjaFileParseResult() {
    variables = Maps.newTreeMap();
    rules = Maps.newTreeMap();
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
  }

  public static NinjaFileParseResult mergeFileParts(Collection<NinjaFileParseResult> parts) {
    Preconditions.checkState(!parts.isEmpty());
    NinjaFileParseResult result = new NinjaFileParseResult();
    for (NinjaFileParseResult part : parts) {
      for (String name : part.variables.keySet()) {
        result
            .variables
            .computeIfAbsent(name, k -> Lists.newArrayList())
            .addAll(part.variables.get(name));
      }
      for (String name : part.rules.keySet()) {
        result.rules.computeIfAbsent(name, k -> Lists.newArrayList()).addAll(part.rules.get(name));
      }
      result.targets.addAll(part.targets);
      result.includedFilesFutures.putAll(part.includedFilesFutures);
      result.subNinjaFilesFutures.putAll(part.subNinjaFilesFutures);
    }
    result.sortResults();
    return result;
  }

  public void expandIntoScope(NinjaScope scope,
      Map<NinjaScope, List<ByteFragmentAtOffset>> rawTargets)
      throws InterruptedException, GenericParsingException, IOException {
    scope.setRules(rules);
    rawTargets.put(scope, targets);

    TreeMap<Integer, NinjaCallable> resolvables = Maps.newTreeMap();
    for (Map.Entry<String, List<Pair<Integer, NinjaVariableValue>>> entry : variables.entrySet()) {
      String name = entry.getKey();
      for (Pair<Integer, NinjaVariableValue> pair : entry.getValue()) {
        int offset = Preconditions.checkNotNull(pair.getFirst());
        NinjaVariableValue variableValue = Preconditions.checkNotNull(pair.getSecond());
        resolvables.put(
            offset,
            () -> scope.addExpandedVariable(offset, name,
                scope.getExpandedValue(offset, variableValue)));
      }
    }
    for (Map.Entry<Integer, NinjaPromise<NinjaFileParseResult>> entry : includedFilesFutures.entrySet()) {
      Integer offset = entry.getKey();
      resolvables.put(offset, () -> {
        NinjaFileParseResult fileParseResult = entry.getValue().compute(scope);
        NinjaScope includedScope = scope.addIncluded(offset);
        fileParseResult.expandIntoScope(includedScope, rawTargets);
      });
    }
    for (Map.Entry<Integer, NinjaPromise<NinjaFileParseResult>> entry : subNinjaFilesFutures.entrySet()) {
      Integer offset = entry.getKey();
      resolvables.put(offset, () -> {
        NinjaFileParseResult fileParseResult = entry.getValue().compute(scope);
        NinjaScope subNinjaScope = scope.addSubNinja(entry.getKey());
        fileParseResult.expandIntoScope(subNinjaScope, rawTargets);
      });
    }

    for (NinjaCallable ninjaCallable : resolvables.values()) {
      ninjaCallable.call();
    }
  }

  public interface NinjaPromise<T> {
    T compute(NinjaScope scope) throws GenericParsingException, InterruptedException, IOException;
  }

  public interface NinjaCallable {
    void call() throws GenericParsingException, InterruptedException, IOException;
  }
}
