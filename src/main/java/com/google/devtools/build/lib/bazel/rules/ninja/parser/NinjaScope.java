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

import static com.google.common.base.Strings.nullToEmpty;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.function.Consumer;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Ninja file scope to keep all defined variables and rules according to the order of their
 * definition (and redefinition).
 */
public class NinjaScope {
  /** Parent scope for the case of subninja/include command */
  @Nullable private final NinjaScope parentScope;
  /** If include command was used for the current scope, the offset of that include command */
  @Nullable private final Long includePoint;

  private final NavigableMap<Long, NinjaScope> includedScopes;
  private final NavigableMap<Long, NinjaScope> subNinjaScopes;
  private Map<String, List<Pair<Long, String>>> expandedVariables;
  private final Map<String, List<Pair<Long, NinjaRule>>> rules;
  private final Map<String, List<Pair<Long, NinjaPool>>> pools;

  public NinjaScope() {
    this(null, null);
  }

  private NinjaScope(@Nullable NinjaScope parentScope, @Nullable Long includePoint) {
    this.parentScope = parentScope;
    this.includePoint = includePoint;
    this.rules = Maps.newTreeMap();
    this.pools = Maps.newTreeMap();
    this.includedScopes = Maps.newTreeMap();
    this.subNinjaScopes = Maps.newTreeMap();
    this.expandedVariables = Maps.newHashMap();
  }

  public void setRules(Map<String, List<Pair<Long, NinjaRule>>> rules) {
    this.rules.putAll(rules);
  }

  public void setPools(Map<String, List<Pair<Long, NinjaPool>>> pools) {
    this.pools.putAll(pools);
  }

  @VisibleForTesting
  public Map<String, List<Pair<Long, NinjaRule>>> getRules() {
    return rules;
  }

  @VisibleForTesting
  public Map<String, List<Pair<Long, NinjaPool>>> getPools() {
    return pools;
  }

  public Collection<NinjaScope> getIncludedScopes() {
    return includedScopes.values();
  }

  public Collection<NinjaScope> getSubNinjaScopes() {
    return subNinjaScopes.values();
  }

  /**
   * Expands variable value at the given offset. If some of the variable references, used in the
   * value, can not be found, uses an empty string as their value.
   */
  public String getExpandedValue(long offset, NinjaVariableValue value) {
    // Cache expanded variables values to save time replacing several references to the same
    // variable.
    // This cache is local to the offset, it depends on the offset of the variable we are expanding.
    Map<String, String> cache = Maps.newHashMap();
    // We are using the start offset of the value holding the reference to the variable.
    // Do the same as Ninja implementation: if the variable is not found, use empty string.
    Function<String, String> expander =
        ref -> cache.computeIfAbsent(ref, (key) -> nullToEmpty(findExpandedVariable(offset, key)));
    return value.getExpandedValue(expander);
  }

  public void addExpandedVariable(Long offset, String name, String value) {
    expandedVariables.computeIfAbsent(name, k -> Lists.newArrayList()).add(Pair.of(offset, value));
  }

  public NinjaScope addIncluded(Long offset) {
    NinjaScope scope = new NinjaScope(this, offset);
    includedScopes.put(offset, scope);
    return scope;
  }

  public NinjaScope addSubNinja(Long offset) {
    NinjaScope scope = new NinjaScope(this, offset);
    subNinjaScopes.put(offset, scope);
    return scope;
  }

  /**
   * Finds expanded variable with the name <code>name</code> to be used in the reference to it at
   * <code>offset</code>. Returns null if nothing was found.
   */
  @Nullable
  public String findExpandedVariable(long offset, String name) {
    return findByNameAndOffsetRecursively(offset, name, scope -> scope.expandedVariables);
  }

  /**
   * Finds a rule with the name <code>name</code> to be used in the reference to it at <code>offset
   * </code>. Returns null if nothing was found.
   */
  @Nullable
  public NinjaRule findRule(long offset, String name) {
    return findByNameAndOffsetRecursively(offset, name, scope -> scope.rules);
  }

  /**
   * Finds a variable or rule with the name <code>name</code> to be used in the reference to it at
   * <code>offset</code>.
   *
   * <p>The following checks are made: - the last definition of variable/rule before the offset in
   * the current scope is looked up. - the last definition of variable/rule inside the relevant
   * included scopes (i.e. in the files from include statements before offset)
   *
   * <p>If any of the definitions are found in the current or included scopes, the value with the
   * largest offset is returned.
   *
   * <p>If nothing is found, we make an attempt to find the definition in the parent scope at offset
   * before the offset at which the current scope was introduced to parent.
   *
   * <p>If no definition was found, we return null.
   */
  @Nullable
  private <T> T findByNameAndOffsetRecursively(
      long offset,
      String name,
      Function<NinjaScope, Map<String, List<Pair<Long, T>>>> mapSupplier) {
    Pair<Long, T> currentScopeValue = findByNameAndOffset(offset, name, this, mapSupplier);

    Long currentScopeOffset =
        currentScopeValue != null ? Preconditions.checkNotNull(currentScopeValue.getFirst()) : -1L;

    // Search in included scopes, which were included after the current scope, so they could
    // override the value, but before the reference offset.
    NavigableMap<Long, NinjaScope> subMap =
        includedScopes.subMap(currentScopeOffset, false, offset, false);
    // Search in descending order, so that the first found value is the result.
    for (NinjaScope includedScope : subMap.descendingMap().values()) {
      T includedValue =
          includedScope.findByNameAndOffsetRecursively(Long.MAX_VALUE, name, mapSupplier);
      if (includedValue != null) {
        return includedValue;
      }
    }
    if (currentScopeValue != null) {
      return currentScopeValue.getSecond();
    }
    if (parentScope != null) {
      Preconditions.checkNotNull(includePoint);
      // -1 is used in order not to conflict with the current scope.
      return parentScope.findByNameAndOffsetRecursively(includePoint - 1, name, mapSupplier);
    }
    return null;
  }

  /**
   * Finds the variable or rule with the name <code>name</code>, defined in the current scope before
   * the <code>offset</code>. (Ninja allows to re-define the values of rules and variables.)
   */
  @Nullable
  private static <T> Pair<Long, T> findByNameAndOffset(
      long offset,
      String name,
      NinjaScope scope,
      Function<NinjaScope, Map<String, List<Pair<Long, T>>>> mapFunction) {
    List<Pair<Long, T>> pairs = Preconditions.checkNotNull(mapFunction.apply(scope)).get(name);
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
    Pair<Long, T> pair = pairs.get(idx);
    return Pair.of(pair.getFirst(), pair.getSecond());
  }

  public NinjaScope createScopeFromExpandedValues(
      ImmutableSortedMap<String, List<Pair<Long, String>>> expandedVariables) {
    NinjaScope scope = new NinjaScope(this, Long.MAX_VALUE);
    scope.expandedVariables.putAll(expandedVariables);
    return scope;
  }

  public void iterate(Consumer<NinjaScope> consumer) {
    ArrayDeque<NinjaScope> queue = new ArrayDeque<>();
    queue.add(this);
    while (!queue.isEmpty()) {
      NinjaScope currentScope = queue.removeFirst();
      consumer.accept(currentScope);
      queue.addAll(currentScope.getIncludedScopes());
      queue.addAll(currentScope.getSubNinjaScopes());
    }
  }
}
