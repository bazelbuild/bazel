// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.parser;

import static com.google.common.base.Strings.nullToEmpty;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Pair;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Ninja file scope to keep all defined variables and rules according to the order of their
 * definition (and redefinition).
 */
public class NinjaScope {
  private final int scopeId;
  /** Parent scope id for the case of subninja/include command */
  @Nullable private final Integer parentScopeId;
  /** If include command was used for the current scope, the offset of that include command */
  @Nullable private final Integer includePoint;

  private final NavigableMap<Integer, Integer> includedScopes;
  private final NavigableMap<Integer, Integer> subNinjaScopes;
  private Map<String, List<Pair<Integer, String>>> expandedVariables;
  private final Map<String, List<Pair<Integer, NinjaRule>>> rules;

  NinjaScope(int scopeId,
      @Nullable Integer parentScopeId,
      @Nullable Integer includePoint) {
    this(scopeId, parentScopeId, includePoint, Maps.newTreeMap(), Maps.newTreeMap(),
        Maps.newTreeMap(), Maps.newTreeMap());
  }

  private NinjaScope(int scopeId,
      @Nullable Integer parentScopeId,
      @Nullable Integer includePoint,
      Map<String, List<Pair<Integer, NinjaRule>>> rules,
      NavigableMap<Integer, Integer> includedScopes,
      NavigableMap<Integer, Integer> subNinjaScopes,
      Map<String, List<Pair<Integer, String>>> expandedVariables) {
    this.scopeId = scopeId;
    this.parentScopeId = parentScopeId;
    this.includePoint = includePoint;
    this.rules = rules;
    this.includedScopes = includedScopes;
    this.subNinjaScopes = subNinjaScopes;
    this.expandedVariables = expandedVariables;
  }

  NinjaScope freeze() {
    return new NinjaScope(
        this.scopeId,
        this.includePoint,
        this.parentScopeId,
        convertToImmutableMap(rules, ImmutableSortedMap.naturalOrder()),
        convertToImmutableMap(includedScopes),
        convertToImmutableMap(subNinjaScopes),
        convertToImmutableMap(expandedVariables, ImmutableSortedMap.naturalOrder()));
  }

  public boolean deepEquals(NinjaScope that) {
    return scopeId == that.scopeId &&
        Objects.equals(parentScopeId, that.parentScopeId) &&
        Objects.equals(includePoint, that.includePoint) &&
        includedScopes.equals(that.includedScopes) &&
        subNinjaScopes.equals(that.subNinjaScopes) &&
        expandedVariables.equals(that.expandedVariables) &&
        rules.equals(that.rules);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    NinjaScope that = (NinjaScope) o;
    return scopeId == that.scopeId;
  }

  @Override
  public int hashCode() {
    return scopeId;
  }

  private ImmutableSortedMap<Integer, Integer> convertToImmutableMap(
      Map<Integer, Integer> map) {
    ImmutableSortedMap.Builder<Integer, Integer> builder = ImmutableSortedMap.naturalOrder();
    for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return builder.build();
  }

  private <K extends Comparable<K>, V> ImmutableSortedMap<K, List<V>>
    convertToImmutableMap(Map<K, List<V>> map, ImmutableSortedMap.Builder<K, List<V>> builder) {
    for (Map.Entry<K, List<V>> entry : map.entrySet()) {
      builder.put(entry.getKey(), ImmutableList.copyOf(entry.getValue()));
    }
    return builder.build();
  }

  public void setRules(Map<String, List<Pair<Integer, NinjaRule>>> rules) {
    this.rules.putAll(rules);
  }

  @VisibleForTesting
  public Map<String, List<Pair<Integer, NinjaRule>>> getRules() {
    return rules;
  }

  public Collection<NinjaScope> getIncludedScopes(NinjaScopeRegister register) {
    return register.getScopesByIds(includedScopes.values());
  }

  public Collection<NinjaScope> getSubNinjaScopes(NinjaScopeRegister register) {
    return register.getScopesByIds(subNinjaScopes.values());
  }

  /**
   * Expands variable value at the given offset. If some of the variable references, used in the
   * value, can not be found, uses an empty string as their value.
   */
  public String getExpandedValue(NinjaScopeRegister register, int offset, NinjaVariableValue value) {
    // Cache expanded variables values to save time replacing several references to the same
    // variable.
    // This cache is local to the offset, it depends on the offset of the variable we are expanding.
    Map<String, String> cache = Maps.newHashMap();
    // We are using the start offset of the value holding the reference to the variable.
    // Do the same as Ninja implementation: if the variable is not found, use empty string.
    Function<String, String> expander =
        ref -> cache.computeIfAbsent(ref, (key) -> nullToEmpty(findExpandedVariable(register, offset, key)));
    return value.getExpandedValue(expander);
  }

  public void addExpandedVariable(int offset, String name, String value) {
    expandedVariables.computeIfAbsent(name, k -> Lists.newArrayList()).add(Pair.of(offset, value));
  }

  public NinjaScope addIncluded(NinjaScopeRegister register, int offset) {
    NinjaScope scope = register.createChildScope(this, offset);
    includedScopes.put(offset, scope.getScopeId());
    return scope;
  }

  public NinjaScope addSubNinja(NinjaScopeRegister register, int offset) {
    NinjaScope scope = register.createChildScope(this, offset);
    subNinjaScopes.put(offset, scope.getScopeId());
    return scope;
  }

  /**
   * Finds expanded variable with the name <code>name</code> to be used in the reference to it at
   * <code>offset</code>. Returns null if nothing was found.
   */
  @Nullable
  public String findExpandedVariable(NinjaScopeRegister register, int offset, String name) {
    return findByNameAndOffsetRecursively(register, offset, name, scope -> scope.expandedVariables);
  }

  /**
   * Finds a rule with the name <code>name</code> to be used in the reference to it at <code>offset
   * </code>. Returns null if nothing was found.
   */
  @Nullable
  public NinjaRule findRule(NinjaScopeRegister register, int offset, String name) {
    return findByNameAndOffsetRecursively(register, offset, name, scope -> scope.rules);
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
      NinjaScopeRegister register,
      int offset,
      String name,
      Function<NinjaScope, Map<String, List<Pair<Integer, T>>>> mapSupplier) {
    Pair<Integer, T> currentScopeValue = findByNameAndOffset(offset, name, this, mapSupplier);

    int currentScopeOffset =
        currentScopeValue != null ? Preconditions.checkNotNull(currentScopeValue.getFirst()) : -1;

    // Search in included scopes, which were included after the current scope, so they could
    // override the value, but before the reference offset.
    NavigableMap<Integer, Integer> subMap =
        includedScopes.subMap(currentScopeOffset, false, offset, false);
    // Search in descending order, so that the first found value is the result.
    for (Integer includedScopeId : subMap.descendingMap().values()) {
      T includedValue = register.getById(includedScopeId)
              .findByNameAndOffsetRecursively(register, Integer.MAX_VALUE, name, mapSupplier);
      if (includedValue != null) {
        return includedValue;
      }
    }
    if (currentScopeValue != null) {
      return currentScopeValue.getSecond();
    }
    if (parentScopeId != null) {
      Preconditions.checkNotNull(includePoint);
      // -1 is used in order not to conflict with the current scope.
      return register.getById(parentScopeId).findByNameAndOffsetRecursively(register, includePoint - 1, name, mapSupplier);
    }
    return null;
  }

  /**
   * Finds the variable or rule with the name <code>name</code>, defined in the current scope before
   * the <code>offset</code>. (Ninja allows to re-define the values of rules and variables.)
   */
  @Nullable
  private static <T> Pair<Integer, T> findByNameAndOffset(
      int offset,
      String name,
      NinjaScope scope,
      Function<NinjaScope, Map<String, List<Pair<Integer, T>>>> mapFunction) {
    List<Pair<Integer, T>> pairs = Preconditions.checkNotNull(mapFunction.apply(scope)).get(name);
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
    Pair<Integer, T> pair = pairs.get(idx);
    return Pair.of(pair.getFirst(), pair.getSecond());
  }

  NinjaScope createTargetsScope(NinjaScopeRegister register,
      ImmutableSortedMap<String, List<Pair<Integer, String>>> expandedVariables) {
    NinjaScope scope = register.createChildScope(this, Integer.MAX_VALUE);
    scope.expandedVariables.putAll(expandedVariables);
    return scope;
  }

  public int getScopeId() {
    return scopeId;
  }
}
