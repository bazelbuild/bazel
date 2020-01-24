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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * An object to hold the references to all {@link NinjaScope} scopes for the scope's tree.
 * Can be converted into an immutable object with the same interface fo serialization.
 */
public class NinjaScopeRegister {
  private transient final Supplier<Integer> idGenerator;
  private final NinjaScope mainScope;
  private final SortedMap<Integer, NinjaScope> childScopesMap;
  private final boolean isFrozen;

  /**
   * Default constructor is used during deserialization, and deserialization should create
   * the frozen (immutable) object.
   */
  private NinjaScopeRegister() {
    this(true);
  }

  private NinjaScopeRegister(boolean isFrozen) {
    AtomicInteger atomicInteger = new AtomicInteger(0);
    this.idGenerator = atomicInteger::getAndIncrement;
    mainScope = new NinjaScope(idGenerator.get(), null, null);
    this.childScopesMap = Collections.synchronizedSortedMap(Maps.newTreeMap());
    this.isFrozen = isFrozen;
  }

  public static NinjaScopeRegister create() {
    return new NinjaScopeRegister(false);
  }

  private NinjaScopeRegister(
      NinjaScope mainScope,
      SortedMap<Integer, NinjaScope> childScopesMap) {
    this.idGenerator = () -> 0;
    this.mainScope = mainScope;
    this.childScopesMap = childScopesMap;
    this.isFrozen = true;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    NinjaScopeRegister register = (NinjaScopeRegister) o;
    if (isFrozen != register.isFrozen) {
      return false;
    }
    // Normally, the values of NinjaScopeRegister are not compared to each other;
    // but we need the equals method for the serialization test.
    // that is why for teh test we also want to have "deep" equals to also check the
    // serialization of embedded data.
    if (! mainScope.deepEquals(register.mainScope)) {
      return false;
    }
    if (childScopesMap.size() != register.childScopesMap.size()) {
      return false;
    }
    for (Map.Entry<Integer, NinjaScope> entry : childScopesMap.entrySet()) {
      NinjaScope otherChildScope = register.childScopesMap.get(entry.getKey());
      if (otherChildScope == null || !entry.getValue().deepEquals(otherChildScope)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hash(mainScope, childScopesMap, isFrozen);
  }

  NinjaScope getById(int scopeId) {
    if (mainScope.getScopeId() == scopeId) {
      return mainScope;
    }
    NinjaScope ninjaScope = childScopesMap.get(scopeId);
    Preconditions.checkNotNull(ninjaScope);
    return ninjaScope;
  }

  NinjaScope createChildScope(NinjaScope parentScope, Integer includePoint) {
    Preconditions.checkState(!isFrozen);
    int newId = idGenerator.get();
    NinjaScope ninjaScope = new NinjaScope(newId, parentScope.getScopeId(), includePoint);
    childScopesMap.put(newId, ninjaScope);
    return ninjaScope;
  }

  public NinjaScopeRegister freeze() {
    Preconditions.checkState(!isFrozen);
    ImmutableSortedMap.Builder<Integer, NinjaScope> builder = ImmutableSortedMap.naturalOrder();
    this.childScopesMap.forEach((key, value) -> builder.put(key, value.freeze()));
    return new NinjaScopeRegister(this.mainScope.freeze(), builder.build());
  }

  public NinjaScope getMainScope() {
    return mainScope;
  }

  public Collection<NinjaScope> getScopesByIds(Collection<Integer> scopeIds) {
    return childScopesMap.entrySet().stream()
        .filter(e -> scopeIds.contains(e.getKey()))
        .map(Map.Entry::getValue)
        .collect(Collectors.toList());
  }
}
