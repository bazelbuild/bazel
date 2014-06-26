// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

// TODO(bazel-team): Consider enabling the += operator for NestedSets and use NestedSet.Builder
// instead of a constructor with a list. This only would work with linear typing.
// Also, right now nset is a special function added in SkylarkruleImplementationFunctions
// but maybe it should be a proper built in language item.
/**
 * A generic type safe NestedSet wrapper for Skylark.
 */
@SkylarkBuiltin(name = "nset", doc = "An efficient nested set for Skylark")
@Immutable
public class SkylarkNestedSet implements Iterable<Object> {

  private final Class<?> genericType;

  private final NestedSet<?> set;

  /**
   * Returns a type safe SkylarkNestedSet. Use this instead of the constructor if possible.
   */
  public static <T> SkylarkNestedSet of(Class<T> genericType, NestedSet<T> set) {
    return new SkylarkNestedSet(genericType, set);
  }

  /**
   * A not type safe constructor for SkylarkNestedSet. It's discouraged to use it
   * unless type generic safety is guaranteed from the caller side.
   */
  SkylarkNestedSet(Class<?> genericType, NestedSet<?> set) {
    // This is here for the sake of FuncallExcpression.
    this.genericType = Preconditions.checkNotNull(genericType, "type cannot be null");
    this.set = Preconditions.checkNotNull(set, "set cannot be null");
  }

  /**
   * Creates a SkylarkNestedSet using the given order and items.
   */
  public SkylarkNestedSet(Order order, Iterable<Object> items) throws IllegalArgumentException {
    Preconditions.checkNotNull(items, "items is null");
    NestedSetBuilder<Object> builder = new NestedSetBuilder<>(order);
    Class<?> type = null;
    for (Object item : items) {
      Preconditions.checkNotNull(item, "item is null");
      // This should never happen since FuncallExpression takes care of this issue.
      // But it's better to be extra safe than debugging ClassCastExceptions.
      Preconditions.checkArgument(!(item instanceof NestedSet<?>),
          "item is a NestedSet");
      if (item instanceof SkylarkNestedSet) {
        SkylarkNestedSet skylarkNestedSet = (SkylarkNestedSet) item;
        type = checkType(type, skylarkNestedSet.genericType);
        builder.addTransitive(skylarkNestedSet.getSet(Object.class));
      } else {
        type = checkType(type, item.getClass());
        builder.add(item);
      }
    }
    if (type == null) {
      // Empty set
      type = Object.class;
    }
    set = builder.build();
    genericType = Preconditions.checkNotNull(type, "type is null");
  }

  private Class<?> checkType(Class<?> builderType, Class<?> itemType) {
    if (Map.class.isAssignableFrom(itemType) || List.class.isAssignableFrom(itemType)) {
      throw new IllegalArgumentException(String.format(
          "nested set item is a collection (type of %s)",
          EvalUtils.getDataTypeNameFromClass(itemType)));
    }
    if (!itemType.isAnnotationPresent(Immutable.class)
        && !itemType.equals(String.class) && !itemType.equals(Integer.class)) {
      throw new IllegalArgumentException(String.format(
          "nested set item is not immutable (type of %s)",
          EvalUtils.getDataTypeNameFromClass(itemType)));
    }
    if (builderType == null) {
      return itemType;
    }
    if (!builderType.equals(itemType)) {
      throw new IllegalArgumentException(String.format(
          "nested set item is type of %s but the nested set accepts only %s-s",
          EvalUtils.getDataTypeNameFromClass(itemType),
          EvalUtils.getDataTypeNameFromClass(builderType)));
    }
    return builderType;
  }

  /**
   * Returns the NestedSet embedded in this SkylarkNestedSet if it is of the parameter type.
   */
  // The precondition ensures generic type safety
  @SuppressWarnings("unchecked")
  public <T> NestedSet<T> getSet(Class<T> type) {
    // Empty sets don't need have to have a type since they don't have items
    if (set.isEmpty()) {
      return (NestedSet<T>) set;
    }
    Preconditions.checkArgument(type.isAssignableFrom(genericType),
        String.format("Expected %s as a type but got %s",
            EvalUtils.getDataTypeNameFromClass(type),
            EvalUtils.getDataTypeNameFromClass(genericType)));
    return (NestedSet<T>) set;
  }

  // For some reason this cast is unsafe in Java
  @SuppressWarnings("unchecked")
  @Override
  public Iterator<Object> iterator() {
    return (Iterator<Object>) set.iterator();
  }

  @SkylarkCallable(doc = "Flattens this nested set of file to a list.")
  public Collection<?> toCollection() {
    return set.toCollection();
  }

  @SkylarkCallable(doc = "Returns true if this file set is empty.")
  public boolean isEmpty() {
    return set.isEmpty();
  }

  @VisibleForTesting
  public Class<?> getGenericTypeInfo() {
    return genericType;
  }
}
