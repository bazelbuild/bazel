// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.syntax;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A generic type safe NestedSet wrapper for Skylark.
 */
@SkylarkModule(name = "set",
    doc = "A language built-in type that supports (nested) sets. "
        + "Sets can be created using the global <code>set</code> function, and they "
        + "support the <code>+</code> operator to extend the set with more elements or to nest "
        + "other sets inside of it. Examples:<br>"
        + "<pre class=language-python>s = set([1, 2])\n"
        + "s += [3]           # s == {1, 2, 3}\n"
        + "s += set([4, 5])   # s == {1, 2, 3, {4, 5}}</pre>"
        + "Note that in these examples <code>{..}</code> is not a valid literal to create sets. "
        + "Sets have a fixed generic type, so <code>set([1]) + [\"a\"]</code> or "
        + "<code>set([1]) + set([\"a\"])</code> results in an error.")
@Immutable
public final class SkylarkNestedSet implements Iterable<Object> {

  private final SkylarkType contentType;
  @Nullable private final List<Object> items;
  @Nullable private final List<NestedSet<Object>> transitiveItems;
  private final NestedSet<?> set;

  public SkylarkNestedSet(Order order, Object item, Location loc) throws EvalException {
    this(order, SkylarkType.TOP, item, loc, new ArrayList<Object>(),
        new ArrayList<NestedSet<Object>>());
  }

  public SkylarkNestedSet(SkylarkNestedSet left, Object right, Location loc) throws EvalException {
    this(left.set.getOrder(), left.contentType, right, loc,
        new ArrayList<Object>(checkItems(left.items, loc)),
        new ArrayList<NestedSet<Object>>(checkItems(left.transitiveItems, loc)));
  }

  private static <T> T checkItems(T items, Location loc) throws EvalException {
    // SkylarkNestedSets created directly from ordinary NestedSets (those were created in a
    // native rule) don't have directly accessible items and transitiveItems, so we cannot
    // add more elements to them.
    if (items == null) {
      throw new EvalException(loc, "Cannot add more elements to this set. Sets created in "
          + "native rules cannot be left side operands of the + operator.");
    }
    return items;
  }

  // This is safe because of the type checking
  @SuppressWarnings("unchecked")
  private SkylarkNestedSet(Order order, SkylarkType contentType, Object item, Location loc,
      List<Object> items, List<NestedSet<Object>> transitiveItems) throws EvalException {

    // Adding the item
    if (item instanceof SkylarkNestedSet) {
      SkylarkNestedSet nestedSet = (SkylarkNestedSet) item;
      if (!nestedSet.isEmpty()) {
        contentType = checkType(contentType, nestedSet.contentType, loc);
        transitiveItems.add((NestedSet<Object>) nestedSet.set);
      }
    } else if (item instanceof SkylarkList) {
      // TODO(bazel-team): we should check ImmutableList here but it screws up genrule at line 43
      for (Object object : (SkylarkList) item) {
        contentType = checkType(contentType, SkylarkType.of(object.getClass()), loc);
        items.add(object);
      }
    } else {
      throw new EvalException(loc,
          String.format("cannot add '%s'-s to nested sets", EvalUtils.getDataTypeName(item)));
    }
    this.contentType = Preconditions.checkNotNull(contentType, "type cannot be null");

    // Initializing the real nested set
    NestedSetBuilder<Object> builder = new NestedSetBuilder<>(order);
    builder.addAll(items);
    try {
      for (NestedSet<Object> nestedSet : transitiveItems) {
        builder.addTransitive(nestedSet);
      }
    } catch (IllegalStateException e) {
      throw new EvalException(loc, e.getMessage());
    }
    this.set = builder.build();
    this.items = ImmutableList.copyOf(items);
    this.transitiveItems = ImmutableList.copyOf(transitiveItems);
  }

  /**
   * Returns a type safe SkylarkNestedSet. Use this instead of the constructor if possible.
   */
  public static <T> SkylarkNestedSet of(SkylarkType contentType, NestedSet<T> set) {
    return new SkylarkNestedSet(contentType, set);
  }

  /**
   * Returns a type safe SkylarkNestedSet. Use this instead of the constructor if possible.
   */
  public static <T> SkylarkNestedSet of(Class<T> contentType, NestedSet<T> set) {
    return of(SkylarkType.of(contentType), set);
  }

  /**
   * A not type safe constructor for SkylarkNestedSet. It's discouraged to use it unless type
   * generic safety is guaranteed from the caller side.
   */
  SkylarkNestedSet(SkylarkType contentType, NestedSet<?> set) {
    // This is here for the sake of FuncallExpression.
    this.contentType = Preconditions.checkNotNull(contentType, "type cannot be null");
    this.set = Preconditions.checkNotNull(set, "set cannot be null");
    this.items = null;
    this.transitiveItems = null;
  }

  /**
   * A not type safe constructor for SkylarkNestedSet, specifying type as a Java class.
   * It's discouraged to use it unless type generic safety is guaranteed from the caller side.
   */
  SkylarkNestedSet(Class<?> contentType, NestedSet<?> set) {
    this(SkylarkType.of(contentType), set);
  }

  private static SkylarkType checkType(SkylarkType builderType, SkylarkType itemType, Location loc)
      throws EvalException {
    if (SkylarkType.intersection(
        SkylarkType.Union.of(SkylarkType.MAP, SkylarkType.LIST, SkylarkType.STRUCT),
        itemType) != SkylarkType.BOTTOM) {
      throw new EvalException(loc, String.format("nested set item is composite (type of %s)",
              itemType));
    }
    if (!EvalUtils.isSkylarkImmutable(itemType.getType())) {
      throw new EvalException(loc, String.format("nested set item is not immutable (type of %s)",
              itemType));
    }
    SkylarkType newType = SkylarkType.intersection(builderType, itemType);
    if (newType == SkylarkType.BOTTOM) {
      throw new EvalException(loc, String.format(
          "cannot add an item of type %s to a nested %s", itemType, builderType));
    }
    return newType;
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
    Preconditions.checkArgument(contentType.canBeCastTo(type),
        String.format("Expected a set of %ss but got a set of %ss",
            EvalUtils.getDataTypeNameFromClass(type),
            contentType));
    return (NestedSet<T>) set;
  }

  public Set<?> expandedSet() {
    return set.toSet();
  }

  // For some reason this cast is unsafe in Java
  @SuppressWarnings("unchecked")
  @Override
  public Iterator<Object> iterator() {
    return (Iterator<Object>) set.iterator();
  }

  public Collection<Object> toCollection() {
    return ImmutableList.copyOf(set.toCollection());
  }

  public boolean isEmpty() {
    return set.isEmpty();
  }

  @VisibleForTesting
  public SkylarkType getContentType() {
    return contentType;
  }

  @Override
  public String toString() {
    return EvalUtils.prettyPrintValue(this);
  }

  /**
   * Parse the string as a set order.
   */
  public static Order parseOrder(String s, Location loc) throws EvalException {
    // Keep in sync with orderString
    if (s == null || s.equals("stable")) {
      return Order.STABLE_ORDER;
    } else if (s.equals("compile")) {
      return Order.COMPILE_ORDER;
    } else if (s.equals("link")) {
      return Order.LINK_ORDER;
    } else if (s.equals("naive_link")) {
      return Order.NAIVE_LINK_ORDER;
    } else {
      throw new EvalException(loc, "Invalid order: " + s);
    }
  }

  /**
   * Get the order as a string.
   */
  public static String orderString(Order order) {
    // Keep in sync with parseOrder
    switch (order) {
      case STABLE_ORDER: return "stable";
      case COMPILE_ORDER: return "compile";
      case LINK_ORDER: return "link";
      case NAIVE_LINK_ORDER: return "naive_link";
      default: throw new IllegalStateException("unknown order: " + order);
    }
  }

  public Order getOrder() {
    return set.getOrder();
  }
}
