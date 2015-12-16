// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.util.Preconditions;

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
    doc = "A language built-in type that supports sets. "
        + "Sets can be created using the <a href=\"globals.html#set\">set</a> function, and "
        + "they support the <code>|</code> operator to extend the set with more elements or "
        + "to nest other sets inside of it. Examples:<br>"
        + "<pre class=language-python>s = set([1, 2])\n"
        + "s = s | [3]           # s == {1, 2, 3}\n"
        + "s = s | set([4, 5])   # s == {1, 2, 3, {4, 5}}\n"
        + "other = set([\"a\", \"b\", \"c\"], order=\"compile\")</pre>"
        + "Note that in these examples <code>{..}</code> is not a valid literal to create sets. "
        + "Sets have a fixed generic type, so <code>set([1]) + [\"a\"]</code> or "
        + "<code>set([1]) + set([\"a\"])</code> results in an error.<br>"
        + "Elements in a set can neither be mutable or be of type <code>list</code>, "
        + "<code>struct</code> or <code>dict</code>.<br>"
        + "When aggregating data from providers, sets can take significantly less memory than "
        + "other types as they support nesting, that is, their subsets are shared in memory.<br>"
        + "Every set has an <code>order</code> parameter which determines the iteration order. "
        + "There are four possible values:"
        + "<ul><li><code>compile</code>: Defines a left-to-right post-ordering where child "
        + "elements come after those of nested sets (parent-last). For example, "
        + "<code>{1, 2, 3, {4, 5}}</code> leads to <code>4 5 1 2 3</code>. Left-to-right order "
        + "is preserved for both the child elements and the references to nested sets.</li>"
        + "<li><code>stable</code>: Same behavior as <code>compile</code>.</li>"
        + "<li><code>link</code>: Defines a variation of left-to-right pre-ordering, i.e. "
        + "<code>{1, 2, 3, {4, 5}}</code> leads to <code>1 2 3 4 5</code>. "
        + "This ordering enforces that elements of the set always come before elements of "
        + "nested sets (parent-first), which may lead to situations where left-to-right "
        + "order cannot be preserved (<a href=\"https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/collect/nestedset/LinkOrderExpander.java#L56\">Example</a>)."
        + "</li>"
        + "<li><code>naive_link</code>: Defines \"naive\" left-to-right pre-ordering "
        + "(parent-first), i.e. <code>{1, 2, 3, {4, 5}}</code> leads to <code>1 2 3 4 5</code>. "
        + "Unlike <code>link</code> ordering, it will sacrifice the parent-first property in "
        + "order to uphold left-to-right order in cases where both properties cannot be "
        + "guaranteed (<a href=\"https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/lib/collect/nestedset/NaiveLinkOrderExpander.java#L26\">Example</a>)."
        + "</li></ul>"
        + "Except for <code>stable</code>, the above values are incompatible with each other. "
        + "Consequently, two sets can only be merged via the <code>|</code> operator or via "
        + "<code>union()</code> if either both sets have the same <code>order</code> or one of "
        + "the sets has <code>stable</code> order. In the latter case the iteration order will be "
        + "determined by the outer set, thus ignoring the <code>order</code> parameter of "
        + "nested sets.")
@Immutable
public final class SkylarkNestedSet implements Iterable<Object>, SkylarkValue {

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
      throw new EvalException(
          loc,
          String.format("cannot add value of type '%s' to a set", EvalUtils.getDataTypeName(item)));
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
      throw new EvalException(
          loc, String.format("sets cannot contain items of type '%s'", itemType));
    }
    if (!EvalUtils.isImmutable(itemType.getType())) {
      throw new EvalException(
          loc, String.format("sets cannot contain items of type '%s' (mutable type)", itemType));
    }
    SkylarkType newType = SkylarkType.intersection(builderType, itemType);
    if (newType == SkylarkType.BOTTOM) {
      throw new EvalException(
          loc,
          String.format("cannot add an item of type '%s' to a set of '%s'", itemType, builderType));
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
        String.format("Expected a set of '%s' but got a set of '%s'",
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
    // Do not remove <Object>: workaround for Java 7 type inference.
    return ImmutableList.<Object>copyOf(set.toCollection());
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
    return Printer.repr(this);
  }

  public Order getOrder() {
    return set.getOrder();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.append(buffer, "set(");
    Printer.printList(buffer, this, "[", ", ", "]", null, quotationMark);
    Order order = getOrder();
    if (order != Order.STABLE_ORDER) {
      Printer.append(buffer, ", order = \"" + order.getName() + "\"");
    }
    Printer.append(buffer, ")");
  }
}
