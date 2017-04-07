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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A generic, type-safe {@link NestedSet} wrapper for Skylark.
 *
 * <p>The content type of a {@code SkylarkNestedSet} is the intersection of the {@link SkylarkType}
 * of each of its elements. It is an error if this intersection is {@link SkylarkType#BOTTOM}. An
 * empty set has a content type of {@link SkylarkType#TOP}.
 *
 * <p>It is also an error if this type has a non-bottom intersection with {@link SkylarkType#DICT}
 * or {@link SkylarkType#LIST}, unless the set is empty.
 *
 * <p>TODO(bazel-team): Decide whether this restriction is still useful.
 */
@SkylarkModule(
  name = "depset",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "<p>A specialized data structure that supports efficient merge operations and has a defined "
          + "traversal order. Commonly used for accumulating data from transitive dependencies in "
          + "rules and aspects. For more information see <a href=\"../depsets.md\">here</a>."
          + "<p>"
          + "Depsets are not implemented as hash sets and do not support fast membership tests. If "
          + "you need a general set datatype, you can simulate one using a dictionary where all "
          + "keys map to <code>None</code>."
          + "<p>"
          + "Depsets are immutable. They can be created using their "
          + "<a href=\"globals.html#depset\">constructor function</a> and merged or augmented "
          + "using the <code>+</code> operator."
          + "<p>"
          + "The <code>order</code> parameter determines the kind of traversal that is done to "
          + "convert the depset to an iterable. There are four possible values:"
          + "<ul>"
          + "<li><code>\"default\"</code> (formerly <code>\"stable\"</code>): Order is unspecified "
          + "(but deterministic).</li>"
          + "<li><code>\"postorder\"</code> (formerly <code>\"compile\"</code>): A left-to-right "
          + "post-ordering. Precisely, this recursively traverses all children leftmost-first, "
          + "then the direct elements leftmost-first.</li>"
          + "<li><code>\"preorder\"</code> (formerly <code>\"naive_link\"</code>): A left-to-right "
          + "pre-ordering. Precisely, this traverses the direct elements leftmost-first, then "
          + "recursively traverses the children leftmost-first.</li>"
          + "<li><code>\"topological\"</code> (formerly <code>\"link\"</code>): A topological "
          + "ordering from the root down to the leaves. There is no left-to-right guarantee.</li>"
          + "</ul>"
          + "<p>"
          + "Two depsets may only be merged (via <code>+</code> or the <code>union()</code> "
          + "method) if either both depsets have the same order, or one of them has <code>"
          + "\"default\"</code> order. In the latter case the resulting depset's order will be the "
          + "same as the left operand's."
          + "<p>"
          + "Depsets may contain duplicate values but these will be suppressed when iterating "
          + "(using <code>to_list()</code>). Duplicates may interfere with the ordering semantics."
          + "<p>"
          + "The function <code>set()</code> is a deprecated alias for <code>depset()</code>. "
          + "Please update legacy code and use only <code>depset()</code>."
)
@Immutable
public final class SkylarkNestedSet implements SkylarkValue, SkylarkQueryable {

  private final SkylarkType contentType;
  private final NestedSet<?> set;
  @Nullable
  private final List<Object> items;
  @Nullable
  private final List<NestedSet> transitiveItems;

  // Dummy class used to create a documentation for the deprecated `set` type
  // TODO(bazel-team): remove before the end of 2017
  @SkylarkModule(
      name = "set",
      category = SkylarkModuleCategory.BUILTIN,
      doc = "A deprecated alias for <a href=\"depset.html\">depset</a>. "
          + "Please use <a href=\"depset.html\">depset</a> instead. "
          + "If you need a hash set that supports O(1) membership testing "
          + "consider using a <a href=\"dict.html\">dict</a>."
  )
  static final class LegacySet {
    private LegacySet() {}
  }

  public SkylarkNestedSet(Order order, Object item, Location loc) throws EvalException {
    this(order, SkylarkType.TOP, item, loc, null);
  }

  public SkylarkNestedSet(SkylarkNestedSet left, Object right, Location loc) throws EvalException {
    this(left.set.getOrder(), left.contentType, right, loc, left);
  }

  // This is safe because of the type checking
  @SuppressWarnings("unchecked")
  private SkylarkNestedSet(Order order, SkylarkType contentType, Object item, Location loc,
      @Nullable SkylarkNestedSet left) throws EvalException {

    ArrayList<Object> items = new ArrayList<>();
    ArrayList<NestedSet> transitiveItems = new ArrayList<>();
    if (left != null) {
      if (left.items == null) { // SkylarkSet created from native NestedSet
        transitiveItems.add(left.set);
      } else { // Preserving the left-to-right addition order.
        items.addAll(left.items);
        transitiveItems.addAll(left.transitiveItems);
      }
    }
    // Adding the item
    if (item instanceof SkylarkNestedSet) {
      SkylarkNestedSet nestedSet = (SkylarkNestedSet) item;
      if (!nestedSet.isEmpty()) {
        contentType = getTypeAfterInsert(contentType, nestedSet.contentType, loc);
        transitiveItems.add(nestedSet.set);
      }
    } else if (item instanceof SkylarkList) {
      // TODO(bazel-team): we should check ImmutableList here but it screws up genrule at line 43
      for (Object object : (SkylarkList) item) {
        contentType = getTypeAfterInsert(contentType, SkylarkType.of(object.getClass()), loc);
        checkImmutable(object, loc);
        items.add(object);
      }
    } else {
      throw new EvalException(
          loc,
          String.format(
              "cannot union value of type '%s' to a depset", EvalUtils.getDataTypeName(item)));
    }
    this.contentType = Preconditions.checkNotNull(contentType, "type cannot be null");

    // Initializing the real nested set
    NestedSetBuilder<Object> builder = new NestedSetBuilder<>(order);
    builder.addAll(items);
    try {
      for (NestedSet<?> nestedSet : transitiveItems) {
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
    this.set = Preconditions.checkNotNull(set, "depset cannot be null");
    this.items = null;
    this.transitiveItems = null;
  }

  /**
   * A not type safe constructor for SkylarkNestedSet, specifying type as a Java class.
   * It's discouraged to use it unless type generic safety is guaranteed from the caller side.
   */
  public SkylarkNestedSet(Class<?> contentType, NestedSet<?> set) {
    this(SkylarkType.of(contentType), set);
  }

  private static final SkylarkType DICT_LIST_UNION =
      SkylarkType.Union.of(SkylarkType.DICT, SkylarkType.LIST);

  /**
   * Throws EvalException if a type overlaps with DICT or LIST.
   */
  private static void checkTypeNotDictOrList(SkylarkType type, Location loc)
      throws EvalException {
    if (SkylarkType.intersection(DICT_LIST_UNION, type) != SkylarkType.BOTTOM) {
      throw new EvalException(
          loc, String.format("depsets cannot contain items of type '%s'", type));
    }
  }

  /**
   * Returns the intersection of two types, and throws EvalException if the intersection is bottom.
   */
  private static SkylarkType commonNonemptyType(
      SkylarkType depsetType, SkylarkType itemType, Location loc) throws EvalException {
    SkylarkType resultType = SkylarkType.intersection(depsetType, itemType);
    if (resultType == SkylarkType.BOTTOM) {
      throw new EvalException(
          loc,
          String.format(
              "cannot add an item of type '%s' to a depset of '%s'", itemType, depsetType));
    }
    return resultType;
  }

  /**
   * Checks that an item type is allowed in a given set type, and returns the type of a new depset
   * with that item inserted.
   */
  private static SkylarkType getTypeAfterInsert(
      SkylarkType depsetType, SkylarkType itemType, Location loc) throws EvalException {
    checkTypeNotDictOrList(itemType, loc);
    return commonNonemptyType(depsetType, itemType, loc);
  }

  /**
   * Throws EvalException if a given value is mutable.
   */
  private static void checkImmutable(Object o, Location loc) throws EvalException {
    if (!EvalUtils.isImmutable(o)) {
      throw new EvalException(loc, "depsets cannot contain mutable items");
    }
  }

  private void checkHasContentType(Class<?> type) {
    // Empty sets should be SkylarkType.TOP anyway.
    if (!set.isEmpty()) {
      Preconditions.checkArgument(
          contentType.canBeCastTo(type),
          "Expected a depset of '%s' but got a depset of '%s'",
          EvalUtils.getDataTypeNameFromClass(type), contentType);
    }
  }

  /**
   * Returns the embedded {@link NestedSet}, while asserting that its elements all have the given
   * type.
   *
   * <p>If you do not specifically need the {@code NestedSet} and you are going to flatten it
   * anyway, prefer {@link #toCollection} to make your intent clear.
   *
   * @param type a {@link Class} representing the expected type of the contents
   * @return the {@code NestedSet}, with the appropriate generic type
   * @throws IllegalArgumentException if the type does not accurately describe all elements
   */
  // The precondition ensures generic type safety.
  @SuppressWarnings("unchecked")
  public <T> NestedSet<T> getSet(Class<T> type) {
    checkHasContentType(type);
    return (NestedSet<T>) set;
  }

  /**
   * Returns the contents of the set as a {@link Collection}.
   */
  public Collection<Object> toCollection() {
    // Do not remove <Object>: workaround for Java 7 type inference.
    return ImmutableList.<Object>copyOf(set.toCollection());
  }

  /**
   * Returns the contents of the set as a {@link Collection}, asserting that the set type is
   * compatible with {@code T}.
   *
   * @param type a {@link Class} representing the expected type of the contents
   * @throws IllegalArgumentException if the type does not accurately describe all elements
   */
  // The precondition ensures generic type safety.
  @SuppressWarnings("unchecked")
  public <T> Collection<T> toCollection(Class<T> type) {
    checkHasContentType(type);
    return (Collection<T>) toCollection();
  }

  public boolean isEmpty() {
    return set.isEmpty();
  }

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
    Printer.append(buffer, "depset(");
    Printer.printList(buffer, set, "[", ", ", "]", null, quotationMark);
    Order order = getOrder();
    if (order != Order.STABLE_ORDER) {
      Printer.append(buffer, ", order = \"" + order.getSkylarkName() + "\"");
    }
    Printer.append(buffer, ")");
  }

  @Override
  public final boolean containsKey(Object key, Location loc) throws EvalException {
    return (set.toSet().contains(key));
  }
}
