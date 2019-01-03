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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import java.util.Collection;
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
          + "keys map to <code>True</code>."
          + "<p>"
          + "Depsets are immutable. They should be created using their "
          + "<a href=\"globals.html#depset\">constructor function</a> and merged or augmented with "
          + "other depsets via the <code>transitive</code> argument. There are other deprecated "
          + "methods (<code>|</code> and <code>+</code> operators, <code>union</code> method) that "
          + "will eventually go away."
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
          + "Two depsets may only be merged if either both depsets have the same order, or one of "
          + "them has <code>\"default\"</code> order. In the latter case the resulting depset's "
          + "order will be the same as the other's order."
          + "<p>"
          + "Depsets may contain duplicate values but these will be suppressed when iterating "
          + "(using <code>to_list()</code>). Duplicates may interfere with the ordering semantics."
)
@Immutable
@AutoCodec
public final class SkylarkNestedSet implements SkylarkValue, SkylarkQueryable {
  private final SkylarkType contentType;
  private final NestedSet<?> set;
  @Nullable private final ImmutableList<Object> items;
  @Nullable private final ImmutableList<NestedSet<?>> transitiveItems;

  @AutoCodec.VisibleForSerialization
  SkylarkNestedSet(
      SkylarkType contentType,
      NestedSet<?> set,
      ImmutableList<Object> items,
      ImmutableList<NestedSet<?>> transitiveItems) {
    this.contentType = Preconditions.checkNotNull(contentType, "type cannot be null");
    this.set = set;
    this.items = items;
    this.transitiveItems = transitiveItems;
  }

  static SkylarkNestedSet of(
      Order order,
      SkylarkType contentType,
      Object item,
      Location loc,
      @Nullable SkylarkNestedSet left)
      throws EvalException {
    ImmutableList.Builder<Object> itemsBuilder = ImmutableList.builder();
    ImmutableList.Builder<NestedSet<?>> transitiveItemsBuilder = ImmutableList.builder();
    if (left != null) {
      if (left.items == null) { // SkylarkSet created from native NestedSet
        transitiveItemsBuilder.add(left.set);
      } else { // Preserving the left-to-right addition order.
        itemsBuilder.addAll(left.items);
        transitiveItemsBuilder.addAll(left.transitiveItems);
      }
    }
    // Adding the item
    if (item instanceof SkylarkNestedSet) {
      SkylarkNestedSet nestedSet = (SkylarkNestedSet) item;
      if (!nestedSet.isEmpty()) {
        contentType = getTypeAfterInsert(
            contentType, nestedSet.contentType, /*lastInsertedType=*/ null, loc);
        transitiveItemsBuilder.add(nestedSet.set);
      }
    } else if (item instanceof SkylarkList) {
      SkylarkType lastInsertedType = null;
      // TODO(bazel-team): we should check ImmutableList here but it screws up genrule at line 43
      for (Object object : (SkylarkList) item) {
        SkylarkType elemType = SkylarkType.of(object);
        contentType = getTypeAfterInsert(contentType, elemType, lastInsertedType, loc);
        lastInsertedType = elemType;
        checkImmutable(object, loc);
        itemsBuilder.add(object);
      }
    } else {
      throw new EvalException(
          loc,
          String.format(
              "cannot union value of type '%s' to a depset", EvalUtils.getDataTypeName(item)));
    }
    ImmutableList<Object> items = itemsBuilder.build();
    ImmutableList<NestedSet<?>> transitiveItems = transitiveItemsBuilder.build();
    // Initializing the real nested set
    NestedSetBuilder<Object> builder = new NestedSetBuilder<>(order);
    builder.addAll(items);
    try {
      for (NestedSet<?> nestedSet : transitiveItems) {
        builder.addTransitive(nestedSet);
      }
    } catch (IllegalArgumentException e) {
      // Order mismatch between item and builder.
      throw new EvalException(loc, e.getMessage());
    }
    return new SkylarkNestedSet(contentType, builder.build(), items, transitiveItems);
  }

  public static SkylarkNestedSet of(Order order, Object item, Location loc) throws EvalException {
    return of(order, SkylarkType.TOP, item, loc, null);
  }

  public static SkylarkNestedSet of(SkylarkNestedSet left, Object right, Location loc)
      throws EvalException {
    return of(left.set.getOrder(), left.contentType, right, loc, left);
  }

  /**
   * Returns a type safe SkylarkNestedSet. Use this instead of the constructor if possible.
   */
  public static <T> SkylarkNestedSet of(SkylarkType contentType, NestedSet<T> set) {
    return new SkylarkNestedSet(contentType, set, null, null);
  }

  /**
   * Returns a type safe SkylarkNestedSet. Use this instead of the constructor if possible.
   */
  public static <T> SkylarkNestedSet of(Class<T> contentType, NestedSet<T> set) {
    return of(SkylarkType.of(contentType), set);
  }

  private static final SkylarkType DICT_LIST_UNION =
      SkylarkType.Union.of(SkylarkType.DICT, SkylarkType.LIST);

  /**
   * Checks that an item type is allowed in a given set type, and returns the type of a new depset
   * with that item inserted.
   */
  private static SkylarkType getTypeAfterInsert(
      SkylarkType depsetType, SkylarkType itemType, SkylarkType lastInsertedType, Location loc)
      throws EvalException {
    if (lastInsertedType != null && lastInsertedType.equals(itemType)) {
      // Fast path, type shouldn't have changed, so no need to check.
      // TODO(bazel-team): Make skylark type checking less expensive.
      return depsetType;
    }

    // Check not dict or list.
    if (SkylarkType.intersection(DICT_LIST_UNION, itemType) != SkylarkType.BOTTOM) {
      throw new EvalException(
          loc, String.format("depsets cannot contain items of type '%s'", itemType));
    }

    SkylarkType resultType = SkylarkType.intersection(depsetType, itemType);

    // New depset type should follow the following rules:
    // 1. Only empty depsets may be of type TOP.
    // 2. If the previous depset type fully contains the new item type, then the depset type is
    //    retained.
    // 3. If the item type fully contains the old depset type, then the depset type becomes the
    //    item type. (The depset type becomes less strict.)
    // 4. Otherwise, the insert is invalid.
    if (depsetType == SkylarkType.TOP) {
      return resultType;
    } else if (resultType.equals(itemType)) {
      return depsetType;
    } else if (resultType.equals(depsetType)) {
      return itemType;
    } else {
      throw new EvalException(
          loc,
          String.format(
              "cannot add an item of type '%s' to a depset of '%s'", itemType, depsetType));
    }
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
    return ImmutableList.copyOf(set.toCollection());
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
  public void repr(SkylarkPrinter printer) {
    printer.append("depset(");
    printer.printList(set, "[", ", ", "]", null);
    Order order = getOrder();
    if (order != Order.STABLE_ORDER) {
      printer.append(", order = ");
      printer.repr(order.getSkylarkName());
    }
    printer.append(")");
  }

  @Override
  public final boolean containsKey(Object key, Location loc) throws EvalException {
    return (set.toList().contains(key));
  }

  @SkylarkCallable(
      name = "union",
      doc =
          "<i>(Deprecated)</i> Returns a new <a href=\"depset.html\">depset</a> that is the merge "
              + "of the given depset and <code>new_elements</code>. Use the "
              + "<code>transitive</code> constructor argument instead.",
      parameters = {
          @Param(name = "new_elements", type = Object.class, doc = "The elements to be added.")
      },
      useLocation = true,
      useEnvironment = true
  )
  public SkylarkNestedSet union(Object newElements, Location loc, Environment env)
      throws EvalException {
    if (env.getSemantics().incompatibleDepsetUnion()) {
      throw new EvalException(
          loc,
          "depset method `.union` has been removed. See "
              + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
              + "recommendations. Use --incompatible_depset_union=false "
              + "to temporarily disable this check.");
    }
    // newElements' type is Object because of the polymorphism on unioning two
    // SkylarkNestedSets versus a set and another kind of iterable.
    // Can't use EvalUtils#toIterable since that would discard this information.
    return SkylarkNestedSet.of(this, newElements, loc);
  }

  @SkylarkCallable(
      name = "to_list",
      doc =
          "Returns a list of the elements, without duplicates, in the depset's traversal order. "
              + "Note that order is unspecified (but deterministic) for elements that were added "
              + "more than once to the depset. Order is also unspecified for <code>\"default\""
              + "</code>-ordered depsets, and for elements of child depsets whose order differs "
              + "from that of the parent depset. The list is a copy; modifying it has no effect "
              + "on the depset and vice versa.",
      useEnvironment = true
  )
  public MutableList<Object> toList(Environment env) {
    return MutableList.copyOf(env, this.toCollection());
  }

  /**
   * Create a {@link Builder} with specified order.
   *
   * <p>The {@code Builder} will use {@code location} to report errors.
   */
  public static Builder builder(Order order, Location location) {
    return new Builder(order, location);
  }

  /**
   * Builder for {@link SkylarkNestedSet}.
   *
   * <p>Use this to construct typesafe Skylark nested sets (depsets).
   * Encapsulates content type checking logic.
   */
  public static final class Builder {

    private final Order order;
    private final NestedSetBuilder<Object> builder;
    /** Location for error messages */
    private final Location location;
    private SkylarkType contentType = SkylarkType.TOP;
    private SkylarkType lastInsertedType = null;

    private Builder(Order order, Location location) {
      this.order = order;
      this.location = location;
      this.builder = new NestedSetBuilder<>(order);
    }

    /**
     * Add a direct element, checking its type to be compatible to already added
     * elements and transitive sets.
     */
    public Builder addDirect(Object direct) throws EvalException {
      SkylarkType elemType = SkylarkType.of(direct);
      contentType = getTypeAfterInsert(contentType, elemType, lastInsertedType, location);
      lastInsertedType = elemType;
      builder.add(direct);
      return this;
    }

    /**
     * Add a transitive set, checking its content type to be compatible to already added
     * elements and transitive sets.
     */
    public Builder addTransitive(SkylarkNestedSet transitive) throws EvalException {
      if (transitive.isEmpty()) {
        return this;
      }

      contentType = getTypeAfterInsert(
          contentType, transitive.getContentType(), lastInsertedType, this.location);
      lastInsertedType = transitive.getContentType();

      if (!order.isCompatible(transitive.getOrder())) {
        throw new EvalException(location,
            String.format("Order '%s' is incompatible with order '%s'",
                          order.getSkylarkName(), transitive.getOrder().getSkylarkName()));
      }
      builder.addTransitive(transitive.getSet(Object.class));
      return this;
    }

    public SkylarkNestedSet build() {
      return new SkylarkNestedSet(contentType, builder.build(), null, null);
    }
  }
}
