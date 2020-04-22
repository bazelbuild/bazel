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
import com.google.devtools.build.lib.collect.nestedset.NestedSet.NestedSetDepthException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A Depset is a Starlark value that wraps a {@link NestedSet}.
 *
 * <p>A NestedSet has a type parameter that describes, at compile time, the elements of the set. By
 * contrast, a Depset has a value, {@link #getContentType}, that describes the elements during
 * execution. This type symbol permits the element type of a Depset value to be queried, after the
 * type parameter has been erased, without visiting each element of the often-vast data structure.
 *
 * <p>The content type of a non-empty {@code Depset} is determined by its first element. All
 * elements must have the same type. An empty depset has type {@code SkylarkType.TOP}, and may be
 * combined with any other depset.
 */
// TODO(adonovan): move to lib.packages, as this is a Bazelism. Requires:
// - moving the function to StarlarkLibrary.COMMON.
// - making SkylarkType.getGenericArgType extensible somehow
// - relaxing StarlarkThread.checkStateEquals (or defining Depset.equals)
@SkylarkModule(
    name = "depset",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "<p>A specialized data structure that supports efficient merge operations and has a"
            + " defined traversal order. Commonly used for accumulating data from transitive"
            + " dependencies in rules and aspects. For more information see <a"
            + " href=\"../depsets.md\">here</a>."
            + " <p>The elements of a depset must be hashable and all of the same type (as"
            + " defined by the built-in type(x) function), but depsets are not simply"
            + " hash sets and do not support fast membership tests."
            + " If you need a general set datatype, you can"
            + " simulate one using a dictionary where all keys map to <code>True</code>."
            + "<p>Depsets are immutable. They should be created using their <a"
            + " href=\"globals.html#depset\">constructor function</a> and merged or augmented with"
            + " other depsets via the <code>transitive</code> argument. "
            + "<p>The <code>order</code> parameter determines the"
            + " kind of traversal that is done to convert the depset to an iterable. There are"
            + " four possible values:"
            + "<ul><li><code>\"default\"</code> (formerly"
            + " <code>\"stable\"</code>): Order is unspecified (but"
            + " deterministic).</li>"
            + "<li><code>\"postorder\"</code> (formerly"
            + " <code>\"compile\"</code>): A left-to-right post-ordering. Precisely, this"
            + " recursively traverses all children leftmost-first, then the direct elements"
            + " leftmost-first.</li>"
            + "<li><code>\"preorder\"</code> (formerly"
            + " <code>\"naive_link\"</code>): A left-to-right pre-ordering. Precisely, this"
            + " traverses the direct elements leftmost-first, then recursively traverses the"
            + " children leftmost-first.</li>"
            + "<li><code>\"topological\"</code> (formerly"
            + " <code>\"link\"</code>): A topological ordering from the root down to the leaves."
            + " There is no left-to-right guarantee.</li>"
            + "</ul>"
            + "<p>Two depsets may only be merged if"
            + " either both depsets have the same order, or one of them has"
            + " <code>\"default\"</code> order. In the latter case the resulting depset's order"
            + " will be the same as the other's order."
            + "<p>Depsets may contain duplicate values but"
            + " these will be suppressed when iterating (using <code>to_list()</code>). Duplicates"
            + " may interfere with the ordering semantics.")
@Immutable
@AutoCodec
public final class Depset implements StarlarkValue {
  private final SkylarkType contentType;
  private final NestedSet<?> set;
  @Nullable private final ImmutableList<Object> items; // TODO(laurentlb): Delete field.
  @Nullable private final ImmutableList<NestedSet<?>> transitiveItems;

  @AutoCodec.VisibleForSerialization
  Depset(
      SkylarkType contentType,
      NestedSet<?> set,
      ImmutableList<Object> items,
      ImmutableList<NestedSet<?>> transitiveItems) {
    this.contentType = Preconditions.checkNotNull(contentType, "type cannot be null");
    this.set = set;
    this.items = items;
    this.transitiveItems = transitiveItems;
  }

  // TODO(laurentlb): Remove the left argument once `unionOf` is deleted.
  private static Depset create(
      Order order, SkylarkType contentType, Object item, @Nullable Depset left)
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
    if (item instanceof Depset) {
      Depset nestedSet = (Depset) item;
      if (!nestedSet.isEmpty()) {
        contentType = checkType(contentType, nestedSet.contentType);
        transitiveItemsBuilder.add(nestedSet.set);
      }
    } else if (item instanceof Sequence) {
      for (Object x : (Sequence) item) {
        checkElement(x, /*strict=*/ true);
        SkylarkType xt = SkylarkType.of(x);
        contentType = checkType(contentType, xt);
        itemsBuilder.add(x);
      }
    } else {
      throw Starlark.errorf(
          "cannot union value of type '%s' to a depset", EvalUtils.getDataTypeName(item));
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
      throw Starlark.errorf("%s", e.getMessage());
    }
    return new Depset(contentType, builder.build(), items, transitiveItems);
  }

  private static void checkElement(Object x, boolean strict) throws EvalException {
    // Historically the requirement for a depset element was isImmutable(x).
    // However, this check is neither necessary not sufficient.
    // It is unnecessary because elements need only be hashable,
    // as with dicts, whose keys may be mutable so long as mutations
    // don't affect the hash code. (Elements of a NestedSet must be
    // hashable because a hash-based set is used to de-duplicate
    // elements during iteration.)
    // And it is insufficient because some values are immutable
    // but not Starlark-hashable, such as frozen lists.
    // NestedSet calls its hashCode method regardless.
    //
    // TODO(adonovan): use this check instead:
    //   EvalUtils.checkHashable(x);
    // and delete the StarlarkValue.isImmutable and EvalUtils.isImmutable.
    // Unfortunately this is a breaking change because some users
    // construct depsets whose elements contain lists of strings,
    // which are Starlark-unhashable even if frozen.
    // TODO(adonovan): also remove StarlarkList.hashCode.
    if (strict && !EvalUtils.isImmutable(x)) {
      throw Starlark.errorf("depset elements must not be mutable values");
    }

    // Even the looser regime forbids the top-level class to be list or dict.
    if (x instanceof StarlarkList || x instanceof Dict) {
      throw Starlark.errorf(
          "depsets cannot contain items of type '%s'", EvalUtils.getDataTypeName(x));
    }
  }

  // implementation of deprecated depset(x) constructor
  static Depset legacyOf(Order order, Object items) throws EvalException {
    // TODO(adonovan): rethink this API. TOP is a pessimistic type for item, and it's wrong
    // (should be BOTTOM) if items is an empty Depset or Sequence.
    return create(order, SkylarkType.TOP, items, null);
  }

  // TODO(laurentlb): Delete the method. It's used only in tests.
  static Depset unionOf(Depset left, Object right) throws EvalException {
    return create(left.set.getOrder(), left.contentType, right, left);
  }

  /**
   * Returns a Depset that wraps the specified NestedSet.
   *
   * <p>This operation is type-safe only if the specified element type is appropriate for every
   * element of the set.
   */
  // TODO(adonovan): enforce that we never construct a Depset with a StarlarkType
  // that represents a non-Starlark type (e.g. NestedSet<PathFragment>).
  // One way to do that is to disallow constructing StarlarkTypes for classes
  // that would fail Starlark.valid; however remains the problem that
  // Object.class means "any Starlark value" but in fact allows any Java value.
  //
  // TODO(adonovan): it is possible to create an empty depset with a contentType other than EMPTY.
  // The union operation will fail if it's combined with another depset of incompatible contentType.
  // Options:
  // - prohibit or ignore a non-EMPTY contentType when passed an empty NestedSet
  // - continue to allow empty depsets to be distinguished by their nominal contentTypes for
  //   union purposes, but allow casting them to NestedSet<T> for arbitrary T.
  // - distinguish them for both union and casting, i.e. replace set.isEmpty() with a check for the
  // empty type.
  public static <T> Depset of(SkylarkType contentType, NestedSet<T> set) {
    return new Depset(contentType, set, null, null);
  }

  /**
   * Checks that an item type is allowed in a given set type, and returns the type of a new depset
   * with that item inserted.
   */
  private static SkylarkType checkType(SkylarkType depsetType, SkylarkType itemType)
      throws EvalException {
    // An initially empty depset takes its type from the first element added.
    // Otherwise, the types of the item and depset must match exactly.
    //
    // TODO(adonovan): why is the empty depset TOP, not BOTTOM?
    // T ^ TOP == TOP, whereas T ^ BOTTOM == T.
    // This can't be changed without breaking callers of getContentType who
    // expect to see TOP. Maybe this is minor, but it at least would require
    // changes to EvalUtils#getDataTypeName so that it
    // can continue to print "depset of Objects" instead of "depset of EmptyTypes".
    // Better yet, break the behavior and change it to "empty depset".
    if (depsetType == SkylarkType.TOP || depsetType.equals(itemType)) {
      return itemType;
    }
    throw Starlark.errorf(
        "cannot add an item of type '%s' to a depset of '%s'", itemType, depsetType);
  }

  /**
   * Returns the embedded {@link NestedSet}, first asserting that its elements are instances of the
   * given class. Only the top-level class is verified.
   *
   * <p>If you do not specifically need the {@code NestedSet} and you are going to flatten it
   * anyway, prefer {@link #toCollection} to make your intent clear.
   *
   * @param type a {@link Class} representing the expected type of the contents
   * @return the {@code NestedSet}, with the appropriate generic type
   * @throws TypeException if the type does not accurately describe all elements
   */
  public <T> NestedSet<T> getSet(Class<T> type) throws TypeException {
    if (!set.isEmpty() && !contentType.canBeCastTo(type)) {
      throw new TypeException(
          String.format(
              "got a depset of '%s', expected a depset of '%s'",
              contentType, EvalUtils.getDataTypeNameFromClass(type)));
    }
    @SuppressWarnings("unchecked")
    NestedSet<T> res = (NestedSet<T>) set;
    return res;
  }

  /**
   * Returns the embedded {@link NestedSet} without asserting the type of its elements---and thus
   * cannot fail. To validate the type of elements in the set, call {@link #getSet(Class)} instead.
   */
  public NestedSet<?> getSet() {
    return set;
  }

  // TODO(adonovan): rename these toCollection methods toList.

  /**
   * Returns an ImmutableList containing the set elements, asserting that each element is an
   * instance of class {@code type}. Requires traversing the entire graph of the underlying
   * NestedSet.
   *
   * @param type a {@link Class} representing the expected type of the contents
   * @throws TypeException if the type does not accurately describe all elements
   */
  public <T> ImmutableList<T> toCollection(Class<T> type) throws TypeException {
    return getSet(type).toList();
  }

  /**
   * Returns an ImmutableList containing the set elements. Requires traversing the entire graph of
   * the underlying NestedSet.
   */
  public ImmutableList<?> toCollection() {
    return set.toList();
  }

  /**
   * Casts a non-null Starlark value {@code x} to a {@code Depset} and returns its {@code
   * NestedSet<T>}, after checking that all elements are instances of {@code type}. On error, it
   * throws an EvalException whose message includes {@code what}, ideally a string literal, as a
   * description of the role of {@code x}.
   */
  public static <T> NestedSet<T> cast(Object x, Class<T> type, String what) throws EvalException {
    if (!(x instanceof Depset)) {
      throw Starlark.errorf(
          "for %s, got %s, want a depset of %s",
          what, EvalUtils.getDataTypeName(x, true), EvalUtils.getDataTypeNameFromClass(type));
    }
    try {
      return ((Depset) x).getSet(type);
    } catch (TypeException ex) {
      throw Starlark.errorf("for '%s', %s", what, ex.getMessage());
    }
  }

  /** Like {@link #cast}, but if x is None, returns an empty stable-order NestedSet. */
  public static <T> NestedSet<T> noneableCast(Object x, Class<T> type, String what)
      throws EvalException {
    if (x == Starlark.NONE) {
      @SuppressWarnings("unchecked")
      NestedSet<T> empty = (NestedSet<T>) EMPTY;
      return empty;
    }
    return cast(x, type, what);
  }

  private static final NestedSet<?> EMPTY = NestedSetBuilder.<Object>emptySet(Order.STABLE_ORDER);

  public boolean isEmpty() {
    return set.isEmpty();
  }

  @Override
  public boolean truth() {
    return !set.isEmpty();
  }

  public SkylarkType getContentType() {
    return contentType;
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  public Order getOrder() {
    return set.getOrder();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("depset(");
    printer.printList(set.toList(), "[", ", ", "]", null);
    Order order = getOrder();
    if (order != Order.STABLE_ORDER) {
      printer.append(", order = ");
      printer.repr(order.getSkylarkName());
    }
    printer.append(")");
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
      useStarlarkThread = true)
  public StarlarkList<Object> toList(StarlarkThread thread) throws EvalException {
    try {
      return StarlarkList.copyOf(thread.mutability(), this.toCollection());
    } catch (NestedSetDepthException exception) {
      throw new EvalException(
          null,
          "depset exceeded maximum depth "
              + exception.getDepthLimit()
              + ". This was only discovered when attempting to flatten the depset for to_list(), "
              + "as the size of depsets is unknown until flattening. "
              + "See https://github.com/bazelbuild/bazel/issues/9180 for details and possible "
              + "solutions.");
    }
  }

  /** Create a Depset from the given direct and transitive components. */
  static Depset fromDirectAndTransitive(
      Order order, List<Object> direct, List<Depset> transitive, boolean strict)
      throws EvalException {
    NestedSetBuilder<Object> builder = new NestedSetBuilder<>(order);
    SkylarkType type = SkylarkType.TOP;

    // Check direct elements' type is equal to elements already added.
    for (Object x : direct) {
      // Historically, checkElement was called only by some depset constructors,
      // but not this one, depset(direct=[x]).
      // This was a regrettable oversight that allowed users to put mutable values
      // such as lists into depsets, doubly so because we have just forced our
      // users to migrate away from the legacy constructor which applied the check.
      //
      // We are currently discovering and fixing existing violations, for example
      // marking the relevant Starlark types as immutable where appropriate
      // (e.g. ConfiguredTarget), but violations are numerous so we must
      // suppress the checkElement call below and reintroduce it as a breaking change.
      // See b/144992997 or github.com/bazelbuild/bazel/issues/10289.
      checkElement(x, /*strict=*/ strict);

      SkylarkType xt = SkylarkType.of(x);
      type = checkType(type, xt);
    }
    builder.addAll(direct);

    // Add transitive sets, checking that type is equal to elements already added.
    for (Depset x : transitive) {
      if (!x.isEmpty()) {
        type = checkType(type, x.getContentType());
        if (!order.isCompatible(x.getOrder())) {
          throw Starlark.errorf(
              "Order '%s' is incompatible with order '%s'",
              order.getSkylarkName(), x.getOrder().getSkylarkName());
        }
        builder.addTransitive(x.getSet());
      }
    }

    return new Depset(type, builder.build(), null, null);
  }

  /** An exception thrown when validation fails on the type of elements of a nested set. */
  public static class TypeException extends Exception {
    TypeException(String message) {
      super(message);
    }
  }
}
