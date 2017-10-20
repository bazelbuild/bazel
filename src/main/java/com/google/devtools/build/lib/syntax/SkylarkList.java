// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.SkylarkMutable.BaseMutableList;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.RandomAccess;
import javax.annotation.Nullable;

/**
 * A Skylark list or tuple.
 *
 * <p>Although this implements the {@link List} interface, it is not mutable via that interface's
 * methods. Instead, use the mutators that take in a {@link Mutability} object.
 */
@SkylarkModule(
  name = "sequence",
  documented = false,
  category = SkylarkModuleCategory.BUILTIN,
  doc = "common type of lists and tuples."
)
public abstract class SkylarkList<E> extends BaseMutableList<E>
    implements List<E>, RandomAccess, SkylarkIndexable {

  /** Returns true if this list is a Skylark tuple. */
  public abstract boolean isTuple();

  /**
   * Returns an ImmutableList object with the current underlying contents of this SkylarkList.
   */
  public abstract ImmutableList<E> getImmutableList();

  /**
   * Retrieve an entry from a SkylarkList.
   *
   * @param key the index
   * @param loc a {@link Location} in case of error
   * @throws EvalException if the key is invalid
   */
  @Override
  public E getIndex(Object key, Location loc) throws EvalException {
    List<E> list = getContentsUnsafe();
    int index = EvalUtils.getSequenceIndex(key, list.size(), loc);
    return list.get(index);
  }

  @Override
  public boolean containsKey(Object key, Location loc) throws EvalException {
    for (Object obj : this) {
      if (obj.equals(key)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Constructs a version of this {@code SkylarkList} containing just the items in a slice.
   *
   * <p>{@code mutability} will be used for the resulting list. If it is null, the list will be
   * immutable. For {@code Tuple}s, which are always immutable, this argument is ignored.
   *
   * @see EvalUtils#getSliceIndices
   * @throws EvalException if the key is invalid; uses {@code loc} for error reporting
   */
  public abstract SkylarkList<E> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException;

  /**
   * Constructs a repetition of this {@code SkylarkList}.
   *
   * <p>{@code mutability} will be used for the resulting list. If it is null, the list will be
   * immutable. For {@code Tuple}s, which are always immutable, this argument is ignored.
   */
  public abstract SkylarkList<E> repeat(int times, Mutability mutability);

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.printList(getContentsUnsafe(), isTuple());
  }

  @Override
  public String toString() {
    return Printer.repr(this);
  }

  // Note that the following two functions slightly violate the Java List protocol,
  // in that it does NOT consider that a SkylarkList .equals() an arbitrary List with same contents.
  // This is because we use .equals() to model skylark equality, which like Python
  // distinguishes a MutableList from a Tuple.
  @Override
  public boolean equals(Object object) {
    return (this == object)
        || ((this.getClass() == object.getClass())
            && getContentsUnsafe().equals(((SkylarkList) object).getContentsUnsafe()));
  }

  @Override
  public int hashCode() {
    return getClass().hashCode() + 31 * getContentsUnsafe().hashCode();
  }

  /**
   * Casts a {@code List<?>} to an unmodifiable {@code List<T>}, after checking that its contents
   * all have type {@code type}.
   *
   * <p>The returned list may or may not be a view that is affected by updates to the original list.
   *
   * @param list the original list to cast
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  // We could have used bounded generics to ensure that only downcasts are possible (i.e. cast
  // List<S> to List<T extends S>), but this would be less convenient for some callers, and it would
  // disallow casting an empty list to any type.
  @SuppressWarnings("unchecked")
  public static <T> List<T> castList(List<?> list, Class<T> type, @Nullable String description)
      throws EvalException {
    Object desc = description == null ? null : Printer.formattable("'%s' element", description);
    for (Object value : list) {
      SkylarkType.checkType(value, type, desc);
    }
    return Collections.unmodifiableList((List<T>) list);
  }

  /**
   * If {@code obj} is a {@code SkylarkList}, casts it to an unmodifiable {@code List<T>} after
   * checking that each element has type {@code type}. If {@code obj} is {@code None} or null,
   * treats it as an empty list. For all other values, throws an {@link EvalException}.
   *
   * <p>The returned list may or may not be a view that is affected by updates to the original list.
   *
   * @param obj the object to cast. null and None are treated as an empty list.
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  public static <T> List<T> castSkylarkListOrNoneToList(
      Object obj, Class<T> type, @Nullable String description)
      throws EvalException {
    if (EvalUtils.isNullOrNone(obj)) {
      return ImmutableList.of();
    }
    if (obj instanceof SkylarkList) {
      return ((SkylarkList<?>) obj).getContents(type, description);
    }
    throw new EvalException(null,
        String.format("Illegal argument: %s is not of expected type list or NoneType",
            description == null ? Printer.repr(obj) : String.format("'%s'", description)));
  }

  /**
   * Casts this list as an unmodifiable {@code List<T>}, after checking that each element has
   * type {@code type}.
   *
   * @param type the expected type of all the list's elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  public <T> List<T> getContents(Class<T> type, @Nullable String description)
      throws EvalException {
    return castList(getContentsUnsafe(), type, description);
  }

  /**
   * Creates an immutable Skylark list with the given elements.
   *
   * It is unspecified whether this is a Skylark list or tuple. For more control, use one of the
   * factory methods in {@link MutableList} or {@link Tuple}.
   */
  // TODO(bazel-team): Eliminate this function in favor of a new MutableList factory method. With
  // such a method, we may no longer need to take null as a possible value for the Mutability or
  // Environment. That in turn would allow us to overload MutableList#of to take either a Mutability
  // or Environment.
  public static <E> SkylarkList<E> createImmutable(Iterable<? extends E> contents) {
    return MutableList.copyOf(Mutability.IMMUTABLE, contents);
  }

  /**
   * A Skylark list, i.e., the value represented by {@code [1, 2, 3]}. Lists are mutable datatypes.
   */
  @SkylarkModule(
    name = "list",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A language built-in type to support lists. Example of list literal:<br>"
            + "<pre class=language-python>x = [1, 2, 3]</pre>"
            + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
            + "<pre class=language-python>e = x[1]   # e == 2</pre>"
            + "Lists support the <code>+</code> operator to concatenate two lists. Example:<br>"
            + "<pre class=language-python>x = [1, 2] + [3, 4]   # x == [1, 2, 3, 4]\n"
            + "x = [\"a\", \"b\"]\n"
            + "x += [\"c\"]            # x == [\"a\", \"b\", \"c\"]</pre>"
            + "Similar to strings, lists support slice operations:"
            + "<pre class=language-python>['a', 'b', 'c', 'd'][1:3]   # ['b', 'c']\n"
            + "['a', 'b', 'c', 'd'][::2]  # ['a', 'c']\n"
            + "['a', 'b', 'c', 'd'][3:0:-1]  # ['d', 'c', 'b']</pre>"
            + "Lists are mutable, as in Python."
  )
  public static final class MutableList<E> extends SkylarkList<E> {

    private final ArrayList<E> contents;

    // Treat GlobList specially: external code depends on it.
    // TODO(bazel-team): make data structures *and binary operators* extensible
    // (via e.g. interface classes for each binary operator) so that GlobList
    // can be implemented outside of the core of Skylark.
    // TODO(bazel-team): move GlobList out of Skylark, into an extension.
    @Nullable private GlobList<E> globList;

    private final Mutability mutability;

    private MutableList(
        ArrayList<E> rawContents,
        @Nullable GlobList<E> globList,
        @Nullable Mutability mutability) {
      this.contents = Preconditions.checkNotNull(rawContents);
      this.globList = globList;
      this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
    }

    /**
     * Creates an instance, taking ownership of the supplied {@link ArrayList}. This is exposed for
     * performance reasons. May be used when the supplied list is certainly not a {@link GlobList}
     * (should be enforced by type system) and the calling code will not modify the supplied list
     * after calling (honor system).
     */
    static <T> MutableList<T> wrapUnsafe(@Nullable Environment env, ArrayList<T> rawContents) {
      return wrapUnsafe(env == null ? null : env.mutability(), rawContents);
    }

    /**
     * Create an instance, taking ownership of the supplied {@link ArrayList}. This is exposed for
     * performance reasons. May be used when the supplied list is certainly not a {@link GlobList}
     * (enforced by type system as long as {@link GlobList} doesn't extend {@link ArrayList}) and
     * the calling code will not modify the supplied list after calling (honor system).
     */
    static <T> MutableList<T> wrapUnsafe(
        @Nullable Mutability mutability, ArrayList<T> rawContents) {
      return new MutableList<>(rawContents, /*globList=*/ null, mutability);
    }

    /**
     * A shared instance for the empty list with immutable mutability.
     *
     * <p>Other immutable empty list objects can exist, e.g. lists that were once mutable but whose
     * environments were then frozen. This instance is for empty lists that were always frozen from
     * the beginning.
     */
    private static final MutableList<?> EMPTY =
        MutableList.copyOf(Mutability.IMMUTABLE, ImmutableList.of());

    /** Returns an empty frozen list, cast to have an arbitrary content type. */
    @SuppressWarnings("unchecked")
    public static <T> MutableList<T> empty() {
      return (MutableList<T>) EMPTY;
    }

    /**
     * Returns a {@code MutableList} whose items are given by an iterable and which has the given
     * {@link Mutability}. If {@code mutability} is null, the list is immutable.
     */
    @SuppressWarnings("unchecked")  // GlobList cast.
    public static <T> MutableList<T> copyOf(
        @Nullable Mutability mutability, Iterable<? extends T> contents) {
      return new MutableList<>(
          Lists.newArrayList(contents),
          contents instanceof GlobList ? (GlobList<T>) contents : null,
          mutability);
    }

    /**
     * Returns a {@code MutableList} whose items are given by an iterable and which has the {@link
     * Mutability} belonging to the given {@link Environment}. If {@code env} is null, the list is
     * immutable.
     */
    public static <T> MutableList<T> copyOf(
        @Nullable Environment env, Iterable<? extends T> contents) {
      return MutableList.copyOf(
          env == null ? null : env.mutability(),
          contents);
    }

    /**
     * Returns a {@code MutableList} with the given items and the {@link Mutability} of the given
     * {@link Environment}. If {@code env} is null, the list is immutable.
     */
    public static <T> MutableList<T> of(@Nullable Environment env, T... contents) {
      // Safe since it's definitely not a GlobList, and we're taking a copy of the input.
      return MutableList.wrapUnsafe(
          env == null ? null : env.mutability(),
          Lists.newArrayList(contents));
    }

    @Override
    public Mutability mutability() {
      return mutability;
    }

    @Override
    protected void checkMutable(Location loc, Mutability mutability) throws EvalException {
      super.checkMutable(loc, mutability);
      globList = null; // If you're going to mutate it, invalidate the underlying GlobList.
    }

    /** Returns the {@link GlobList} if there is one, or else null. */
    @Nullable public GlobList<E> getGlobList() {
      return globList;
    }

    @Override
    public boolean isTuple() {
      return false;
    }

    @Override
    public ImmutableList<E> getImmutableList() {
      return ImmutableList.copyOf(contents);
    }

    @Override
    protected List<E> getContentsUnsafe() {
      return contents;
    }

    /** Returns the {@link GlobList} if there is one, otherwise the regular contents. */
    private List<E> getGlobListOrContentsUnsafe() {
      return globList != null ? globList : contents;
    }

    /**
     * Returns a new {@code MutableList} that is the concatenation of two {@code MutableList}s. The
     * new list will have the given {@link Mutability}.
     */
    public static <T> MutableList<T> concat(
        MutableList<? extends T> left,
        MutableList<? extends T> right,
        Mutability mutability) {
      if (left.getGlobList() == null && right.getGlobList() == null) {
        ArrayList<T> newContents = new ArrayList<>(left.size() + right.size());
        addAll(newContents, left.contents);
        addAll(newContents, right.contents);
        return new MutableList<>(newContents, /*globList=*/ null, mutability);
      } else {
        // Preserve glob criteria.
        GlobList<T> newGlobList = GlobList.concat(
            left.getGlobListOrContentsUnsafe(),
            right.getGlobListOrContentsUnsafe());
        return new MutableList<>(new ArrayList<>(newGlobList), newGlobList, mutability);
      }
    }

    /**  More efficient {@link List#addAll} replacement when both lists are {@link ArrayList}s. */
    private static <T> void addAll(ArrayList<T> addTo, ArrayList<? extends T> addFrom) {
      // Hot code path, skip iterator.
      for (int i = 0; i < addFrom.size(); i++) {
        addTo.add(addFrom.get(i));
      }
    }

    @Override
    public MutableList<E> repeat(int times, Mutability mutability) {
      if (times <= 0) {
        return MutableList.wrapUnsafe(mutability, new ArrayList<>());
      }

      if (getGlobList() == null) {
        ArrayList<E> repeated = new ArrayList<>(this.size() * times);
        for (int i = 0; i < times; i++) {
          repeated.addAll(this);
        }
        return MutableList.wrapUnsafe(mutability, repeated);
      } else {
        // Preserve glob criteria.
        List<? extends E> globs = getGlobListOrContentsUnsafe();
        List<? extends E> original = globs;
        for (int i = 1; i < times; i++) {
          globs = GlobList.concat(globs, original);
        }
        return MutableList.copyOf(mutability, globs);
      }
    }

    @Override
    public MutableList<E> getSlice(
        Object start, Object end, Object step, Location loc, Mutability mutability)
        throws EvalException {
      List<Integer> sliceIndices = EvalUtils.getSliceIndices(start, end, step, this.size(), loc);
      ArrayList<E> list = new ArrayList<>(sliceIndices.size());
      for (int pos : sliceIndices) {
        list.add(this.get(pos));
      }
      return MutableList.wrapUnsafe(mutability, list);
    }

    /**
     * Appends an element to the end of the list, after validating that mutation is allowed.
     *
     * @param element the element to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void add(E element, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.add(element);
    }

    /**
     * Inserts an element at a given position to the list.
     *
     * @param index the new element's index
     * @param element the element to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void add(int index, E element, Location loc, Mutability mutability)
        throws EvalException {
      checkMutable(loc, mutability);
      contents.add(index, element);
    }

    /**
     * Appends all the elements to the end of the list.
     *
     * @param elements the elements to add
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void addAll(Iterable<? extends E> elements, Location loc, Mutability mutability)
        throws EvalException {
      checkMutable(loc, mutability);
      Iterables.addAll(contents, elements);
    }

    /**
     * Removes the element at a given index. The index must already have been validated to be in
     * range.
     *
     * @param index the index of the element to remove
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void remove(int index, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.remove(index);
    }

    /**
     * Sets the position at the given index to contain the given value. The index must already have
     * been validated to be in range.
     *
     * @param index the position to change
     * @param value the new value
     * @param loc the location to use for error reporting
     * @param mutability the {@link Mutability} associated with the operation
     */
    public void set(int index, E value, Location loc, Mutability mutability) throws EvalException {
      checkMutable(loc, mutability);
      contents.set(index, value);
    }
  }

  /**
   * A Skylark tuple, i.e. the value represented by {@code (1, 2, 3)}. Tuples are always immutable
   * (regardless of the {@link Environment} they are created in).
   */
  @SkylarkModule(
    name = "tuple",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A language built-in type to support tuples. Example of tuple literal:<br>"
            + "<pre class=language-python>x = (1, 2, 3)</pre>"
            + "Accessing elements is possible using indexing (starts from <code>0</code>):<br>"
            + "<pre class=language-python>e = x[1]   # e == 2</pre>"
            + "Lists support the <code>+</code> operator to concatenate two tuples. Example:<br>"
            + "<pre class=language-python>x = (1, 2) + (3, 4)   # x == (1, 2, 3, 4)\n"
            + "x = (\"a\", \"b\")\n"
            + "x += (\"c\",)            # x == (\"a\", \"b\", \"c\")</pre>"
            + "Similar to lists, tuples support slice operations:"
            + "<pre class=language-python>('a', 'b', 'c', 'd')[1:3]   # ('b', 'c')\n"
            + "('a', 'b', 'c', 'd')[::2]  # ('a', 'c')\n"
            + "('a', 'b', 'c', 'd')[3:0:-1]  # ('d', 'c', 'b')</pre>"
            + "Tuples are immutable, therefore <code>x[1] = \"a\"</code> is not supported."
  )
  public static final class Tuple<E> extends SkylarkList<E> {

    private final ImmutableList<E> contents;

    private Tuple(ImmutableList<E> contents) {
      this.contents = contents;
    }

    /**
     * A shared instance for the empty tuple.
     *
     * <p>This instance should be the only empty tuple.
     */
    private static final Tuple<?> EMPTY = new Tuple<>(ImmutableList.of());

    /** Returns the empty tuple, cast to have an arbitrary content type. */
    @SuppressWarnings("unchecked")
    public static <T> Tuple<T> empty() {
      return (Tuple<T>) EMPTY;
    }

    /**
     * Creates a {@code Tuple} from an {@link ImmutableList}, reusing the empty instance if
     * applicable.
     */
    private static<T> Tuple<T> create(ImmutableList<T> contents) {
      if (contents.isEmpty()) {
        return empty();
      }
      return new Tuple<>(contents);
    }

    /** Returns a {@code Tuple} whose items are given by an iterable. */
    public static <T> Tuple<T> copyOf(Iterable<? extends T> contents) {
      return create(ImmutableList.<T>copyOf(contents));
    }

    /** Returns a {@code Tuple} with the given items. */
    public static <T> Tuple<T> of(T... elements) {
      return Tuple.create(ImmutableList.copyOf(elements));
    }

    @Override
    public boolean isImmutable() {
      for (Object item : this) {
        if (!EvalUtils.isImmutable(item)) {
          return false;
        }
      }
      return true;
    }

    @Override
    public Mutability mutability() {
      return Mutability.IMMUTABLE;
    }

    @Override
    public boolean isTuple() {
      return true;
    }

    @Override
    public ImmutableList<E> getImmutableList() {
      return contents;
    }

    @Override
    protected List<E> getContentsUnsafe() {
      return contents;
    }

    /** Returns a {@code Tuple} that is the concatenation of two {@code Tuple}s. */
    public static <T> Tuple<T> concat(Tuple<? extends T> left, Tuple<? extends T> right) {
      // Build the ImmutableList directly rather than use Iterables.concat, to avoid unnecessary
      // array resizing.
      return create(ImmutableList.<T>builder()
          .addAll(left)
          .addAll(right)
          .build());
    }

    @Override
    public Tuple<E> getSlice(
        Object start, Object end, Object step, Location loc, Mutability mutability)
        throws EvalException {
      List<Integer> sliceIndices = EvalUtils.getSliceIndices(start, end, step, this.size(), loc);
      ImmutableList.Builder<E> builder = ImmutableList.builder();
      for (int pos : sliceIndices) {
        builder.add(this.get(pos));
      }
      return copyOf(builder.build());
    }

    @Override
    public Tuple<E> repeat(int times, Mutability mutability) {
      ImmutableList.Builder<E> builder = ImmutableList.builder();
      for (int i = 0; i < times; i++) {
        builder.addAll(this);
      }
      return copyOf(builder.build());
    }
  }
}
