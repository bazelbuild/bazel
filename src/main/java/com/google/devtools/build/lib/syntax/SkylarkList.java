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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.SkylarkMutable.BaseMutableList;
import java.util.ArrayList;
import java.util.Collection;
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
   * Constructs an {@link ImmutableList} containing the items in a slice of the given {@code
   * SkylarkList}.
   *
   * @see EvalUtils#getSliceIndices
   * @throws EvalException if the key is invalid; uses {@code loc} for error reporting
   */
  protected static <T> ImmutableList<T> getSliceContents(
      SkylarkList<T> list, Object start, Object end, Object step, Location loc)
      throws EvalException {
    int length = list.size();
    ImmutableList.Builder<T> items = ImmutableList.builder();
    for (int pos : EvalUtils.getSliceIndices(start, end, step, length, loc)) {
      items.add(list.get(pos));
    }
    return items.build();
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
   * Constructs an {@link ImmutableList} containing the items in a repetition of the given {@code
   * SkylarkList}.
   *
   * <p>A repetition is produced by concatenating the list with itself {@code times - 1} many times.
   * If {@code times} is 1, the new list's contents are the same as the original list. If {@code
   * times} is <= 0, an empty list is returned.
   */
  public static <T> ImmutableList<T> repeatContents(SkylarkList<? extends T> list, int times) {
    ImmutableList.Builder<T> builder = ImmutableList.builder();
    for (int i = 0; i < times; i++) {
      builder.addAll(list);
    }
    return builder.build();
  }

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
   * Casts a {@code List<?>} to a {@code List<T>} after checking its contents.
   *
   * @param list the List to cast
   * @param type the expected class of elements
   * @param description a description of the argument being converted, or null, for debugging
   */
  @SuppressWarnings("unchecked")
  public static <T> List<T> castList(List<?> list, Class<T> type, @Nullable String description)
      throws EvalException {
    Object desc = description == null ? null : Printer.formattable("'%s' element", description);
    for (Object value : list) {
      SkylarkType.checkType(value, type, desc);
    }
    return (List<T>) list;
  }

  /**
   * If {@code obj} is a {@code SkylarkList}, casts it to a {@code List<T>} after checking its
   * contents. If {@code obj} is {@code None} or null, treats it as an empty list. For all other
   * values, throws an {@link EvalException}.
   *
   * @param obj the Object to cast. null and None are treated as an empty list.
   * @param type the expected class of elements
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
   * Casts this list as a List of {@code T}, checking that each element has type {@code T}.
   *
   * @param type the expected class of elements
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
    return new MutableList<>(contents, Mutability.IMMUTABLE);
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

    private final ArrayList<E> contents = new ArrayList<>();

    // Treat GlobList specially: external code depends on it.
    // TODO(bazel-team): make data structures *and binary operators* extensible
    // (via e.g. interface classes for each binary operator) so that GlobList
    // can be implemented outside of the core of Skylark.
    // TODO(bazel-team): move GlobList out of Skylark, into an extension.
    @Nullable private GlobList<E> globList;

    private final Mutability mutability;

    /**
     * Constructs from the given items and {@link Mutability}.
     *
     * @param contents the contents of the new list. If this is a {@link GlobList}, it is also
     *     stored in {@code globList}.
     * @param capacity an size to pre-allocate the array to. Use 0 if unsure. This is unnecessary if
     *     {@code contents} is a {@link Collection}.
     * @param mutability the {@code Mutability} to use for the new list. If null, the new list is
     *     immutable.
     */
    // Suppress warning for cast guarded by instanceof.
    @SuppressWarnings("unchecked")
    private MutableList(
        Iterable<? extends E> contents, int capacity, @Nullable Mutability mutability) {
      this.contents.ensureCapacity(capacity);
      addAllUnsafe(contents);
      if (contents instanceof GlobList) {
        globList = (GlobList<E>) contents;
      }
      this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
    }

    private MutableList(Iterable<? extends E> contents, @Nullable Mutability mutability) {
      this(contents, 0, mutability);
    }

    /**
     * Constructs from the given items and the {@link Mutability} belonging to the given {@link
     * Environment}. If {@code env} is null, the list is immutable.
     *
     * @deprecated prefer using {@link #copyOf}
     */
    @Deprecated
    public MutableList(Iterable<? extends E> contents, @Nullable Environment env) {
      this(contents, 0, env == null ? null : env.mutability());
    }

    /**
     * A shared instance for the empty list with immutable mutability.
     *
     * <p>Other immutable empty list objects can exist, e.g. lists that were once mutable but whose
     * environments were then frozen. This instance is for empty lists that were always frozen from
     * the beginning.
     */
    private static final MutableList<?> EMPTY =
        new MutableList<>(ImmutableList.of(), Mutability.IMMUTABLE);

    /** Returns an empty frozen list, cast to have an arbitrary content type. */
    @SuppressWarnings("unchecked")
    public static <T> MutableList<T> empty() {
      return (MutableList<T>) EMPTY;
    }

    /**
     * Returns a {@code MutableList} whose items are given by an iterable and which has the given
     * {@link Mutability}. If {@code mutability} is null, the list is immutable.
     */
    public static <T> MutableList<T> copyOf(
        @Nullable Mutability mutability, Iterable<? extends T> contents) {
      return new MutableList<>(contents, mutability);
    }

    /**
     * Returns a {@code MutableList} whose items are given by an iterable and which has the {@link
     * Mutability} belonging to the given {@link Environment}. If {@code env} is null, the list is
     * immutable.
     */
    public static <T> MutableList<T> copyOf(
        @Nullable Environment env, Iterable<? extends T> contents) {
      return new MutableList<>(contents, env.mutability());
    }

    /**
     * Returns a {@code MutableList} with the given items and the {@link Mutability} of the given
     * {@link Environment}. If {@code env} is null, the list is immutable.
     */
    public static <T> MutableList<T> of(@Nullable Environment env, T... contents) {
      return new MutableList<>(
          ImmutableList.copyOf(contents), env == null ? null : env.mutability());
    }

    /** Appends the given elements to the end of the list, without calling {@link #checkMutable}. */
    private void addAllUnsafe(Iterable<? extends E> elements) {
      Iterables.addAll(contents, elements);
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
        return new MutableList<>(
            Iterables.concat(left, right),
            left.size() + right.size(),
            mutability);
      } else {
        // Preserve glob criteria.
        return new MutableList<>(
            GlobList.concat(
                left.getGlobListOrContentsUnsafe(),
                right.getGlobListOrContentsUnsafe()),
            mutability);
      }
    }

    @Override
    public MutableList<E> repeat(int times, Mutability mutability) {
      if (getGlobList() == null) {
        return new MutableList<>(repeatContents(this, times), mutability);
      } else {
        if (times <= 0) {
          return new MutableList<>(ImmutableList.of(), mutability);
        } else {
          // Preserve glob criteria.
          List<? extends E> globs = getGlobListOrContentsUnsafe();
          List<? extends E> original = globs;
          for (int i = 1; i < times; i++) {
            globs = GlobList.concat(globs, original);
          }
          return new MutableList<>(globs, mutability);
        }
      }
    }

    @Override
    public MutableList<E> getSlice(
        Object start, Object end, Object step, Location loc, Mutability mutability)
        throws EvalException {
      return new MutableList<>(getSliceContents(this, start, end, step, loc), mutability);
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
      addAllUnsafe(elements);
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
      return copyOf(getSliceContents(this, start, end, step, loc));
    }

    @Override
    public Tuple<E> repeat(int times, Mutability mutability) {
      return copyOf(repeatContents(this, times));
    }
  }
}
