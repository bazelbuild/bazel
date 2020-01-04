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
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/** A StarlarkList is a mutable finite sequence of values. */
@SkylarkModule(
    name = "list",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "The built-in list type. Example list expressions:<br>"
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
            + "Lists are mutable, as in Python.")
public final class StarlarkList<E> extends AbstractList<E> implements Sequence<E>, StarlarkMutable {

  // The implementation strategy is similar to ArrayList,
  // but without the extra indirection of using ArrayList.

  private int size;
  private Object[] elems = EMPTY_ARRAY; // elems[i] == null  iff  i >= size

  /** Final except for {@link #unsafeShallowFreeze}; must not be modified any other way. */
  private Mutability mutability;

  private static final Object[] EMPTY_ARRAY = {};

  private StarlarkList(@Nullable Mutability mutability, Object[] elems) {
    this.elems = elems;
    this.size = elems.length;
    this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
  }

  /**
   * Takes ownership of the supplied array and returns a new StarlarkList instance that initially
   * wraps the array. The caller must not subsequently modify the array, but the StarlarkList
   * instance may do so.
   */
  static <T> StarlarkList<T> wrap(@Nullable Mutability mutability, Object[] elems) {
    return new StarlarkList<>(mutability, elems);
  }

  @Override
  public boolean isImmutable() {
    return mutability().isFrozen();
  }

  @Override
  public boolean isHashable() {
    return false; // even a frozen list is unhashable in Starlark
  }

  /**
   * A shared instance for the empty list with immutable mutability.
   *
   * <p>Other immutable empty list objects can exist, e.g. lists that were once mutable but whose
   * environments were then frozen. This instance is for empty lists that were always frozen from
   * the beginning.
   */
  private static final StarlarkList<?> EMPTY = wrap(Mutability.IMMUTABLE, EMPTY_ARRAY);

  /** Returns an empty frozen list of the desired type. */
  @SuppressWarnings("unchecked")
  public static <T> StarlarkList<T> empty() {
    return (StarlarkList<T>) EMPTY;
  }

  /** Returns a new, empty list with the specified Mutability. */
  public static <T> StarlarkList<T> newList(Mutability mutability) {
    return wrap(mutability, EMPTY_ARRAY);
  }

  /**
   * Returns a {@code StarlarkList} whose items are given by an iterable and which has the given
   * {@link Mutability}. If {@code mutability} is null, the list is immutable.
   */
  public static <T> StarlarkList<T> copyOf(
      @Nullable Mutability mutability, Iterable<? extends T> elems) {
    return wrap(mutability, Iterables.toArray(elems, Object.class));
  }

  /**
   * Returns an immutable list with the given elements. Equivalent to {@code copyOf(null, elems)}.
   */
  public static <T> StarlarkList<T> immutableCopyOf(Iterable<? extends T> elems) {
    return copyOf(null, elems);
  }

  /**
   * Returns a {@code StarlarkList} with the given items and the {@link Mutability}. If {@code
   * mutability} is null, the list is immutable.
   */
  public static <T> StarlarkList<T> of(@Nullable Mutability mutability, T... elems) {
    return wrap(mutability, Arrays.copyOf(elems, elems.length));
  }

  @Override
  public Mutability mutability() {
    return mutability;
  }

  @Override
  public void unsafeShallowFreeze() {
    Mutability.Freezable.checkUnsafeShallowFreezePrecondition(this);
    this.mutability = Mutability.IMMUTABLE;
  }

  @Override
  public ImmutableList<E> getImmutableList() {
    // Optimization: a frozen array needn't be copied.
    // If the entire array is full, we can wrap it directly.
    if (elems.length == size && mutability().isFrozen()) {
      return Tuple.wrapImmutable(elems);
    }

    return ImmutableList.copyOf(this);
  }

  /**
   * Returns a new {@code StarlarkList} that is the concatenation of two {@code StarlarkList}s. The
   * new list will have the given {@link Mutability}.
   */
  public static <T> StarlarkList<T> concat(
      StarlarkList<? extends T> x, StarlarkList<? extends T> y, Mutability mutability) {
    Object[] res = new Object[x.size + y.size];
    System.arraycopy(x.elems, 0, res, 0, x.size);
    System.arraycopy(y.elems, 0, res, x.size, y.size);
    return wrap(mutability, res);
  }

  @Override
  public boolean equals(Object that) {
    // This slightly violates the java.util.List equivalence contract
    // because it considers the class, not just the elements.
    return this == that || (that instanceof StarlarkList && sameElems(this, ((StarlarkList) that)));
  }

  private static boolean sameElems(StarlarkList<?> x, StarlarkList<?> y) {
    if (x.size != y.size) {
      return false;
    }
    for (int i = 0; i < x.size; i++) {
      if (!x.elems[i].equals(y.elems[i])) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    return 6047 + 4673 * Arrays.hashCode(elems);
  }

  @Override
  public void repr(Printer printer) {
    printer.printList(this, /*isTuple=*/ false);
  }

  // TODO(adonovan): StarlarkValue has 3 String methods yet still we need this fourth. Why?
  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  /** Returns a new StarlarkList containing n consecutive repeats of this tuple. */
  public StarlarkList<E> repeat(int n, Mutability mutability) {
    if (n <= 0) {
      return wrap(mutability, EMPTY_ARRAY);
    }

    // TODO(adonovan): reject unreasonably large n.
    Object[] res = new Object[n * size];
    for (int i = 0; i < n; i++) {
      System.arraycopy(elems, 0, res, i * size, size);
    }
    return wrap(mutability, res);
  }

  @Override
  @SuppressWarnings("unchecked")
  public E get(int i) {
    return (E) elems[i]; // unchecked
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public StarlarkList<E> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException {
    // TODO(adonovan): this is horribly inefficient.
    List<Integer> indices = EvalUtils.getSliceIndices(start, end, step, size(), loc);
    Object[] array = new Object[indices.size()];
    for (int i = 0; i < indices.size(); ++i) {
      array[i] = elems[indices.get(i)];
    }
    return wrap(mutability, array);
  }

  // Postcondition: elems.length >= mincap.
  private void grow(int mincap) {
    int oldcap = elems.length;
    if (oldcap < mincap) {
      int newcap = oldcap + (oldcap >> 1); // grow by at least 50%
      if (newcap < mincap) {
        newcap = mincap;
      }
      elems = Arrays.copyOf(elems, newcap);
    }
  }

  /**
   * Appends an element to the end of the list, after validating that mutation is allowed.
   *
   * @param element the element to add
   * @param loc the location to use for error reporting
   */
  public void add(E element, Location loc) throws EvalException {
    checkMutable(loc);
    grow(size + 1);
    elems[size++] = element;
  }

  /**
   * Inserts an element at a given position to the list.
   *
   * @param index the new element's index
   * @param element the element to add
   * @param loc the location to use for error reporting
   */
  public void add(int index, E element, Location loc) throws EvalException {
    checkMutable(loc);
    grow(size + 1);
    System.arraycopy(elems, index, elems, index + 1, size - index);
    elems[index] = element;
    size++;
  }

  /**
   * Appends all the elements to the end of the list.
   *
   * @param elements the elements to add
   * @param loc the location to use for error reporting
   */
  public void addAll(Iterable<? extends E> elements, Location loc) throws EvalException {
    checkMutable(loc);
    if (elements instanceof StarlarkList) {
      StarlarkList<?> that = (StarlarkList) elements;
      // (safe even if this == that)
      grow(this.size + that.size);
      System.arraycopy(that.elems, 0, this.elems, this.size, that.size);
      this.size += that.size;
    } else if (elements instanceof Collection) {
      // collection of known size
      Collection<?> that = (Collection) elements;
      grow(size + that.size());
      for (Object x : that) {
        elems[size++] = x;
      }
    } else {
      // iterable
      for (Object x : elements) {
        grow(size + 1);
        elems[size++] = x;
      }
    }
  }

  /**
   * Removes the element at a given index. The index must already have been validated to be in
   * range.
   *
   * @param index the index of the element to remove
   * @param loc the location to use for error reporting
   */
  public void remove(int index, Location loc) throws EvalException {
    checkMutable(loc);
    int n = size - index - 1;
    if (n > 0) {
      System.arraycopy(elems, index + 1, elems, index, n);
    }
    elems[--size] = null; // aid GC
  }

  @SkylarkCallable(
      name = "remove",
      doc =
          "Removes the first item from the list whose value is x. "
              + "It is an error if there is no such item.",
      parameters = {@Param(name = "x", type = Object.class, doc = "The object to remove.")})
  public NoneType removeObject(Object x) throws EvalException {
    for (int i = 0; i < size; i++) {
      if (elems[i].equals(x)) {
        remove(i, /*loc=*/ null);
        return Starlark.NONE;
      }
    }
    throw Starlark.errorf("item %s not found in list", Starlark.repr(x));
  }

  /**
   * Sets the position at the given index to contain the given value. The index must already have
   * been validated to be in range.
   *
   * @param index the position to change
   * @param value the new value
   * @param loc the location to use for error reporting
   */
  public void set(int index, E value, Location loc) throws EvalException {
    checkMutable(loc);
    elems[index] = value;
  }

  @SkylarkCallable(
      name = "append",
      doc = "Adds an item to the end of the list.",
      parameters = {
        @Param(name = "item", type = Object.class, doc = "Item to add at the end.", noneable = true)
      })
  @SuppressWarnings("unchecked")
  public NoneType append(Object item) throws EvalException {
    add((E) item, /*loc=*/ null); // unchecked
    return Starlark.NONE;
  }

  @SkylarkCallable(name = "clear", doc = "Removes all the elements of the list.")
  public NoneType clearMethod() throws EvalException {
    checkMutable(/*loc=*/ null);
    for (int i = 0; i < size; i++) {
      elems[i] = null; // aid GC
    }
    size = 0;
    return Starlark.NONE;
  }

  @SkylarkCallable(
      name = "insert",
      doc = "Inserts an item at a given position.",
      parameters = {
        @Param(name = "index", type = Integer.class, doc = "The index of the given position."),
        @Param(name = "item", type = Object.class, doc = "The item.", noneable = true)
      })
  @SuppressWarnings("unchecked")
  public NoneType insert(Integer index, Object item) throws EvalException {
    add(EvalUtils.clampRangeEndpoint(index, size), (E) item, /*loc=*/ null); // unchecked
    return Starlark.NONE;
  }

  @SkylarkCallable(
      name = "extend",
      doc = "Adds all items to the end of the list.",
      parameters = {@Param(name = "items", type = Object.class, doc = "Items to add at the end.")})
  public NoneType extend(Object items) throws EvalException {
    @SuppressWarnings("unchecked")
    Iterable<? extends E> src = (Iterable<? extends E>) Starlark.toIterable(items);
    addAll(src, (Location) null);
    return Starlark.NONE;
  }

  @SkylarkCallable(
      name = "index",
      doc =
          "Returns the index in the list of the first item whose value is x. "
              + "It is an error if there is no such item.",
      parameters = {
        @Param(name = "x", type = Object.class, doc = "The object to search."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc = "The start index of the list portion to inspect."),
        @Param(
            name = "end",
            type = Integer.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc = "The end index of the list portion to inspect.")
      })
  public Integer index(Object x, Object start, Object end) throws EvalException {
    int i = start == Starlark.NONE ? 0 : EvalUtils.clampRangeEndpoint((Integer) start, size);
    int j = end == Starlark.NONE ? size : EvalUtils.clampRangeEndpoint((Integer) end, size);
    for (; i < j; i++) {
      if (elems[i].equals(x)) {
        return i;
      }
    }
    throw Starlark.errorf("item %s not found in list", Starlark.repr(x));
  }

  @SkylarkCallable(
      name = "pop",
      doc =
          "Removes the item at the given position in the list, and returns it. "
              + "If no <code>index</code> is specified, "
              + "it removes and returns the last item in the list.",
      parameters = {
        @Param(
            name = "i",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "The index of the item.")
      })
  public Object pop(Object i) throws EvalException {
    int arg = i == Starlark.NONE ? -1 : (Integer) i;
    int index = EvalUtils.getSequenceIndex(arg, size, /*loc=*/ null);
    Object result = elems[index];
    remove(index, /*loc=*/ null);
    return result;
  }
}
