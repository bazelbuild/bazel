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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A StarlarkList is a mutable finite sequence of values.
 *
 * <p>Starlark operations on lists, including element update and the {@code append}, {@code insert},
 * and {@code extend} methods, may insert arbitrary Starlark values as list elements, regardless of
 * the type argument used to reference to the list from Java code. Therefore, as long as a list is
 * mutable, Java code should refer to it only through a type such as {@code StarlarkList<Object>} or
 * {@code StarlarkList<?>} to avoid undermining the type-safety of the Java application. Once the
 * list becomes frozen, it is safe to {@link #cast} it to a more specific type that accurately
 * reflects its elements, such as {@code StarlarkList<String>}.
 *
 * <p>The following List methods, by inheriting their implementations from AbstractList, are
 * effectively disabled. Use the corresponding methods with "element" in their name; they may report
 * mutation failure by throwing a checked exception.
 *
 * <pre>
 *   boolean add(E)                    -- use addElement
 *   boolean remove(Object)            -- use removeElement
 *   boolean addAll(Collection)        -- use addElements
 *   boolean addAll(int, Collection)
 *   boolean removeAll(Collection)     -- use removeElements
 *   boolean retainAll(Collection)
 *   void clear()                      -- use clearElements
 *   E set(int, E)                     -- use setElementAt
 *   void add(int, E)                  -- use addElementAt
 *   E remove(int)                     -- use removeElementAt
 * </pre>
 */
@StarlarkBuiltin(
    name = "list",
    category = "core",
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
public final class StarlarkList<E> extends AbstractList<E>
    implements Sequence<E>, StarlarkValue, Mutability.Freezable, Comparable<StarlarkList<?>> {

  // The implementation strategy is similar to ArrayList,
  // but without the extra indirection of using ArrayList.

  private int size;
  private int iteratorCount; // number of active iterators (unused once frozen)
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
  public void checkHashable() throws EvalException {
    // Even a frozen list is unhashable.
    throw Starlark.errorf("unhashable type: 'list'");
  }

  @Override
  public boolean updateIteratorCount(int delta) {
    if (mutability().isFrozen()) {
      return false;
    }
    if (delta > 0) {
      iteratorCount++;
    } else if (delta < 0) {
      iteratorCount--;
    }
    return iteratorCount > 0;
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
  public int compareTo(StarlarkList<?> that) {
    return Sequence.compare(this, that);
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
    // Roll our own hash code to avoid iterating through null part of elems.
    int result = 1;
    for (int i = 0; i < size; i++) {
      result = 31 * result + elems[i].hashCode();
    }
    return 6047 + 4673 * result;
  }

  @Override
  public void repr(Printer printer) {
    printer.printList(this, "[", ", ", "]");
  }

  // TODO(adonovan): StarlarkValue has 3 String methods yet still we need this fourth. Why?
  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  /** Returns a new StarlarkList containing n consecutive repeats of this tuple. */
  public StarlarkList<E> repeat(StarlarkInt n, Mutability mutability) throws EvalException {
    if (n.signum() <= 0) {
      return wrap(mutability, EMPTY_ARRAY);
    }

    // TODO(adonovan): reject unreasonably large n.
    int ni = n.toInt("repeat");
    Object[] res = new Object[ni * size];
    for (int i = 0; i < ni; i++) {
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
  public StarlarkList<E> getSlice(Mutability mu, int start, int stop, int step) {
    RangeList indices = new RangeList(start, stop, step);
    int n = indices.size();
    Object[] res = new Object[n];
    if (step == 1) { // common case
      System.arraycopy(elems, indices.at(0), res, 0, n);
    } else {
      for (int i = 0; i < n; ++i) {
        res[i] = elems[indices.at(i)];
      }
    }
    return wrap(mu, res);
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
   */
  public void addElement(E element) throws EvalException {
    Starlark.checkMutable(this);
    grow(size + 1);
    elems[size++] = element;
  }

  /**
   * Inserts an element at a given position to the list.
   *
   * @param index the new element's index
   * @param element the element to add
   */
  public void addElementAt(int index, E element) throws EvalException {
    Starlark.checkMutable(this);
    grow(size + 1);
    System.arraycopy(elems, index, elems, index + 1, size - index);
    elems[index] = element;
    size++;
  }

  /**
   * Appends all the elements to the end of the list.
   *
   * @param elements the elements to add
   */
  public void addElements(Iterable<? extends E> elements) throws EvalException {
    Starlark.checkMutable(this);
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
   */
  public void removeElementAt(int index) throws EvalException {
    Starlark.checkMutable(this);
    int n = size - index - 1;
    if (n > 0) {
      System.arraycopy(elems, index + 1, elems, index, n);
    }
    elems[--size] = null; // aid GC
  }

  @StarlarkMethod(
      name = "remove",
      doc =
          "Removes the first item from the list whose value is x. "
              + "It is an error if there is no such item.",
      parameters = {@Param(name = "x", doc = "The object to remove.")})
  public NoneType removeElement(Object x) throws EvalException {
    for (int i = 0; i < size; i++) {
      if (elems[i].equals(x)) {
        removeElementAt(i);
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
   */
  public void setElementAt(int index, E value) throws EvalException {
    Starlark.checkMutable(this);
    elems[index] = value;
  }

  @StarlarkMethod(
      name = "append",
      doc = "Adds an item to the end of the list.",
      parameters = {@Param(name = "item", doc = "Item to add at the end.")})
  @SuppressWarnings("unchecked")
  public NoneType append(Object item) throws EvalException {
    addElement((E) item); // unchecked
    return Starlark.NONE;
  }

  @StarlarkMethod(name = "clear", doc = "Removes all the elements of the list.")
  public NoneType clearElements() throws EvalException {
    Starlark.checkMutable(this);
    for (int i = 0; i < size; i++) {
      elems[i] = null; // aid GC
    }
    size = 0;
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "insert",
      doc = "Inserts an item at a given position.",
      parameters = {
        @Param(name = "index", doc = "The index of the given position."),
        @Param(name = "item", doc = "The item.")
      })
  @SuppressWarnings("unchecked")
  public NoneType insert(StarlarkInt index, Object item) throws EvalException {
    addElementAt(EvalUtils.toIndex(index.toInt("index"), size), (E) item); // unchecked
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "extend",
      doc = "Adds all items to the end of the list.",
      parameters = {@Param(name = "items", doc = "Items to add at the end.")})
  public NoneType extend(Object items) throws EvalException {
    @SuppressWarnings("unchecked")
    Iterable<? extends E> src = (Iterable<? extends E>) Starlark.toIterable(items);
    addElements(src);
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "index",
      doc =
          "Returns the index in the list of the first item whose value is x. "
              + "It is an error if there is no such item.",
      parameters = {
        @Param(name = "x", doc = "The object to search."),
        @Param(
            name = "start",
            allowedTypes = {
              @ParamType(type = StarlarkInt.class),
              @ParamType(type = NoneType.class), // TODO(adonovan): this is wrong
            },
            defaultValue = "None",
            named = true, // TODO(adonovan): this is wrong
            doc = "The start index of the list portion to inspect."),
        @Param(
            name = "end",
            allowedTypes = {
              @ParamType(type = StarlarkInt.class),
              @ParamType(type = NoneType.class), // TODO(adonovan): this is wrong
            },
            defaultValue = "None",
            named = true, // TODO(adonovan): this is wrong
            doc = "The end index of the list portion to inspect.")
      })
  public Integer index(Object x, Object start, Object end) throws EvalException {
    int i = start == Starlark.NONE ? 0 : EvalUtils.toIndex(Starlark.toInt(start, "start"), size);
    int j = end == Starlark.NONE ? size : EvalUtils.toIndex(Starlark.toInt(end, "end"), size);
    for (; i < j; i++) {
      if (elems[i].equals(x)) {
        return i;
      }
    }
    throw Starlark.errorf("item %s not found in list", Starlark.repr(x));
  }

  @StarlarkMethod(
      name = "pop",
      doc =
          "Removes the item at the given position in the list, and returns it. "
              + "If no <code>index</code> is specified, "
              + "it removes and returns the last item in the list.",
      parameters = {
        @Param(
            name = "i",
            allowedTypes = {
              @ParamType(type = StarlarkInt.class),
              @ParamType(type = NoneType.class), // TODO(adonovan): this is not what Python3 does
            },
            defaultValue = "-1",
            doc = "The index of the item.")
      })
  public Object pop(Object i) throws EvalException {
    int arg = i == Starlark.NONE ? -1 : Starlark.toInt(i, "i");
    int index = EvalUtils.getSequenceIndex(arg, size);
    Object result = elems[index];
    removeElementAt(index);
    return result;
  }

  @Override
  public Object[] toArray() {
    return size != 0 ? Arrays.copyOf(elems, size) : EMPTY_ARRAY;
  }
}
