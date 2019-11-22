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
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import java.util.AbstractCollection;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * A Skylark tuple, i.e. the value represented by {@code (1, 2, 3)}. Tuples are always immutable
 * (regardless of the {@link StarlarkThread} they are created in).
 */
@SkylarkModule(
    name = "tuple",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "The built-in tuple type. Example tuple expressions:<br>"
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
            + "Tuples are immutable, therefore <code>x[1] = \"a\"</code> is not supported.")
public final class Tuple<E> extends AbstractList<E> implements Sequence<E> {

  private final Object[] elems;

  private Tuple(Object[] elems) {
    this.elems = elems;
  }

  // The shared (sole) empty tuple.
  private static final Tuple<?> EMPTY = new Tuple<>(new Object[0]);

  /** Returns the empty tuple, cast to have an arbitrary content type. */
  @SuppressWarnings("unchecked")
  public static <T> Tuple<T> empty() {
    return (Tuple<T>) EMPTY; // unchecked
  }

  /**
   * Returns a Tuple that wraps the specified array, which must not be subsequently modified. The
   * caller is additionally trusted to choose an appropriate type T.
   */
  static <T> Tuple<T> wrap(Object[] array) {
    return array.length == 0 ? empty() : new Tuple<T>(array);
  }

  /** Returns a tuple containing the given elements. */
  @SuppressWarnings("unchecked")
  public static <T> Tuple<T> copyOf(Iterable<? extends T> seq) {
    if (seq instanceof Tuple) {
      return (Tuple<T>) seq; // unchecked
    }
    return wrap(Iterables.toArray(seq, Object.class));
  }

  /** Returns a tuple containing the given elements. */
  public static <T> Tuple<T> of(T... elems) {
    if (elems.length == 0) {
      return empty();
    }
    return new Tuple<T>(Arrays.copyOf(elems, elems.length));
  }

  /** Returns a two-element tuple. */
  public static <T> Tuple<T> pair(T a, T b) {
    // Equivalent to of(a, b) but avoids variadic array allocation.
    return wrap(new Object[] {a, b});
  }

  /** Returns a three-element tuple. */
  public static <T> Tuple<T> triple(T a, T b, T c) {
    // Equivalent to of(a, b, c) but avoids variadic array allocation.
    return wrap(new Object[] {a, b, c});
  }

  /** Returns a tuple that is the concatenation of two tuples. */
  public static <T> Tuple<T> concat(Tuple<? extends T> x, Tuple<? extends T> y) {
    // TODO(adonovan): opt: exploit x + () == x; y + () == y.
    return wrap(ObjectArrays.concat(x.elems, y.elems, Object.class));
  }

  @Override
  public boolean isImmutable() {
    for (Object x : elems) {
      if (!EvalUtils.isImmutable(x)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean isHashable() {
    for (Object x : elems) {
      if (!EvalUtils.isHashable(x)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    return 9857 + 8167 * Arrays.hashCode(elems);
  }

  @Override
  public boolean equals(Object that) {
    // This slightly violates the java.util.List equivalence contract
    // because it considers the class, not just the elements.
    return this == that
        || (that instanceof Tuple && Arrays.equals(this.elems, ((Tuple) that).elems));
  }

  @Override
  @SuppressWarnings("unchecked")
  public E get(int i) {
    return (E) elems[i]; // unchecked
  }

  @Override
  public int size() {
    return elems.length;
  }

  @Override
  public Tuple<E> subList(int from, int to) {
    return wrap(Arrays.copyOfRange(elems, from, to));
  }

  @Override
  public Object[] toArray() {
    return Arrays.copyOf(elems, elems.length);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    // TODO(adonovan): when SkylarkPrinter moves into this package,
    // inline and simplify this, the sole call with isTuple=true.
    printer.printList(this, /*isTuple=*/ true);
  }

  // TODO(adonovan): SkylarkValue has 3 String methods yet still we need this fourth. Why?
  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  @Override
  public ImmutableList<E> getImmutableList() {
    // Share the array with this (immutable) Tuple.
    return wrapImmutable(elems);
  }

  /**
   * Returns a new ImmutableList<T> backed by {@code array}, which must not be subsequently
   * modified.
   */
  // TODO(adonovan): move this somewhere more appropriate.
  static <T> ImmutableList<T> wrapImmutable(Object[] array) {
    // Construct an ImmutableList that shares the array.
    // ImmutableList relies on the implementation of Collection.toArray
    // not subsequently modifying the returned array.
    return ImmutableList.copyOf(
        new AbstractCollection<T>() {
          @Override
          public Object[] toArray() {
            return array;
          }

          @Override
          public int size() {
            return array.length;
          }

          @Override
          public Iterator<T> iterator() {
            throw new UnsupportedOperationException();
          }
        });
  }

  @Override
  public Tuple<E> getSlice(
      Object start, Object end, Object step, Location loc, Mutability mutability)
      throws EvalException {
    // TODO(adonovan): opt: this is horribly inefficient.
    List<Integer> indices = EvalUtils.getSliceIndices(start, end, step, this.size(), loc);
    Object[] res = new Object[indices.size()];
    for (int i = 0; i < indices.size(); ++i) {
      res[i] = elems[indices.get(i)];
    }
    return wrap(res);
  }

  /** Returns a Tuple containing n consecutive repeats of this tuple. */
  public Tuple<E> repeat(int n, Mutability mutability) {
    if (n <= 0 || isEmpty()) {
      return empty();
    }
    // TODO(adonovan): reject unreasonably large n.
    Object[] res = new Object[n * elems.length];
    for (int i = 0; i < n; i++) {
      System.arraycopy(elems, 0, res, i * elems.length, elems.length);
    }
    return wrap(res);
  }
}
