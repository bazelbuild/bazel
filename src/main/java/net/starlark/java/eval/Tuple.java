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
import com.google.common.collect.ObjectArrays;
import java.util.AbstractCollection;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.Iterator;
import net.starlark.java.annot.StarlarkBuiltin;

/** A Tuple is an immutable finite sequence of values. */
@StarlarkBuiltin(
    name = "tuple",
    category = "core",
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
public abstract class Tuple extends AbstractList<Object>
    implements Sequence<Object>, Comparable<Tuple> {

  // Prohibit instantiation outside of package.
  Tuple() {}

  /** Returns the empty tuple. */
  public static Tuple empty() {
    return RegularTuple.EMPTY;
  }

  /** Returns a Tuple that wraps the specified array, which must not be subsequently modified. */
  static Tuple wrap(Object[] array) {
    switch (array.length) {
      case 0:
        return RegularTuple.EMPTY;
      case 1:
        return new SingletonTuple(array[0]);
      default:
        return new RegularTuple(array);
    }
  }

  /** Returns a tuple containing the given elements. */
  public static Tuple copyOf(Iterable<?> seq) {
    if (seq instanceof Tuple) {
      return (Tuple) seq;
    }
    return wrap(Iterables.toArray(seq, Object.class));
  }

  /** Returns a tuple containing the given elements. */
  public static Tuple of(Object... elems) {
    return wrap(Arrays.copyOf(elems, elems.length));
  }

  /** Returns a two-element tuple. */
  public static Tuple pair(Object a, Object b) {
    // Equivalent to of(a, b) but avoids variadic array allocation.
    return wrap(new Object[] {a, b});
  }

  /** Returns a three-element tuple. */
  public static Tuple triple(Object a, Object b, Object c) {
    // Equivalent to of(a, b, c) but avoids variadic array allocation.
    return wrap(new Object[] {a, b, c});
  }

  /** Returns a tuple that is the concatenation of two tuples. */
  public static Tuple concat(Tuple x, Tuple y) {
    if (x.isEmpty()) {
      return y;
    } else if (y.isEmpty()) {
      return x;
    } else {
      Object[] xelems =
          x instanceof SingletonTuple
              ? new Object[] {((SingletonTuple) x).elem}
              : ((RegularTuple) x).elems;
      Object[] yelems =
          y instanceof SingletonTuple
              ? new Object[] {((SingletonTuple) y).elem}
              : ((RegularTuple) y).elems;
      return wrap(ObjectArrays.concat(xelems, yelems, Object.class));
    }
  }

  @Override
  public int compareTo(Tuple that) {
    return Sequence.compare(this, that);
  }

  @Override
  public boolean equals(Object that) {
    // This slightly violates the java.util.List equivalence contract
    // because it considers the class, not just the elements.
    // This is needed because in Starlark tuples are never equal to lists, however in Java they both
    // implement List interface.
    return this == that || (that instanceof Tuple && Sequence.sameElems(this, ((Tuple) that)));
  }

  // TODO(adonovan): StarlarkValue has 3 String methods yet still we need this fourth. Why?
  @Override
  public String toString() {
    return Starlark.repr(this);
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

  /** Returns a Tuple containing n consecutive repeats of this tuple. */
  abstract Tuple repeat(StarlarkInt n) throws EvalException;
}
