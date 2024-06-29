// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.Arrays;

/**
 * An implementation of non-singleton (empty or more than 1 element) Tuple using an {@code Object}
 * array.
 */
final class RegularTuple extends Tuple {

  // The shared (sole) empty tuple.
  static final Tuple EMPTY = new RegularTuple(new Object[] {});

  final Object[] elems;

  RegularTuple(Object[] elems) {
    Preconditions.checkArgument(elems.length != 1);
    this.elems = elems;
  }

  @Override
  public boolean isImmutable() {
    for (Object x : elems) {
      if (!Starlark.isImmutable(x)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void checkHashable() throws EvalException {
    for (Object x : elems) {
      Starlark.checkHashable(x);
    }
  }

  @Override
  public int hashCode() {
    return 9857 + 8167 * Arrays.hashCode(elems);
  }

  @Override
  public Object get(int i) {
    return elems[i];
  }

  @Override
  public int size() {
    return elems.length;
  }

  @Override
  public boolean contains(Object o) {
    // Tuple contains only valid Starlark objects (which are non-null)
    if (o == null) {
      return false;
    }
    for (Object elem : elems) {
      if (o.equals(elem)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Tuple subList(int from, int to) {
    Preconditions.checkPositionIndexes(from, to, elems.length);
    return wrap(Arrays.copyOfRange(elems, from, to));
  }

  /** Returns a new array of class Object[] containing the tuple elements. */
  @Override
  public Object[] toArray() {
    return elems.length != 0 ? Arrays.copyOf(elems, elems.length, Object[].class) : elems;
  }

  @SuppressWarnings("unchecked")
  @Override
  public <T> T[] toArray(T[] a) {
    if (a.length < elems.length) {
      return (T[]) Arrays.copyOf(elems, elems.length, a.getClass());
    } else {
      System.arraycopy(elems, 0, a, 0, elems.length);
      Arrays.fill(a, elems.length, a.length, null);
      return a;
    }
  }

  @Override
  public void repr(Printer printer) {
    // Remark: singletons, are represented as {@code '(x,)'}; whereas this implementation would
    // return
    // {@code '(x)'}
    printer.append('(');
    String sep = "";
    for (Object elem : elems) {
      printer.append(sep);
      sep = ", ";
      printer.repr(elem);
    }
    printer.append(')');
  }

  @Override
  public ImmutableList<Object> getImmutableList() {
    // Share the array with this (immutable) Tuple.
    return wrapImmutable(elems);
  }

  @Override
  public Tuple getSlice(Mutability mu, int start, int stop, int step) throws EvalException {
    RangeList indices = new RangeList(start, stop, step);
    int n = indices.size();
    if (step == 1) { // common case
      return subList(indices.at(0), indices.at(n));
    }
    Object[] res = new Object[n];
    for (int i = 0; i < n; ++i) {
      res[i] = elems[indices.at(i)];
    }
    return wrap(res);
  }

  @Override
  Tuple repeat(StarlarkInt n) throws EvalException {
    if (n.signum() <= 0 || isEmpty()) {
      return empty();
    }

    int ni = n.toInt("repeat");
    long sz = (long) ni * elems.length;
    if (sz > StarlarkList.MAX_ALLOC) {
      throw Starlark.errorf("excessive repeat (%d * %d elements)", elems.length, ni);
    }
    Object[] res = new Object[(int) sz];
    for (int i = 0; i < ni; i++) {
      System.arraycopy(elems, 0, res, i * elems.length, elems.length);
    }
    return wrap(res);
  }
}
