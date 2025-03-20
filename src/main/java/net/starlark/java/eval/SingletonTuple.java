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

/** A specific implementation of a Tuple that has exactly 1 element. */
final class SingletonTuple extends Tuple {
  final Object elem;

  SingletonTuple(Object elem) {
    this.elem = elem;
  }

  @Override
  public boolean isImmutable() {
    return Starlark.isImmutable(elem);
  }

  @Override
  public void checkHashable() throws EvalException {
    Starlark.checkHashable(elem);
  }

  @Override
  public int hashCode() {
    // This produces the same results as RegularTuple.hashCode(),
    return 9857 + 8167 * (31 + elem.hashCode());
  }

  @Override
  public Object get(int i) {
    Preconditions.checkElementIndex(i, 1);
    return elem;
  }

  @Override
  public int size() {
    return 1;
  }

  @Override
  public boolean contains(Object o) {
    // Tuple contains only valid Starlark objects (which are non-null)
    if (o == null) {
      return false;
    }
    return o.equals(elem);
  }

  @Override
  public Tuple subList(int from, int to) {
    Preconditions.checkPositionIndexes(from, to, 1);
    return from <= 0 && to >= 1 ? this : empty();
  }

  /** Returns a new array of class Object[] containing the tuple elements. */
  @Override
  public Object[] toArray() {
    return new Object[] {elem};
  }

  @SuppressWarnings("unchecked")
  @Override
  public <T> T[] toArray(T[] a) {
    if (a.length < 1) {
      return (T[]) new Object[] {elem};
    } else {
      a[0] = (T) elem;
      Arrays.fill(a, 1, a.length, null);
      return a;
    }
  }

  @Override
  public void repr(Printer printer) {
    printer.append('(').repr(elem).append(",)");
  }

  @Override
  public ImmutableList<Object> getImmutableList() {
    return ImmutableList.of(elem);
  }

  @Override
  public Tuple getSlice(Mutability mu, int start, int stop, int step) throws EvalException {
    RangeList indices = new RangeList(start, stop, step);
    return indices.isEmpty() ? Tuple.empty() : this;
  }

  @Override
  Tuple repeat(StarlarkInt n) throws EvalException {
    if (n.signum() <= 0) {
      return empty();
    }

    int ni = n.toInt("repeat");
    if (ni > StarlarkList.MAX_ALLOC) {
      throw Starlark.errorf("excessive repeat (%d * %d elements)", 1, ni);
    }
    Object[] res = new Object[ni];
    Arrays.fill(res, 0, ni, elem);
    return wrap(res);
  }
}
