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
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import javax.annotation.Nullable;

/** An immutable singleton implementation of a {@code StarlarkList}. */
public final class ImmutableSingletonStarlarkList<E> extends ImmutableStarlarkList<E> {
  private Object elem; // non-final and nullable prior to deserialization completion

  public ImmutableSingletonStarlarkList(@Nullable Object elem) {
    this.elem = elem;
    // `elem` is non-final to support async deserialization. This fence replicates the publication
    // guarantee HotSpot gives final fields: instances may be published racily (e.g. by
    // unsafeOptimizeMemoryLayout() callers that store into a shared table), and readers must
    // observe the initialized element via dependent-load ordering.
    VarHandle.storeStoreFence();
  }

  public void setElementForDeserialization(Object elem) {
    this.elem = elem;
  }

  // The element is only ever set from a value of type E.
  @Override
  @SuppressWarnings("unchecked")
  public ImmutableList<E> getImmutableList() {
    return ImmutableList.of((E) elem);
  }

  // The element is only ever set from a value of type E.
  @Override
  @SuppressWarnings("unchecked")
  public E get(int i) {
    Preconditions.checkElementIndex(i, 1);
    return (E) elem; // unchecked
  }

  @Override
  public int size() {
    return 1;
  }

  @Override
  int computeHashCode() {
    return 6047 + 4673 * (31 + elem.hashCode());
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return Starlark.checkedEquals(key, elem);
  }

  @Override
  public boolean contains(Object o) {
    // StarlarkList contains only valid Starlark objects (which are non-null)
    if (o == null) {
      return false;
    }
    return o.equals(elem);
  }

  /** Returns a new array of class Object[] containing the list elements. */
  @Override
  public Object[] toArray() {
    return new Object[] {elem};
  }

  // Generic arrays cannot be created without unchecked casts. Per the Collection#toArray(T[])
  // contract, the caller guarantees that T is a supertype of the elements' runtime types; stores
  // into `a` are additionally checked by the JVM at runtime.
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
  public Object[] elems() {
    return new Object[] {elem};
  }
}
