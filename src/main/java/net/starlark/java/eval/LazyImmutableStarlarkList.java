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
import java.util.function.Supplier;

/** An immutable {@link StarlarkList} that lazily invokes a supplier to obtain its elements. */
final class LazyImmutableStarlarkList<E> extends ImmutableStarlarkList<E> {
  private Supplier<ImmutableList<E>> supplier;
  private volatile Object[] elems;

  LazyImmutableStarlarkList(Supplier<ImmutableList<E>> supplier) {
    this.supplier = supplier;
  }

  @Override
  public int size() {
    return elems().length;
  }

  @Override
  @SuppressWarnings("unchecked")
  public E get(int i) {
    Object[] elems = elems();
    Preconditions.checkElementIndex(i, elems.length);
    return (E) elems[i];
  }

  @Override
  Object[] elems() {
    if (elems == null) {
      synchronized (this) {
        if (elems == null) {
          elems = supplier.get().toArray();
          supplier = null;
        }
      }
    }
    return elems;
  }

  @Override
  public StarlarkList<E> unsafeOptimizeMemoryLayout() {
    if (elems != null) {
      return StarlarkList.wrap(Mutability.IMMUTABLE, elems);
    }
    return this;
  }
}
