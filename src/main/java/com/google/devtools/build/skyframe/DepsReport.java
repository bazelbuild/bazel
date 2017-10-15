// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.collect.Iterators;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

/**
 * Result of {@link EvaluableGraph#analyzeDepsDoneness}: Equivalent to an {@code
 * Optional<Collection<SkyKey>>} but without the overhead of the wrapper {@code Optional}.
 */
public class DepsReport implements Collection<SkyKey> {
  public static final DepsReport NO_INFORMATION = new DepsReport(-1, null);

  private final int size;
  /** Note that this array may have trailing null elements past {@link #size}. */
  private final SkyKey[] arr;

  private DepsReport(int size, SkyKey[] arr) {
    this.size = size;
    this.arr = arr;
  }

  boolean hasInformation() {
    return arr != null;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public boolean isEmpty() {
    return size == 0;
  }

  @Override
  public Iterator<SkyKey> iterator() {
    return Iterators.limit(Iterators.forArray(arr), size);
  }

  private UnsupportedOperationException throwUnsupported() {
    throw new UnsupportedOperationException(this.toString());
  }

  @Override
  public boolean contains(Object o) {
    throw throwUnsupported();
  }

  @Override
  public Object[] toArray() {
    throw throwUnsupported();
  }

  @Override
  public <T> T[] toArray(T[] a) {
    throw throwUnsupported();
  }

  @Override
  public boolean add(SkyKey skyKey) {
    throw throwUnsupported();
  }

  @Override
  public boolean remove(Object o) {
    throw throwUnsupported();
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    throw throwUnsupported();
  }

  @Override
  public boolean addAll(Collection<? extends SkyKey> c) {
    throw throwUnsupported();
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    throw throwUnsupported();
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    throw throwUnsupported();
  }

  @Override
  public void clear() {
    throw throwUnsupported();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("size", size)
        .add("arr", Arrays.toString(arr))
        .toString();
  }

  /** Builder for {@link DepsReport}. */
  public static class Builder {
    private int size = 0;
    private final SkyKey[] arr;

    public Builder(int maxSize) {
      arr = new SkyKey[maxSize];
    }

    public void add(SkyKey key) {
      if (size >= arr.length) {
        throw new IllegalStateException("Too many adds: " + key + ", " + this);
      }
      arr[size] = key;
      size++;
    }

    public DepsReport build() {
      return new DepsReport(size, arr);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("size", size)
          .add("arr", Arrays.toString(arr))
          .toString();
    }
  }
}
