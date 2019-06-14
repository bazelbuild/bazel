// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.supplier;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.ForOverride;
import java.lang.ref.SoftReference;
import javax.annotation.Nullable;

/**
 * An {@link InterruptibleSupplier} which holds a {@link SoftReference} to a cached value. The value
 * can be evicted from memory by GC, and {@linkplain #computeValue computed} if requested again.
 *
 * <p>It is not guaranteed that the value will be equal to the previously cached one. This behavior
 * is determined by the subclass which implements {@link #computeValue}.
 */
public abstract class EvictableSupplier<T> implements InterruptibleSupplier<T> {

  private volatile SoftReference<T> valueReference;

  /**
   * Creates an {@code EvictableSupplier}.
   *
   * @param cachedValue an already known cached value, or {@code null} if the value should always be
   *     computed on the first call to {@link #get}
   */
  protected EvictableSupplier(@Nullable T cachedValue) {
    this.valueReference = new SoftReference<>(cachedValue);
  }

  @Override
  public final T get() throws InterruptedException {
    T value = valueReference.get();
    if (value != null) {
      return value;
    }

    // Ensure that at most one thread is computing the value.
    synchronized (this) {
      value = valueReference.get();
      if (value != null) {
        return value;
      }

      value = Preconditions.checkNotNull(computeValue());
      valueReference = new SoftReference<>(value);
      return value;
    }
  }

  /**
   * Computes the supplied value.
   *
   * <p>This method is called (under a lock on {@code this}) when the cached value is unavailable,
   * either because it was not initially supplied via the constructor, or because it was evicted by
   * GC.
   *
   * <p>Must not return {@code null}.
   */
  @ForOverride
  protected abstract T computeValue() throws InterruptedException;

  /** Clears the soft reference. Only used in tests. */
  @VisibleForTesting
  public final void evictForTesting() {
    valueReference.clear();
  }

  /**
   * Returns the value if it is currently in memory.
   *
   * <p>If the value is not in memory, {@code null} will be returned. No attempt will be made to
   * {@linkplain #computeValue compute} the value.
   */
  @Nullable
  protected final T peekCachedValue() {
    return valueReference.get();
  }
}
