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

import com.google.common.base.Preconditions;
import javax.annotation.Nullable;

/**
 * An {@link InterruptibleSupplier} which should cache the instance retrieved during the first call
 * to {@link #get} and returns that value on subsequent calls to {@link #get}.
 *
 * <p>This is similar to, but not exactly the same as, what is returned by {@link
 * com.google.common.base.Suppliers#memoize}.
 *
 * <p>Implementations should be thread-safe.
 *
 * <p>Unlike that implementation, this is not serializable, and its initialized state (whether an
 * instance has been retrieved) is visible via {@link #isInitialized}.
 */
public interface MemoizingInterruptibleSupplier<T> extends InterruptibleSupplier<T> {

  /** Returns {@code true} if the result of {@link #get} is readily available. */
  boolean isInitialized();

  static <T> MemoizingInterruptibleSupplier<T> of(InterruptibleSupplier<T> delegate) {
    if (delegate instanceof MemoizingInterruptibleSupplier) {
      return (MemoizingInterruptibleSupplier<T>) delegate;
    }
    return new DelegatingMemoizingSupplier<>(delegate);
  }

  /** Memoizes the result of {@code delegate} after the first call to {@link #get}. */
  final class DelegatingMemoizingSupplier<T> implements MemoizingInterruptibleSupplier<T> {

    @Nullable private InterruptibleSupplier<T> delegate;
    @Nullable private volatile T value = null;

    private DelegatingMemoizingSupplier(InterruptibleSupplier<T> delegate) {
      this.delegate = Preconditions.checkNotNull(delegate);
    }

    @Override
    public T get() throws InterruptedException {
      if (value != null) {
        return value;
      }
      synchronized (this) {
        if (value == null) {
          value = delegate.get();
          delegate = null; // Free up for GC.
        }
      }
      return value;
    }

    @Override
    public boolean isInitialized() {
      return value != null;
    }
  }
}
