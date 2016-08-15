// Copyright 2016 The Bazel Authors. All rights reserved.
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

import javax.annotation.Nullable;

/** Supplier that may throw {@link InterruptedException} when value is retrieved. */
public interface InterruptibleSupplier<T> {
  T get() throws InterruptedException;

  class Instance<T> implements InterruptibleSupplier<T> {
    private final T instance;

    public Instance(T instance) {
      this.instance = instance;
    }

    @Override
    public T get() {
      return instance;
    }
  }

  class Memoize<T> implements InterruptibleSupplier<T> {
    private final InterruptibleSupplier<T> delegate;
    private @Nullable T value = null;

    private Memoize(InterruptibleSupplier<T> delegate) {
      this.delegate = delegate;
    }

    public static <S> InterruptibleSupplier<S> of(InterruptibleSupplier<S> delegate) {
      if (delegate instanceof Memoize) {
        return delegate;
      }
      return new Memoize<>(delegate);
    }

    @Override
    public T get() throws InterruptedException {
      if (value != null) {
        return value;
      }
      synchronized (this) {
        if (value == null) {
          value = delegate.get();
        }
      }
      return value;
    }
  }
}
