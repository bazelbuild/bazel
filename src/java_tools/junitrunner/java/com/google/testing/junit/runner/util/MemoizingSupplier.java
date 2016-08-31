// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

/**
 * Returns a {@link Supplier} which caches the instance retrieved during the first call to
 * {@code get()} and returns that value on subsequent calls to {@code get()}. See:
 * <a href="http://en.wikipedia.org/wiki/Memoization">memoization</a>.
 *
 * <p>The returned supplier is thread-safe. The delegate's {@code get()} method will be invoked at
 * most once.
 *
 * <p>The returned supplier is not serializable.
 */
public class MemoizingSupplier<T> implements Supplier<T> {
  private final Supplier<T> delegate;
  private volatile boolean initialized;
  private T instance;

  public MemoizingSupplier(Supplier<T> delegate) {
    this.delegate = delegate;
  }

  @Override
  public T get() {
    if (!initialized) {
      synchronized (this) {
        if (!initialized) {
          initialized = true;
          instance = delegate.get();
        }
      }
    }
    return instance;
  }
}

