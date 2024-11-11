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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Wrapper around {@link Interners}, with Blaze-specific predetermined concurrency levels. */
public class BlazeInterners {
  private static final int DEFAULT_CONCURRENCY_LEVEL = Runtime.getRuntime().availableProcessors();
  private static final int CONCURRENCY_LEVEL;

  static {
    String val = System.getenv("BLAZE_INTERNER_CONCURRENCY_LEVEL");
    CONCURRENCY_LEVEL = (val == null) ? DEFAULT_CONCURRENCY_LEVEL : Integer.parseInt(val);
  }

  public static int concurrencyLevel() {
    return CONCURRENCY_LEVEL;
  }

  /**
   * Creates an interner which retains a weak reference to each instance it has interned.
   *
   * <p>It is preferred to use {@code SkyKey#SkyKeyInterner} instead for interning {@code SkyKey}
   * types.
   */
  public static <T> Interner<T> newWeakInterner() {
    return Interners.newBuilder().concurrencyLevel(CONCURRENCY_LEVEL).weak().build();
  }

  public static <T> Interner<T> newStrongInterner() {
    return new StrongInterner<>();
  }

  /**
   * Interner based on {@link ConcurrentHashMap}, which offers faster lookups than Guava's strong
   * interner.
   */
  private static final class StrongInterner<T> implements Interner<T> {
    private final Map<T, T> map = new ConcurrentHashMap<>(CONCURRENCY_LEVEL);

    @Override
    public T intern(T sample) {
      T existing = map.putIfAbsent(sample, sample);
      return firstNonNull(existing, sample);
    }
  }
}
