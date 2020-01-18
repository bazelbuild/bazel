// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;

/**
 * Protects against excessive memory consumption when the same transition applies multiple times.
 *
 * <p>For example: an exec transition to {@code //my:exec_platform} for a tool that every rule in
 * the target configuration depends on.
 *
 * <p>Specifically, if {@code (origOptions1, context1)} produces {@code toOptions1}, {@code
 * (origOptions2, context2)} produces {@code toOptions2}, {@code origOptions1 == origOptions2}, and
 * {@code context1.equals(context2)}, this guarantees that {@code toOptions1 == toOptions2}.
 *
 * <p>This means applying the same transition to the same source multiple times always returns the
 * same reference.
 *
 * <p>Theoretically, constructing a unique {@link BuildOptions} instance should be cheap. But {@link
 * BuildOptions#diffForReconstructionCache} keeps references to every instance. It's not hard to get
 * a build graph with hundreds of thousands of nodes, and persisting hundreds of thousands of {@link
 * BuildOptions} instances consumes gigabytes of memory.
 *
 * <p>While {@link BuildOptions#diffForReconstructionCache}'s references are theoretically
 * garbage-collectible through {@link java.lang.ref.WeakReference} and {@link
 * java.lang.ref.SoftReference} wrappers, garbage collection might not react fast enough to rapidly
 * constructing builds to prevent OOMs.
 */
public class BuildOptionsCache<T> {
  private final Cache<ReferenceCacheKey, BuildOptions> cache;

  public BuildOptionsCache() {
    cache = CacheBuilder.newBuilder().build();
  }

  /**
   * Applies the given transition to the given {@code (fromOptions, context)} pair. Returns an
   * existing {@link BuildOptions} instance if one is already associated with that key. Else
   * constructs and caches a new {@link BuildOptions} instance using the given transition function.
   *
   * @param fromOptions the starting options
   * @param context an additional object that affects the transition's result
   * @param transitionFunc the transition to apply if {@code (fromOptions, context)} isn't cached
   */
  public BuildOptions applyTransition(
      BuildOptions fromOptions, T context, Supplier<BuildOptions> transitionFunc) {
    try {
      return cache.get(new ReferenceCacheKey(fromOptions, context), () -> transitionFunc.get());
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Helper class for matching ({@link BuildOptions}, {@link Object}) cache keys by {@link
   * BuildOptions} reference.
   *
   * <p>This is sufficient for a practically effective cache, and saves having the potentially
   * expensive {@link BuildOptions#maybeInitializeFingerprintAndHashCode} that {@link
   * BuildOptions#equals} calls.
   */
  private static final class ReferenceCacheKey {
    private final BuildOptions options;
    private final Object context;

    ReferenceCacheKey(BuildOptions options, Object context) {
      this.options = options;
      this.context = context;
    }

    @Override
    public String toString() {
      return String.format("ReferenceCacheKey(%s, %s)", options, context);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ReferenceCacheKey)) {
        return false;
      }
      ReferenceCacheKey casted = (ReferenceCacheKey) other;
      // See class javadoc above: we intentionally compare BuildOptions by reference because it
      // avoids potentially expensive .equals() computation and practically caches just as well
      // because the memory problem we're trying to solve is the same transition applying to the
      // same BuildOptions instance over and over again.
      @SuppressWarnings("ReferenceEquality")
      boolean match = this.options == casted.options && this.context.equals(casted.context);
      return match;
    }

    @Override
    public int hashCode() {
      // Computing BuildOptions.hashCode is potentially expensive and its reference is sufficient,
      // so we just hash its reference.
      return Objects.hash(System.identityHashCode(options), context);
    }
  }
}
