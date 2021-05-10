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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import java.util.concurrent.ExecutionException;
import java.util.function.BiFunction;

/**
 * Protects against excessive memory consumption when the same transition applies multiple times.
 *
 * <p>For example: an exec transition to {@code //my:exec_platform} for a tool that every rule in
 * the target configuration depends on.
 *
 * <p>Specifically, if {@code (origOptions1, context1)} produces {@code toOptions1}, {@code
 * (origOptions2, context2)} produces {@code toOptions2}, {@code origOptions1.equals(origOptions2)},
 * and {@code context1.equals(context2)}, this guarantees that {@code toOptions1 == toOptions2},
 * assuming the cache entry has not been evicted.
 *
 * <p>This means applying the same transition to the same source multiple times always returns the
 * same reference.
 *
 * <p>{@link BuildOptions} references are stored softly.
 */
public final class BuildOptionsCache<T> {

  private final Cache<CacheKey, BuildOptions> cache =
      CacheBuilder.newBuilder().softValues().build();

  private final BiFunction<BuildOptionsView, T, BuildOptions> transition;

  public BuildOptionsCache(BiFunction<BuildOptionsView, T, BuildOptions> transition) {
    this.transition = checkNotNull(transition);
  }

  /**
   * Applies the given transition to the given {@code (fromOptions, context)} pair. Returns an
   * existing {@link BuildOptions} instance if one is already associated with that key. Else
   * constructs and caches a new {@link BuildOptions} instance using the given transition function.
   *
   * @param fromOptions the starting options
   * @param context an additional object that affects the transition's result
   */
  public BuildOptions applyTransition(BuildOptionsView fromOptions, T context) {
    try {
      return cache.get(
          new CacheKey(fromOptions.underlying().checksum(), context),
          () -> transition.apply(fromOptions, context));
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Helper class for matching ({@link BuildOptions}, {@link Object}) cache keys by {@link
   * BuildOptions#checksum()}.
   */
  private static final class CacheKey {
    private final String checksum;
    private final Object context;

    CacheKey(String checksum, Object context) {
      this.checksum = checksum;
      this.context = context;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("checksum", checksum)
          .add("context", context)
          .toString();
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      }
      if (!(other instanceof CacheKey)) {
        return false;
      }
      CacheKey casted = (CacheKey) other;
      return checksum.equals(casted.checksum) && context.equals(casted.context);
    }

    @Override
    public int hashCode() {
      return 37 * checksum.hashCode() + context.hashCode();
    }
  }
}
