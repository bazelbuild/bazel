// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.concurrent.PooledInterner;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.util.TestType;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.Serializable;
import javax.annotation.Nullable;

/**
 * A {@link SkyKey} is effectively a pair (type, name) that identifies a Skyframe value.
 *
 * <p>SkyKey implementations are heavily used as map keys. Thus, they should have fast {@link
 * #hashCode} implementations (cached if necessary). The same SkyKey may be created multiple times
 * by different {@code SkyFunction}s requesting it, and so it should have effective interning. There
 * will likely be more SkyKeys on the JVM heap than any other non-native type, so be mindful of
 * memory usage (in particular object wrapper size and memory alignment)! Typically the
 * implementation should have a fixed {@link #functionName} implementation and return itself as the
 * {@link #argument} in order to reduce the cost of wrapper objects.
 */
public interface SkyKey extends Serializable {

  /** Returns the canonical representation of the key as a string. */
  default String getCanonicalName() {
    return String.format("%s:%s", functionName(), argument().toString().replace('\n', '_'));
  }

  SkyFunctionName functionName();

  default Object argument() {
    return this;
  }

  /**
   * Returns {@code true} if this key produces a {@link SkyValue} that can be reused across builds.
   *
   * <p>Values may be unshareable because they are just not serializable, or because they contain
   * data that cannot safely be reused as-is by another invocation, such as stamping information or
   * "flaky" values like test statuses.
   *
   * <p>Unshareable data should not be serialized, since it will never be reused. Attempts to fetch
   * a key's serialized data will call this method and only perform the fetch if it returns {@code
   * true}.
   *
   * <p>The result of this method only applies to non-error values. In case of an error, {@link
   * ErrorInfo#isTransitivelyTransient()} can be used to determine shareability.
   */
  default boolean valueIsShareable() {
    return true;
  }

  /**
   * Returns {@code true} if this key's {@link SkyFunction} would like Skyframe to schedule its
   * reevaluation when any of its previously requested unfinished deps completes. Otherwise,
   * Skyframe will schedule reevaluation only when all previously requested unfinished deps
   * complete.
   */
  default boolean supportsPartialReevaluation() {
    return false;
  }

  @Nullable
  default SkyKeyInterner<?> getSkyKeyInterner() {
    return null;
  }

  static <T extends SkyKey> SkyKeyInterner<T> newInterner() {
    return new SkyKeyInterner<>();
  }

  /** {@link PooledInterner} for {@link SkyKey}s. */
  final class SkyKeyInterner<T extends SkyKey> extends PooledInterner<T> {
    @Nullable static Pool<? extends SkyKey> globalPool = null;

    /**
     * Sets the {@link Pool} to be used for interning.
     *
     * <p>The pool is strongly retained until it is cleared, which can be accomplished by passing
     * {@code null} to this method.
     */
    @ThreadSafety.ThreadCompatible
    static void setGlobalPool(@Nullable Pool<SkyKey> pool) {
      // No synchronization is needed. Setting global pool is guaranteed to happen sequentially
      // since only one build can happen at the same time.
      if (pool != null
          && globalPool != null
          && (!TestType.isInTest() || TestType.getTestType() == TestType.SHELL_INTEGRATION)) {
        BugReport.sendNonFatalBugReport(
            new IllegalStateException("Global SkyKey pool not cleared before setting another"));
      }
      globalPool = pool;
    }

    @Override
    @SuppressWarnings("unchecked")
    protected Pool<T> getPool() {
      return (Pool<T>) globalPool;
    }

    /**
     * Call {@link #weakInternUnchecked(SkyKey)} on {@link SkyKeyInterner} returned by {@code
     * key.getSkyKeyInterner}. This method is created to remove casts and
     * {@code @SuppressWarnings("unchecked")} in callers and put them in one place.
     */
    @CanIgnoreReturnValue
    @SuppressWarnings("unchecked")
    T weakInternUnchecked(SkyKey sample) {
      return weakIntern((T) sample);
    }
  }
}
