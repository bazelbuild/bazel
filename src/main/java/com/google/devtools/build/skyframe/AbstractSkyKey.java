// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;

/**
 * For use when the {@link #argument} of the {@link SkyKey} cannot be a {@link SkyKey} itself,
 * either because it is a type like List or because it is already a different {@link SkyKey}.
 * Provides convenient boilerplate.
 */
public abstract class AbstractSkyKey<T> implements SkyKey {
  // Visible for serialization.
  protected final T arg;
  /**
   * Cache the hash code for this object. It might be expensive to compute. It is transient because
   * argument's hash code might not be stable across JVM instances.
   */
  private transient int hashCode;

  protected AbstractSkyKey(T arg) {
    this.arg = Preconditions.checkNotNull(arg);
  }

  @Override
  public final int hashCode() {
    // We use the hash code caching strategy employed by java.lang.String. There are three subtle
    // things going on here:
    //
    // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
    // Yes, this means that if the hash code is really 0 then we will "recompute" it each time. But
    // this isn't a problem in practice since a hash code of 0 should be rare.
    //
    // (2) Since we have no synchronization, multiple threads can race here thinking there are the
    // first one to compute and cache the hash code.
    //
    // (3) Moreover, since 'hashCode' is non-volatile, the cached hash code value written from one
    // thread may not be visible by another.
    //
    // All three of these issues are benign from a correctness perspective; in the end we have no
    // overhead from synchronization, at the cost of potentially computing the hash code more than
    // once.
    int h = hashCode;
    if (h == 0) {
      h = computeHashCode();
      hashCode = h;
    }
    return h;
  }

  @Override
  public final T argument() {
    return arg;
  }

  private int computeHashCode() {
    return 31 * functionName().hashCode() + arg.hashCode();
  }

  @Override
  public final boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    AbstractSkyKey<?> that = (AbstractSkyKey<?>) obj;
    return this.functionName().equals(that.functionName()) && this.arg.equals(that.arg);
  }

  @Override
  public String toString() {
    return functionName() + ":" + arg;
  }
}
