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

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * For use when the {@link #argument} of the {@link SkyKey} cannot be a {@link SkyKey} itself,
 * either because it is a type like List or because it is already a different {@link SkyKey}.
 * Provides convenient boilerplate.
 *
 * <p>An argument's hash code might not be stable across JVM instances if it transitively depends on
 * an object that uses the default identity-based {@link Object#hashCode} or {@link
 * System#identityHashCode}. For this reason, the {@link #hashCode} field is {@code transient}.
 * Subclasses should manage serialization (i.e. using {@code AutoCodec}) to ensure that {@link
 * #AbstractSkyKey(Object)} is invoked on deserialization to freshly compute the hash code.
 */
public abstract class AbstractSkyKey<T> implements SkyKey {

  // Visible for serialization.
  protected final T arg;

  /**
   * Caches the hash code for this object. It might be expensive to compute.
   *
   * <p>The hash code is computed eagerly because it is expected that instances are interned,
   * meaning that {@link #hashCode()} will be called immediately for every instance.
   */
  private final transient int hashCode;

  protected AbstractSkyKey(T arg) {
    this.arg = checkNotNull(arg);
    this.hashCode = 31 * functionName().hashCode() + arg.hashCode();
  }

  @Override
  public final int hashCode() {
    return hashCode;
  }

  @Override
  public final T argument() {
    return arg;
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
    return hashCode == that.hashCode
        && functionName().equals(that.functionName())
        && arg.equals(that.arg);
  }

  @Override
  public String toString() {
    return functionName() + ":" + arg;
  }
}
