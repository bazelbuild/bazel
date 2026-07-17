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
 */
public abstract class AbstractSkyKey<T> implements SkyKey {

  // Visible for serialization.
  protected final T arg;

  protected AbstractSkyKey(T arg) {
    this.arg = checkNotNull(arg);
  }

  @Override
  public final T argument() {
    return arg;
  }

  @Override
  public int hashCode() {
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
    if (this instanceof WithCachedHashCode && hashCode() != obj.hashCode()) {
      return false;
    }
    AbstractSkyKey<?> that = (AbstractSkyKey<?>) obj;
    return functionName().equals(that.functionName()) && arg.equals(that.arg);
  }

  @Override
  public String toString() {
    return functionName() + ":" + arg;
  }

  /**
   * An {@link AbstractSkyKey} that computes and caches its hash code upon creation.
   *
   * <p>Only subclass this class when caching the hash code is worth spending a field on it. If the
   * hash code computation for the key's argument is already fast, just subclass {@link
   * AbstractSkyKey} to save memory.
   */
  public abstract static class WithCachedHashCode<T> extends AbstractSkyKey<T> {
    private final transient int hashCode;

    protected WithCachedHashCode(T arg) {
      super(arg);
      this.hashCode = super.hashCode();
    }

    @Override
    public final int hashCode() {
      return hashCode;
    }
  }
}
