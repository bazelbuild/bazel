// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe.state;

import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import javax.annotation.Nullable;

/** An environment for post-evaluation queries and tests. */
public final class EnvironmentForUtilities
    implements SkyFunction.LookupEnvironment, SkyframeLookupResult {
  /** Provides results. */
  public interface ResultProvider {
    /**
     * Returns {@link SkyValue} or {@link Exception} for {@code key}.
     *
     * <p>May return null for the following reasons.
     *
     * <ul>
     *   <li>The result is not yet determined, possibly due to failing fast.
     *   <li>The result was part of a cycle error.
     * </ul>
     */
    @Nullable
    Object getValueOrException(SkyKey key);
  }

  private final ResultProvider resultProvider;

  public EnvironmentForUtilities(ResultProvider resultProvider) {
    this.resultProvider = resultProvider;
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey depKey) {
    return getValueOrThrow(depKey, null, null, null, null);
  }

  @Nullable
  @Override
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass1)
      throws E {
    return getValueOrThrow(depKey, exceptionClass1, null, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2 {
    return getValueOrThrow(depKey, exceptionClass1, exceptionClass2, null, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3 {
    return getValueOrThrow(depKey, exceptionClass1, exceptionClass2, exceptionClass3, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4 {
    Object result = resultProvider.getValueOrException(depKey);
    if (result == null) {
      return null;
    }
    if (result instanceof SkyValue) {
      return (SkyValue) result;
    }
    if (exceptionClass1 != null && exceptionClass1.isInstance(result)) {
      throw exceptionClass1.cast(result);
    }
    if (exceptionClass2 != null && exceptionClass2.isInstance(result)) {
      throw exceptionClass2.cast(result);
    }
    if (exceptionClass3 != null && exceptionClass3.isInstance(result)) {
      throw exceptionClass3.cast(result);
    }
    if (exceptionClass4 != null && exceptionClass4.isInstance(result)) {
      throw exceptionClass4.cast(result);
    }
    return null;
  }

  @Override
  public SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> unused) {
    return this;
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    return this;
  }

  // -------------------- SkyframeLookupResult Implementation --------------------

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey,
      @Nullable Class<E1> exceptionClass1,
      @Nullable Class<E2> exceptionClass2,
      @Nullable Class<E3> exceptionClass3)
      throws E1, E2, E3 {
    return getValueOrThrow(skyKey, exceptionClass1, exceptionClass2, exceptionClass3, null);
  }

  @Override
  public boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
    Object result = resultProvider.getValueOrException(key);
    if (result == null) {
      return false;
    }
    if (result instanceof SkyValue) {
      resultCallback.acceptValue(key, (SkyValue) result);
      return true;
    }
    if (resultCallback.tryHandleException(key, (Exception) result)) {
      return true;
    }
    return false;
  }
}
