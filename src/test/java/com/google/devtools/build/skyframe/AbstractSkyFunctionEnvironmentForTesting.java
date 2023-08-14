// Copyright 2022 The Bazel Authors. All rights reserved.
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

import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Partial {@link SkyFunction.Environment} implementation that allows tests to deal with simple
 * types like {@link ValueOrUntypedException}, {@link Map}, and {@link List}.
 */
public abstract class AbstractSkyFunctionEnvironmentForTesting
    extends AbstractSkyFunctionEnvironment {

  /**
   * Gets a single value or exception.
   *
   * <p>Implementations should set {@link #valuesMissing} as necessary.
   */
  protected abstract ValueOrUntypedException getSingleValueOrUntypedException(SkyKey depKey)
      throws InterruptedException;

  /**
   * Gets a map of values or exceptions.
   *
   * <p>Implementations should set {@link #valuesMissing} as necessary.
   */
  protected abstract Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException;

  /**
   * Gets a list of values or exceptions parallel to the given keys.
   *
   * <p>Implementations should set {@link #valuesMissing} as necessary.
   */
  protected abstract List<ValueOrUntypedException> getOrderedValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException;

  @Nullable
  @Override
  final <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrowInternal(
          SkyKey depKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    ValueOrUntypedException voe = getSingleValueOrUntypedException(depKey);
    SkyValue value = voe.getValue();
    if (value != null) {
      return value;
    }
    SkyFunctionException.throwIfInstanceOf(
        voe.getException(), exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
    valuesMissing = true;
    return null;
  }

  @Override
  public final SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    return new SimpleSkyframeLookupResult(() -> valuesMissing = true, valuesOrExceptions::get);
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    throw new UnsupportedOperationException();
  }
}
