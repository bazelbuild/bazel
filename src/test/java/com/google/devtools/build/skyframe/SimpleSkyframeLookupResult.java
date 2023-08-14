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

import static com.google.common.base.Preconditions.checkNotNull;

import java.util.function.Function;
import javax.annotation.Nullable;

/** Simple implementation of {@link SkyframeLookupResult}. */
public final class SimpleSkyframeLookupResult implements SkyframeLookupResult {
  private final Runnable valuesMissingCallback;
  private final Function<SkyKey, ValueOrUntypedException> valuesOrExceptions;

  public SimpleSkyframeLookupResult(
      Runnable valuesMissingCallback,
      Function<SkyKey, ValueOrUntypedException> valuesOrExceptions) {
    this.valuesMissingCallback = checkNotNull(valuesMissingCallback);
    this.valuesOrExceptions = checkNotNull(valuesOrExceptions);
  }

  /** Similar to {@link #getOrThrow(SkyKey, Class)}, but takes three exception class parameters. */
  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey,
      @Nullable Class<E1> exceptionClass1,
      @Nullable Class<E2> exceptionClass2,
      @Nullable Class<E3> exceptionClass3)
      throws E1, E2, E3 {
    ValueOrUntypedException voe =
        checkNotNull(valuesOrExceptions.apply(skyKey), "Missing value for %s", skyKey);
    SkyValue value = voe.getValue();
    if (value != null) {
      return value;
    }
    SkyFunctionException.throwIfInstanceOf(
        voe.getException(), exceptionClass1, exceptionClass2, exceptionClass3, null);
    valuesMissingCallback.run();
    return null;
  }

  @Override
  public boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
    ValueOrUntypedException voe =
        checkNotNull(valuesOrExceptions.apply(key), "Missing value for %s", key);
    SkyValue value = voe.getValue();
    if (value != null) {
      resultCallback.acceptValue(key, value);
      return true;
    }
    Exception exception = voe.getException();
    if (exception != null && resultCallback.tryHandleException(key, exception)) {
      return true;
    }
    valuesMissingCallback.run();
    return false;
  }
}
