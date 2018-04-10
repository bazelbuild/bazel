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

import com.google.common.annotations.VisibleForTesting;
import javax.annotation.Nullable;

/**
 * Wrapper for a value or the untyped exception thrown when trying to compute it.
 *
 * <p>This is an implementation detail of {@link ParallelEvaluator}. It's an abstract class (as
 * opposed to an interface) to avoid exposing the methods outside the package.
 */
@VisibleForTesting
public abstract class ValueOrUntypedException {
  /** Returns the stored value, if there was one. */
  @Nullable
  abstract SkyValue getValue();

  /** Returns the stored exception, if there was one. */
  @Nullable
  abstract Exception getException();

  public static ValueOrUntypedException ofValueUntyped(SkyValue value) {
    return new ValueOrUntypedExceptionImpl(value);
  }

  public static ValueOrUntypedException ofNull() {
    return ValueOrUntypedExceptionImpl.NULL;
  }

  public static ValueOrUntypedException ofExn(Exception e) {
    return new ValueOrUntypedExceptionExnImpl(e);
  }

  private static class ValueOrUntypedExceptionImpl extends ValueOrUntypedException {
    static final ValueOrUntypedExceptionImpl NULL = new ValueOrUntypedExceptionImpl(null);
    @Nullable private final SkyValue value;

    ValueOrUntypedExceptionImpl(@Nullable SkyValue value) {
      this.value = value;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return value;
    }

    @Override
    public Exception getException() {
      return null;
    }
  }

  private static class ValueOrUntypedExceptionExnImpl extends ValueOrUntypedException {
    private final Exception exception;

    ValueOrUntypedExceptionExnImpl(Exception exception) {
      this.exception = exception;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }

    @Override
    public Exception getException() {
      return exception;
    }
  }
}
