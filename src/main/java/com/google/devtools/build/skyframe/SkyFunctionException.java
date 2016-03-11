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


import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Base class of exceptions thrown by {@link SkyFunction#compute} on failure.
 *
 * SkyFunctions should declare a subclass {@code C} of {@link SkyFunctionException} whose
 * constructors forward fine-grained exception types (e.g. {@code IOException}) to
 * {@link SkyFunctionException}'s constructor, and they should also declare
 * {@link SkyFunction#compute} to throw {@code C}. This way the type system checks that no
 * unexpected exceptions are thrown by the {@link SkyFunction}.
 *
 * <p>We took this approach over using a generic exception class since Java disallows it because of
 * type erasure
 * (see http://docs.oracle.com/javase/tutorial/java/generics/restrictions.html#cannotCatch).
 *
 * <p> Note that there are restrictions on what Exception types are allowed to be wrapped in this
 * manner. See {@link SkyFunctionException#validateExceptionType}.
 *
 * <p>Failures are explicitly either transient or persistent. The transience of the failure from
 * {@link SkyFunction#compute} should be influenced only by the computations done, and not by the
 * transience of the failures from computations requested via
 * {@link SkyFunction.Environment#getValueOrThrow}.
 */
public abstract class SkyFunctionException extends Exception {

  /** The transience of the error. */
  public enum Transience {
    /**
     * An error that may or may not occur again if the computation were re-run. If a computation
     * results in a transient error and is needed on a subsequent MemoizingEvaluator#evaluate call,
     * it will be re-executed.
     */
    TRANSIENT,

    /**
     * An error that is completely deterministic and persistent in terms of the computation's
     * inputs. Persistent errors may be cached.
     */
    PERSISTENT;
  }

  private final Transience transience;
  @Nullable
  private final SkyKey rootCause;

  public SkyFunctionException(Exception cause, Transience transience) {
    this(cause, transience, null);
  }

  /** Used to rethrow a child error that the parent cannot handle. */
  public SkyFunctionException(Exception cause, SkyKey childKey) {
    this(cause, Transience.PERSISTENT, childKey);
  }

  private SkyFunctionException(Exception cause, Transience transience, SkyKey rootCause) {
    super(Preconditions.checkNotNull(cause));
    SkyFunctionException.validateExceptionType(cause.getClass());
    this.transience = transience;
    this.rootCause = rootCause;
  }

  @Nullable
  final SkyKey getRootCauseSkyKey() {
    return rootCause;
  }

  final boolean isTransient() {
    return transience == Transience.TRANSIENT;
  }

  /**
   * Catastrophic failures halt the build even when in keepGoing mode.
   */
  public boolean isCatastrophic() {
    return false;
  }

  @Override
  public synchronized Exception getCause() {
    return (Exception) super.getCause();
  }

  static <E extends Exception> void validateExceptionType(Class<E> exceptionClass) {
    if (exceptionClass.equals(ValueOrExceptionUtils.BottomException.class)) {
      return;
    }

    if (exceptionClass.isAssignableFrom(RuntimeException.class)) {
      throw new IllegalStateException(exceptionClass.getSimpleName() + " is a supertype of "
          + "RuntimeException. Don't do this since then you would potentially swallow all "
          + "RuntimeExceptions, even those from Skyframe");
    }
    if (RuntimeException.class.isAssignableFrom(exceptionClass)) {
      throw new IllegalStateException(exceptionClass.getSimpleName() + " is a subtype of "
          + "RuntimeException. You should rewrite your code to use checked exceptions.");
    }
    if (InterruptedException.class.isAssignableFrom(exceptionClass)) {
      throw new IllegalStateException(exceptionClass.getSimpleName() + " is a subtype of "
          + "InterruptedException. Don't do this; Skyframe handles interrupts separately from the "
          + "general SkyFunctionException mechanism.");
    }
  }

  /** A {@link SkyFunctionException} with a definite root cause. */
  static class ReifiedSkyFunctionException extends SkyFunctionException {
    private final boolean isCatastrophic;

    ReifiedSkyFunctionException(SkyFunctionException e, SkyKey key) {
      super(e.getCause(), e.transience, Preconditions.checkNotNull(e.getRootCauseSkyKey() == null
          ? key : e.getRootCauseSkyKey()));
      this.isCatastrophic = e.isCatastrophic();
    }

    @Override
    public boolean isCatastrophic() {
      return isCatastrophic;
    }
  }
}
