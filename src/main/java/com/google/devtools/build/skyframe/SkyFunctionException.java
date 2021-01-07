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

import com.google.common.base.Preconditions;

/**
 * Base class of exceptions thrown by {@link SkyFunction#compute} on failure.
 *
 * <p>SkyFunctions should declare a subclass {@code C} of {@link SkyFunctionException} whose
 * constructors forward fine-grained exception types (e.g. {@code IOException}) to {@link
 * SkyFunctionException}'s constructor, and they should also declare {@link SkyFunction#compute} to
 * throw {@code C}. This way the type system checks that no unexpected exceptions are thrown by the
 * {@link SkyFunction}.
 *
 * <p>We took this approach over using a generic exception class since Java disallows it because of
 * type erasure (see
 * http://docs.oracle.com/javase/tutorial/java/generics/restrictions.html#cannotCatch).
 *
 * <p>Note that there are restrictions on what Exception types are allowed to be wrapped in this
 * manner. See {@link SkyFunctionException#validateExceptionType}.
 *
 * <p>Failures are explicitly marked transient or persistent. Transient errors indicate that the
 * node may yield a successful result on a retry, while persistent errors are guaranteed to remain
 * if none of the inputs to the node change. An error should be marked persistent if either (1) it
 * is propagating an error from a Skyframe dependency (observed via {@link
 * SkyFunction.Environment#getValueOrThrow}, or (2) it is the result of computation done solely on
 * the inputs received from its Skyframe dependencies. Any other errors should be marked transient.
 * For example, an I/O exception should trigger a transient error in the node that directly
 * performed the I/O, and persistent errors in its callers.
 */
public abstract class SkyFunctionException extends Exception {

  /** The transience of the error. */
  public enum Transience {
    /**
     * An error that may or may not occur again if the node were reevaluated, even when Skyframe
     * dependencies have not changed. If a node results in a transient error and is needed on a
     * subsequent MemoizingEvaluator#evaluate call, it will be reevaluated.
     */
    TRANSIENT,

    /**
     * An error that is completely deterministic in terms of the node's Skyframe dependencies.
     * Persistent errors may be cached.
     */
    PERSISTENT;
  }

  private final Transience transience;

  public SkyFunctionException(Exception cause, Transience transience) {
    super(Preconditions.checkNotNull(cause));
    SkyFunctionException.validateExceptionType(cause.getClass());
    this.transience = transience;
  }

  public final boolean isTransient() {
    return transience == Transience.TRANSIENT;
  }

  /** Catastrophic failures halt the build even when in keepGoing mode. */
  public boolean isCatastrophic() {
    return false;
  }

  @Override
  public synchronized Exception getCause() {
    return (Exception) super.getCause();
  }

  static <E extends Exception> void validateExceptionType(Class<E> exceptionClass) {
    if (exceptionClass.isAssignableFrom(RuntimeException.class)) {
      throw new IllegalStateException(
          exceptionClass.getSimpleName()
              + " is a supertype of RuntimeException. Don't do this since then you would"
              + " potentially swallow all RuntimeExceptions, even those from Skyframe");
    }
    if (RuntimeException.class.isAssignableFrom(exceptionClass)) {
      throw new IllegalStateException(
          exceptionClass.getSimpleName()
              + " is a subtype of RuntimeException. You should rewrite your code to use checked"
              + " exceptions.");
    }
    if (InterruptedException.class.isAssignableFrom(exceptionClass)) {
      throw new IllegalStateException(
          exceptionClass.getSimpleName()
              + " is a subtype of InterruptedException. Don't do this; Skyframe handles interrupts"
              + " separately from the general SkyFunctionException mechanism.");
    }
  }

  /** A {@link SkyFunctionException} with a definite root cause. */
  public static class ReifiedSkyFunctionException extends SkyFunctionException {
    private final boolean isCatastrophic;
    private final SkyFunctionException originalException;

    public ReifiedSkyFunctionException(SkyFunctionException e) {
      this(e, e.transience, e.isCatastrophic());
    }

    protected ReifiedSkyFunctionException(
        SkyFunctionException e,
        Transience transience,
        boolean isCatastrophic) {
      super(e.getCause(), transience);
      this.isCatastrophic = isCatastrophic;
      this.originalException = e;
    }

    @Override
    public boolean isCatastrophic() {
      return isCatastrophic;
    }

    public SkyFunctionException getOriginalException() {
      return originalException;
    }
  }
}
