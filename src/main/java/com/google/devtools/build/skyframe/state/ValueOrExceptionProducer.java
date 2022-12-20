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
package com.google.devtools.build.skyframe.state;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunction.Environment;

/**
 * A state machine that outputs a value or exception.
 *
 * <p>Subclasses should call {@link #setValue} or {@link #setException} to emit results. An API is
 * provided for clients to safely retrieve the result when complete.
 *
 * <p>The parameter {@code V} must not be an exception type.
 */
public abstract class ValueOrExceptionProducer<V, E extends Exception> implements StateMachine {
  /** Will be of type {@code V} or {@code E}. */
  private Object result;

  /** A variant for top-level state machines that require a driver. */
  public abstract static class WithDriver<V, E extends Exception>
      extends ValueOrExceptionProducer<V, E> {
    private final Driver driver = new Driver(this);

    /* Delegates to {@link Driver#drive}. */
    public final boolean drive(Environment env, ExtendedEventHandler listener)
        throws InterruptedException {
      return driver.drive(env, listener);
    }
  }

  protected final void setValue(V value) {
    checkState(result == null);
    this.result = value;
  }

  protected final void setException(E exception) {
    checkState(result == null);
    this.result = exception;
  }

  public final boolean hasResult() {
    return result != null;
  }

  /** The caller must guarantee that there is a result before calling this. */
  @SuppressWarnings("unchecked")
  public final V getValueOrThrow() throws E {
    if (hasValue()) {
      return (V) result;
    }
    throw (E) result;
  }

  public final boolean hasValue() {
    return !hasException() && result != null;
  }

  /** The caller should guarantee that there is a value before calling this. */
  @SuppressWarnings("unchecked")
  public final V getValue() {
    return (V) result;
  }

  public final boolean hasException() {
    return result instanceof Exception;
  }

  /** The caller should guarantee that there is an exception before calling this. */
  @SuppressWarnings("unchecked")
  public final E getException() {
    return (E) result;
  }
}
