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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import javax.annotation.Nullable;

/**
 * A state machine that outputs a value or one of two possible exceptions.
 *
 * <p>Subclasses should call {@link #setValue}, {@link #setException1} or {@link #setException2} to
 * emit results. An API is provided for clients to safely retrieve the result when complete.
 */
public abstract class ValueOrException2Producer<V, E1 extends Exception, E2 extends Exception>
    implements StateMachine {
  private V value;
  private E1 exception1;
  private E2 exception2;

  /** A variant for top-level state machines that require a driver. */
  public abstract static class WithDriver<V, E1 extends Exception, E2 extends Exception>
      extends ValueOrException2Producer<V, E1, E2> {
    private final Driver driver = new Driver(this);

    /* Delegates to {@link Driver#drive}. */
    public final boolean drive(Environment env, ExtendedEventHandler listener)
        throws InterruptedException {
      return driver.drive(env, listener);
    }
  }

  protected final void setValue(V value) {
    checkState(!hasResult(), "%s value=%s", this, value);
    this.value = value;
  }

  protected final void setException1(E1 exception) {
    checkState(!hasResult(), "%s exception=%s", this, exception);
    this.exception1 = exception;
  }

  protected final void setException2(E2 exception) {
    checkState(!hasResult(), "%s exception=%s", this, exception);
    this.exception2 = exception;
  }

  @Nullable
  public final V getValueOrThrow() throws E1, E2 {
    if (value != null) {
      return value;
    }
    if (exception1 != null) {
      throw exception1;
    }
    if (exception2 != null) {
      throw exception2;
    }
    return null;
  }

  public final boolean hasResult() {
    return value != null || exception1 != null || exception2 != null;
  }

  public final boolean hasValue() {
    return value != null;
  }

  @Nullable
  public final V getValue() {
    return value;
  }

  public final boolean hasException1() {
    return exception1 != null;
  }

  @Nullable
  public final E1 getException1() {
    return exception1;
  }

  public final boolean hasException2() {
    return exception2 != null;
  }

  @Nullable
  public final E2 getException2() {
    return exception2;
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("value", value)
        .add("exception1", exception1)
        .add("exception2", exception2)
        .toString();
  }
}
