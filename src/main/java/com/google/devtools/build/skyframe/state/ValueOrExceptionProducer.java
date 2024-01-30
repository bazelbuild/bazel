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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import javax.annotation.Nullable;

/**
 * A state machine that outputs a value or exception.
 *
 * <p>This class serves as a bridge between a {@link StateMachine} and a {@link SkyFunction}.
 *
 * <p>Subclasses should call {@link #setValue} or {@link #setException} to emit results.
 *
 * <p>The parameter {@code V} must not be an exception type.
 */
public abstract class ValueOrExceptionProducer<V, E extends Exception> implements StateMachine {
  private final Driver driver = new Driver(this);

  /** Will be of type {@code V} or {@code E}. */
  private Object result;

  /**
   * Tries to produce the result of the underlying state machine.
   *
   * <p>Note that during error bubbling, the machine may discover and process an input error, even
   * with missing inputs, meaning that exceptions may be thrown before the machine considers itself
   * complete.
   *
   * <p>If both an error and value are set, the exception will take priority.
   *
   * @return null if the underlying state machine did not complete (due to missing inputs).
   */
  @Nullable
  @SuppressWarnings("unchecked")
  public final V tryProduceValue(LookupEnvironment env) throws InterruptedException, E {
    boolean done = driver.drive(env);
    if (result instanceof Exception) {
      throw (E) result;
    }
    if (done) {
      return checkNotNull((V) result);
    }
    return null;
  }

  protected final void setValue(V value) {
    this.result = value;
  }

  protected final void setException(E exception) {
    this.result = exception;
  }

  @Nullable
  @SuppressWarnings("unchecked")
  protected final E getException() {
    if (result instanceof Exception) {
      return (E) result;
    }
    return null;
  }
}
