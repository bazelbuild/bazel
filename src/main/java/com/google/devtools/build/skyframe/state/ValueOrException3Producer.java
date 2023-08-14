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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import javax.annotation.Nullable;

/**
 * A state machine that outputs a value or one of three possible exceptions.
 *
 * <p>This class serves as a bridge between a {@link StateMachine} and a {@link SkyFunction}.
 *
 * <p>Subclasses should call {@link #setValue}, {@link #setException1}, {@link #setException2} or
 * {@link #setException3} to emit results.
 */
public abstract class ValueOrException3Producer<
        V, E1 extends Exception, E2 extends Exception, E3 extends Exception>
    implements StateMachine {
  private final Driver driver = new Driver(this);

  private V value;
  private E1 exception1;
  private E2 exception2;
  private E3 exception3;

  /**
   * Tries to produce the result of the underlying state machine.
   *
   * <p>See comment of {@link ValueOrExceptionProducer#tryProduceValue}. If multiple exceptions are
   * set, they are prioritized by number.
   */
  @Nullable
  public final V tryProduceValue(LookupEnvironment env) throws InterruptedException, E1, E2, E3 {
    boolean done = driver.drive(env);
    if (exception1 != null) {
      throw exception1;
    }
    if (exception2 != null) {
      throw exception2;
    }
    if (exception3 != null) {
      throw exception3;
    }
    if (done) {
      return checkNotNull(value);
    }
    return null;
  }

  protected final void setValue(V value) {
    this.value = value;
  }

  protected final void setException1(E1 exception) {
    this.exception1 = exception;
  }

  @Nullable
  protected final E1 getException1() {
    return exception1;
  }

  protected final void setException2(E2 exception) {
    this.exception2 = exception;
  }

  @Nullable
  protected final E2 getException2() {
    return exception2;
  }

  protected final void setException3(E3 exception) {
    this.exception3 = exception;
  }

  @Nullable
  protected final E3 getException3() {
    return exception3;
  }
}
