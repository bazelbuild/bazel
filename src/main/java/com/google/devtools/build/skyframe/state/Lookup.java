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

import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.function.Consumer;

/** Captures information about a lookup requested by a state machine. */
abstract class Lookup implements SkyframeLookupResult.QueryDepCallback {
  private final TaskTreeNode parent;
  final SkyKey key;

  private Lookup(TaskTreeNode parent, SkyKey key) {
    this.parent = parent;
    this.key = key;
  }

  final SkyKey key() {
    return key;
  }

  /**
   * Performs a lookup directly against the environment.
   *
   * <p>This is more efficient than {@link LookupEnvironment#getValuesAndExceptions} when there is
   * only one key at a time.
   *
   * @return true if a value was available or an exception was handled. Note: this is false for
   *     unhandled exceptions.
   */
  abstract boolean doLookup(LookupEnvironment env) throws InterruptedException;

  @Override
  public final void acceptValue(SkyKey unusedKey, SkyValue value) {
    acceptValue(value);
    parent.signalChildDoneAndEnqueueIfReady();
  }

  abstract void acceptValue(SkyValue value);

  @Override
  public final boolean tryHandleException(SkyKey unusedKey, Exception exception) {
    boolean handled = tryHandleException(exception);
    if (handled) {
      parent.signalChildDoneAndEnqueueIfReady();
    }
    return handled;
  }

  abstract boolean tryHandleException(Exception exception);

  static final class ConsumerLookup extends Lookup {
    private final Consumer<SkyValue> sink;

    ConsumerLookup(TaskTreeNode parent, SkyKey key, Consumer<SkyValue> sink) {
      super(parent, key);
      this.sink = sink;
    }

    @Override
    boolean doLookup(LookupEnvironment env) throws InterruptedException {
      var value = env.getValue(key);
      if (value == null) {
        return false;
      }
      acceptValue(key, value);
      return true;
    }

    @Override
    void acceptValue(SkyValue value) {
      sink.accept(value);
    }

    @Override
    boolean tryHandleException(Exception unusedException) {
      return false;
    }
  }

  static final class ValueOrExceptionLookup<E extends Exception> extends Lookup {
    private final Class<E> exceptionClass;
    private final StateMachine.ValueOrExceptionSink<E> sink;

    ValueOrExceptionLookup(
        TaskTreeNode parent,
        SkyKey key,
        Class<E> exceptionClass,
        StateMachine.ValueOrExceptionSink<E> sink) {
      super(parent, key);
      this.exceptionClass = exceptionClass;
      this.sink = sink;
    }

    @Override
    boolean doLookup(LookupEnvironment env) throws InterruptedException {
      SkyValue value;
      try {
        if ((value = env.getValueOrThrow(key(), exceptionClass)) == null) {
          return false;
        }
        acceptValue(key, value);
      } catch (Exception e) {
        if (e instanceof InterruptedException) {
          throw (InterruptedException) e;
        }
        if (!tryHandleException(e)) {
          throw new IllegalArgumentException("Unexpected exception for " + key(), e);
        }
      }
      return true;
    }

    @Override
    void acceptValue(SkyValue value) {
      sink.acceptValueOrException(value, /* exception= */ null);
    }

    @Override
    boolean tryHandleException(Exception exception) {
      if (exceptionClass.isInstance(exception)) {
        sink.acceptValueOrException(/* value= */ null, exceptionClass.cast(exception));
        return true;
      }
      return false;
    }
  }

  static final class ValueOrException2Lookup<E1 extends Exception, E2 extends Exception>
      extends Lookup {
    private final Class<E1> exceptionClass1;
    private final Class<E2> exceptionClass2;
    private final StateMachine.ValueOrException2Sink<E1, E2> sink;

    ValueOrException2Lookup(
        TaskTreeNode parent,
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        StateMachine.ValueOrException2Sink<E1, E2> sink) {
      super(parent, key);
      this.exceptionClass1 = exceptionClass1;
      this.exceptionClass2 = exceptionClass2;
      this.sink = sink;
    }

    @Override
    boolean doLookup(LookupEnvironment env) throws InterruptedException {
      SkyValue value;
      try {
        if ((value = env.getValueOrThrow(key(), exceptionClass1, exceptionClass2)) == null) {
          return false;
        }
        acceptValue(key, value);
      } catch (Exception e) {
        if (e instanceof InterruptedException) {
          throw (InterruptedException) e;
        }
        if (!tryHandleException(e)) {
          throw new IllegalArgumentException("Unexpected exception for " + key(), e);
        }
      }
      return true;
    }

    @Override
    void acceptValue(SkyValue value) {
      sink.acceptValueOrException2(value, /* e1= */ null, /* e2= */ null);
    }

    @Override
    boolean tryHandleException(Exception exception) {
      if (exceptionClass1.isInstance(exception)) {
        sink.acceptValueOrException2(
            /* value= */ null, exceptionClass1.cast(exception), /* e2= */ null);
        return true;
      }
      if (exceptionClass2.isInstance(exception)) {
        sink.acceptValueOrException2(
            /* value= */ null, /* e1= */ null, exceptionClass2.cast(exception));
        return true;
      }
      return false;
    }
  }

  static final class ValueOrException3Lookup<
          E1 extends Exception, E2 extends Exception, E3 extends Exception>
      extends Lookup {
    private final Class<E1> exceptionClass1;
    private final Class<E2> exceptionClass2;
    private final Class<E3> exceptionClass3;
    private final StateMachine.ValueOrException3Sink<E1, E2, E3> sink;

    ValueOrException3Lookup(
        TaskTreeNode parent,
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        Class<E3> exceptionClass3,
        StateMachine.ValueOrException3Sink<E1, E2, E3> sink) {
      super(parent, key);
      this.exceptionClass1 = exceptionClass1;
      this.exceptionClass2 = exceptionClass2;
      this.exceptionClass3 = exceptionClass3;
      this.sink = sink;
    }

    @Override
    boolean doLookup(LookupEnvironment env) throws InterruptedException {
      SkyValue value;
      try {
        if ((value = env.getValueOrThrow(key(), exceptionClass1, exceptionClass2, exceptionClass3))
            == null) {
          return false;
        }
        acceptValue(key, value);
      } catch (Exception e) {
        if (e instanceof InterruptedException) {
          throw (InterruptedException) e;
        }
        if (!tryHandleException(e)) {
          throw new IllegalArgumentException("Unexpected exception for " + key(), e);
        }
      }
      return true;
    }

    @Override
    void acceptValue(SkyValue value) {
      sink.acceptValueOrException3(value, /* e1= */ null, /* e2= */ null, /* e3= */ null);
    }

    @Override
    boolean tryHandleException(Exception exception) {
      if (exceptionClass1.isInstance(exception)) {
        sink.acceptValueOrException3(
            /* value= */ null, exceptionClass1.cast(exception), /* e2= */ null, /* e3= */ null);
        return true;
      }
      if (exceptionClass2.isInstance(exception)) {
        sink.acceptValueOrException3(
            /* value= */ null, /* e1= */ null, exceptionClass2.cast(exception), /* e3= */ null);
        return true;
      }
      if (exceptionClass3.isInstance(exception)) {
        sink.acceptValueOrException3(
            /* value= */ null, /* e1= */ null, /* e2= */ null, exceptionClass3.cast(exception));
        return true;
      }
      return false;
    }
  }

  @Override
  public String toString() {
    return toStringHelper(this).add("parent", parent).add("key", key).toString();
  }
}
