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

import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A simple state machine with structured concurrency.
 *
 * <p>This is used to implement {@link SkyFunction}s with logical concurrency within {@link
 * SkyKeyComputeState}. All execution is singly-threaded. It can be used in places where further
 * stateful decomposition of a computation is desirable but more {@code Skyframe} entries would
 * create too much overhead. However, the key motivation is to facilitate logical concurrency.
 *
 * <p>For example, consider a {@link SkyFunction} that processes dependencies. Each dependency
 * requires a sequence of processing steps, some of which have {@code Skyframe} lookups. Let {@code
 * A = <A1, A2, A3>} and {@code B = <B1, B2, B3>} be sequences of dependent {@code SkyKey}s. {@code
 * A3} depends on {@code A2} depends on {@code A1} and similarly for {@code B3, B2, B1}. The
 * processing of {@code A} and {@code B} are independent and therefore logically concurrent.
 *
 * <p>One conventional approach is to implement the {@code SkyFunction} to make groups like {@code
 * (A1,B1), (A2,B2), (A3,B3)}. Grouping is more efficient than one-at-a-time lookups because in
 * builds where lookups can be performed remotely, such implementations can parallelize over groups
 * but process single queries sequentially to avoid wasted speculative work.
 *
 * <p>However, this manual batching may creates false dependencies that lead to unnecessary latency.
 * In the example, {@code A2} can't be evaluated until {@code B1} is available. If both {@code B1}
 * and {@code A2} are slow, the critical path is unnecessarily lengthened by forcing {@code A2} to
 * happen after {@code B1} instead of allowing {@code A2} to proceed once {@code A1} is available.
 * From the perspective of restarts, if both {@code B1} and {@code A2} require restarts, evaluation
 * requires 2 restarts. However, by running concurrently, the restart of {@code B1} and {@code A2}
 * can be grouped together, reducing it to 1 restart.
 *
 * <p>The {@link Driver} class and {@link Driver#drive} are used to run a state machine.
 *
 * <p>A guide is available at <a href="https://bazel.build/contribute/statemachine-guide"/>.
 */
@FunctionalInterface
public interface StateMachine {
  /** A sentinel value returned when a {@code StateMachine} is done. */
  public static final StateMachine DONE =
      t -> {
        throw new IllegalStateException("Sentinel DONE state should not be executed.");
      };

  /**
   * Step performs the next computation.
   *
   * <p>{@link Tasks#lookup} may be used to request {@link SkyKey}s. The next step will not be
   * executed until all requested {@link SkyValue}s are available and their associated callbacks
   * have been called. Similarly, {@link Tasks#enqueue} can be used to spawn a concurrent
   * subcomputation, which must also complete before the next computation step.
   *
   * <p>Note that recursive decomposition within subtasks is possible and can be used to capture
   * fine-grain dependency structures. This is required to correctly model the example in the class
   * description. {@code <A1, A2, A3>} and {@code <B1, B2, B3>} become concurrent, multi-step,
   * subtasks.
   *
   * @param tasks an interface for adding subtasks, which may be either {@link SkyKey} lookups or
   *     child state machines. The {@code tasks} handle is associated with this state machine and
   *     other state machines should not use it.
   * @return an instance indicating the next computation or {@link #DONE} on completion.
   */
  StateMachine step(Tasks tasks) throws InterruptedException;

  /**
   * Tasks allows registering logically parallel subtasks.
   *
   * <p>Completion of the current step waits until all subtasks are complete.
   */
  interface Tasks {
    /**
     * Enqueues a subtask for logically concurrent evaluation.
     *
     * <p>The next step will not be executed until the subtask completes. If more than one subtask
     * is enqueued, the next step waits on all subtasks.
     */
    void enqueue(StateMachine subtask);

    /**
     * A lookup that handles no exceptions.
     *
     * <p>This lookup is logically concurrent with other subtasks. The state machine {@link Driver}
     * may defer the callback until after a {@code Skyframe} restart if it is not immediately
     * available.
     *
     * <p>Unhandled exceptions eventually set a fail fast condition over the entire state machine
     * tree and no further processing occurs afterwards.
     *
     * <p>IMPLEMENTATION: if an unhandled exception occurs immediately (without a restart) on a
     * lookup, {@link Driver} observes unavailability and returns an incomplete status. The driver
     * cannot distinguish here between a result that is not yet computed and an unhandled exception.
     *
     * <p>After a restart, all previously requested values should be available, so observing
     * unavailability implies an unhandled exception, triggering fail-fast.
     */
    void lookUp(SkyKey key, Consumer<SkyValue> sink);

    /**
     * A lookup that handles exceptions of the specified type.
     *
     * <p>The callback could be deferred until the next {@code Skyframe} restart if the queried key
     * is not immediately available.
     */
    <E extends Exception> void lookUp(
        SkyKey key, Class<E> exceptionClass, ValueOrExceptionSink<E> sink);

    /** A lookup that handles exceptions of the specified 2 types. */
    <E1 extends Exception, E2 extends Exception> void lookUp(
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        ValueOrException2Sink<E1, E2> sink);

    /** A lookup that handles exceptions of the specified 3 types. */
    <E1 extends Exception, E2 extends Exception, E3 extends Exception> void lookUp(
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        Class<E3> exceptionClass3,
        ValueOrException3Sink<E1, E2, E3> sink);
  }

  /**
   * Receives the result of a lookup.
   *
   * <p>Exactly one of {@code value} or {@code exception} will be non-null.
   */
  @FunctionalInterface
  interface ValueOrExceptionSink<E extends Exception> {
    void acceptValueOrException(@Nullable SkyValue value, @Nullable E exception);
  }

  /**
   * Receives the result of a lookup.
   *
   * <p>Exactly one of {@code value}, {@code e1} or {@code e2} will be non-null.
   */
  @FunctionalInterface
  interface ValueOrException2Sink<E1 extends Exception, E2 extends Exception> {
    void acceptValueOrException2(@Nullable SkyValue value, @Nullable E1 e1, @Nullable E2 e2);
  }

  /**
   * Receives the result of a lookup.
   *
   * <p>Exactly one of {@code value}, {@code e1}, {@code e2} or {@code e3} will be non-null.
   */
  @FunctionalInterface
  interface ValueOrException3Sink<
      E1 extends Exception, E2 extends Exception, E3 extends Exception> {
    void acceptValueOrException3(
        @Nullable SkyValue value, @Nullable E1 e1, @Nullable E2 e2, @Nullable E3 e3);
  }
}
