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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.state.Driver;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.ValueOrException2Producer;
import com.google.devtools.build.skyframe.state.ValueOrExceptionProducer;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class StateMachineTest {
  private final ProcessableGraph graph = new InMemoryGraphImpl();
  private final GraphTester tester = new GraphTester();

  private final StoredEventHandler reportedEvents = new StoredEventHandler();
  private final DirtyTrackingProgressReceiver revalidationReceiver =
      new DirtyTrackingProgressReceiver(null);

  private static final Version VERSION = IntVersion.of(0);

  // TODO(shahan): consider factoring this boilerplate out to a common location.
  private <T extends SkyValue> EvaluationResult<T> eval(SkyKey root, boolean keepGoing)
      throws InterruptedException {
    return new ParallelEvaluator(
            graph,
            VERSION,
            Version.minimal(),
            tester.getSkyFunctionMap(),
            reportedEvents,
            new NestedSetVisitor.VisitedState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            keepGoing,
            revalidationReceiver,
            GraphInconsistencyReceiver.THROWING,
            () -> AbstractQueueVisitor.createExecutorService(200, "test-pool"),
            new SimpleCycleDetector(),
            /* cpuHeavySkyKeysThreadPoolSize= */ 0,
            /* executionJobsThreadPoolSize= */ 0,
            UnnecessaryTemporaryStateDropperReceiver.NULL)
        .eval(ImmutableList.of(root));
  }

  private static final SkyKey KEY_A1 = GraphTester.toSkyKey("A1");
  private static final SkyValue VALUE_A1 = new StringValue("A1");
  private static final SkyKey KEY_A2 = GraphTester.toSkyKey("A2");
  private static final SkyValue VALUE_A2 = new StringValue("A2");
  private static final SkyKey KEY_A3 = GraphTester.toSkyKey("A3");
  private static final SkyValue VALUE_A3 = new StringValue("A3");
  private static final SkyKey KEY_B1 = GraphTester.toSkyKey("B1");
  private static final SkyValue VALUE_B1 = new StringValue("B1");
  private static final SkyKey KEY_B2 = GraphTester.toSkyKey("B2");
  private static final SkyValue VALUE_B2 = new StringValue("B2");
  private static final SkyKey KEY_B3 = GraphTester.toSkyKey("B3");
  private static final SkyValue VALUE_B3 = new StringValue("B3");

  private static final SkyKey ROOT_KEY = GraphTester.toSkyKey("root");
  private static final SkyValue DONE_VALUE = new StringValue("DONE");
  private static final StringValue SUCCESS_VALUE = new StringValue("SUCCESS");

  @Before
  public void predefineCommonEntries() {
    tester.getOrCreate(KEY_A1).setConstantValue(VALUE_A1);
    tester.getOrCreate(KEY_A2).setConstantValue(VALUE_A2);
    tester.getOrCreate(KEY_A3).setConstantValue(VALUE_A3);
    tester.getOrCreate(KEY_B1).setConstantValue(VALUE_B1);
    tester.getOrCreate(KEY_B2).setConstantValue(VALUE_B2);
    tester.getOrCreate(KEY_B3).setConstantValue(VALUE_B3);
  }

  private static class StateMachineWrapper implements SkyKeyComputeState {
    private final Driver driver;

    private StateMachineWrapper(StateMachine machine) {
      this.driver = new Driver(machine);
    }

    private boolean drive(Environment env, ExtendedEventHandler listener)
        throws InterruptedException {
      return driver.drive(env, listener);
    }
  }

  /**
   * Defines a {@link SkyFunction} that executes the gives state machine.
   *
   * <p>The function always has key {@link ROOT_KEY} and value {@link DONE_VALUE}. State machine
   * internal can be observed with consumers.
   *
   * @return an counter that stores the restart count.
   */
  private AtomicInteger defineRootMachine(Supplier<StateMachine> rootMachineSupplier) {
    var restartCount = new AtomicInteger();
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              if (!env.getState(() -> new StateMachineWrapper(rootMachineSupplier.get()))
                  .drive(env, env.getListener())) {
                restartCount.getAndIncrement();
                return null;
              }
              return DONE_VALUE;
            });
    return restartCount;
  }

  private int evalMachine(Supplier<StateMachine> rootMachineSupplier) throws InterruptedException {
    var restartCount = defineRootMachine(rootMachineSupplier);
    assertThat(eval(ROOT_KEY, /* keepGoing= */ false).get(ROOT_KEY)).isEqualTo(DONE_VALUE);
    return restartCount.get();
  }

  /**
   * A simple machine having two states, fetching one value from each.
   *
   * <p>This machine causes two restarts, one for each of the lookups from the two states.
   */
  private static class TwoStepMachine implements StateMachine {
    private final Consumer<SkyValue> sink1;
    private final Consumer<SkyValue> sink2;

    private TwoStepMachine(Consumer<SkyValue> sink1, Consumer<SkyValue> sink2) {
      this.sink1 = sink1;
      this.sink2 = sink2;
    }

    @Override
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_A1, sink1);
      return this::step2;
    }

    @Nullable
    public StateMachine step2(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_A2, sink2);
      return null;
    }
  }

  @Test
  public void smoke() throws InterruptedException {
    var v1Sink = new SkyValueSink();
    var v2Sink = new SkyValueSink();
    assertThat(evalMachine(() -> new TwoStepMachine(v1Sink, v2Sink))).isEqualTo(2);
    assertThat(v1Sink.get()).isEqualTo(VALUE_A1);
    assertThat(v2Sink.get()).isEqualTo(VALUE_A2);
  }

  /** Example modeled after the one described in the documentation of {@link StateMachine}. */
  private static class ExampleWithSubmachines implements StateMachine, SkyKeyComputeState {
    private final Consumer<SkyValue> sinkA1;
    private final Consumer<SkyValue> sinkA2;
    private final Consumer<SkyValue> sinkA3;
    private final Consumer<SkyValue> sinkB1;
    private final Consumer<SkyValue> sinkB2;
    private final Consumer<SkyValue> sinkB3;

    private ExampleWithSubmachines(
        Consumer<SkyValue> sinkA1,
        Consumer<SkyValue> sinkA2,
        Consumer<SkyValue> sinkA3,
        Consumer<SkyValue> sinkB1,
        Consumer<SkyValue> sinkB2,
        Consumer<SkyValue> sinkB3) {
      this.sinkA1 = sinkA1;
      this.sinkA2 = sinkA2;
      this.sinkA3 = sinkA3;
      this.sinkB1 = sinkB1;
      this.sinkB2 = sinkB2;
      this.sinkB3 = sinkB3;
    }

    @Override
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      // Starts submachines in parallel.
      tasks.enqueue(this::stepA1);
      tasks.enqueue(this::stepB1);
      return null;
    }

    private StateMachine stepA1(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_A1, sinkA1);
      return this::stepA2;
    }

    private StateMachine stepA2(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_A2, sinkA2);
      return this::stepA3;
    }

    @Nullable
    private StateMachine stepA3(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_A3, sinkA3);
      return null;
    }

    private StateMachine stepB1(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_B1, sinkB1);
      return this::stepB2;
    }

    private StateMachine stepB2(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_B2, sinkB2);
      return this::stepB3;
    }

    @Nullable
    private StateMachine stepB3(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(KEY_B3, sinkB3);
      return null;
    }
  }

  @Test
  public void parallelSubmachines_runInParallel() throws InterruptedException {
    var a1Sink = new SkyValueSink();
    var a2Sink = new SkyValueSink();
    var a3Sink = new SkyValueSink();
    var b1Sink = new SkyValueSink();
    var b2Sink = new SkyValueSink();
    var b3Sink = new SkyValueSink();

    assertThat(
            evalMachine(
                () -> new ExampleWithSubmachines(a1Sink, a2Sink, a3Sink, b1Sink, b2Sink, b3Sink)))
        .isEqualTo(3);

    assertThat(a1Sink.get()).isEqualTo(VALUE_A1);
    assertThat(a2Sink.get()).isEqualTo(VALUE_A2);
    assertThat(a3Sink.get()).isEqualTo(VALUE_A3);
    assertThat(b1Sink.get()).isEqualTo(VALUE_B1);
    assertThat(b2Sink.get()).isEqualTo(VALUE_B2);
    assertThat(b3Sink.get()).isEqualTo(VALUE_B3);
  }

  @Test
  public void parallelSubmachines_shorteningBothPathsReducesRestarts() throws InterruptedException {
    var a1Sink = new SkyValueSink();
    var a2Sink = new SkyValueSink();
    var a3Sink = new SkyValueSink();
    var b1Sink = new SkyValueSink();
    var b2Sink = new SkyValueSink();
    var b3Sink = new SkyValueSink();

    // Shortens both paths by 1, but at different execution steps.
    assertThat(eval(KEY_A1, /* keepGoing= */ false).get(KEY_A1)).isEqualTo(VALUE_A1);
    assertThat(eval(KEY_B3, /* keepGoing= */ false).get(KEY_B3)).isEqualTo(VALUE_B3);

    assertThat(
            evalMachine(
                () -> new ExampleWithSubmachines(a1Sink, a2Sink, a3Sink, b1Sink, b2Sink, b3Sink)))
        .isEqualTo(2);

    assertThat(a1Sink.get()).isEqualTo(VALUE_A1);
    assertThat(a2Sink.get()).isEqualTo(VALUE_A2);
    assertThat(a3Sink.get()).isEqualTo(VALUE_A3);
    assertThat(b1Sink.get()).isEqualTo(VALUE_B1);
    assertThat(b2Sink.get()).isEqualTo(VALUE_B2);
    assertThat(b3Sink.get()).isEqualTo(VALUE_B3);
  }

  @Test
  public void unhandledException(@TestParameter boolean keepGoing) throws InterruptedException {
    var a1Sink = new SkyValueSink();
    var a2Sink = new SkyValueSink();
    var a3Sink = new SkyValueSink();
    var b1Sink = new SkyValueSink();
    var b2Sink = new SkyValueSink();
    var b3Sink = new SkyValueSink();

    tester.getOrCreate(KEY_A1).unsetConstantValue().setHasError(true);

    AtomicInteger instantiationCount = new AtomicInteger();
    var restartCount =
        defineRootMachine(
            () -> {
              instantiationCount.getAndIncrement();
              return new ExampleWithSubmachines(a1Sink, a2Sink, a3Sink, b1Sink, b2Sink, b3Sink);
            });
    assertThat(eval(ROOT_KEY, keepGoing).getError(ROOT_KEY)).isNotNull();

    assertThat(restartCount.get()).isEqualTo(2);
    assertThat(a1Sink.get()).isNull();
    if (keepGoing) {
      // On restart, all values are processed before failing, so B1 is observed after restarting and
      // after A1's unhandled error.
      assertThat(b1Sink.get()).isEqualTo(VALUE_B1);
    }
    // In noKeepGoing, error bubbling resets the state cache and B1 is sometimes observed on the
    // first pass by a re-instantiated state machine. However, B1 can be slow and there is no
    // guarantee that it is available.

    assertThat(b2Sink.get()).isNull();

    if (keepGoing) {
      assertThat(instantiationCount.get()).isEqualTo(1);
    } else {
      // The state cache is dropped in noKeepGoing during error bubbling, resulting in a new
      // instantiation of the state machine.
      assertThat(instantiationCount.get()).isEqualTo(2);
    }
  }

  @Test
  public void handledException(@TestParameter boolean keepGoing) throws InterruptedException {
    tester.getOrCreate(KEY_A1).unsetConstantValue().setHasError(true);

    var a1Sink = new SkyValueSink();
    var errorSink = new AtomicReference<SomeErrorException>();
    var restartCount =
        defineRootMachine(
            () ->
                (tasks, listener) -> {
                  // Fully swallows the error.
                  tasks.lookUp(
                      KEY_A1,
                      SomeErrorException.class,
                      (v, e) -> {
                        if (v != null) {
                          a1Sink.accept(v);
                          return;
                        }
                        errorSink.set(e);
                      });
                  return null;
                });
    var result = eval(ROOT_KEY, keepGoing);
    if (keepGoing) {
      // In keepGoing mode, the swallowed error vanishes.
      assertThat(result.get(ROOT_KEY)).isEqualTo(DONE_VALUE);
      assertThat(result.hasError()).isFalse();
    } else {
      // In nokeepGoing mode, the error is procesed in error bubbling, but the function does not
      // complete and the error is still propagated to the top level.
      assertThat(result.get(ROOT_KEY)).isNull();
      assertThatEvaluationResult(result).hasSingletonErrorThat(KEY_A1);
    }
    assertThat(restartCount.get()).isEqualTo(1);
    assertThat(a1Sink.get()).isNull();
    assertThat(errorSink.get()).isNotNull();
  }

  private static class StringOrExceptionProducer
      extends ValueOrExceptionProducer.WithDriver<StringValue, SomeErrorException>
      implements SkyKeyComputeState {
    @Override
    @Nullable
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(
          KEY_A1,
          SomeErrorException.class,
          (v, e) -> {
            if (v != null) {
              setValue((StringValue) v);
              return;
            }
            setException(e);
          });
      return null;
    }
  }

  @Test
  public void valueOrExceptionProducer_propagatesValues() throws InterruptedException {
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrExceptionProducer::new);
              if (!producer.drive(env, env.getListener())) {
                return null;
              }
              assertThat(producer.hasResult()).isTrue();
              assertThat(producer.hasValue()).isTrue();
              assertThat(producer.getValue()).isEqualTo(VALUE_A1);
              assertThat(producer.hasException()).isFalse();
              try {
                assertThat(producer.getValueOrThrow()).isEqualTo(VALUE_A1);
              } catch (SomeErrorException e) {
                fail("Unexpecteded exception: " + e);
              }
              return DONE_VALUE;
            });
    assertThat(eval(ROOT_KEY, /* keepGoing= */ false).get(ROOT_KEY)).isEqualTo(DONE_VALUE);
  }

  @Test
  public void valueOrExceptionProducer_propagatesExceptions(@TestParameter boolean keepGoing)
      throws InterruptedException {
    tester.getOrCreate(KEY_A1).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrExceptionProducer::new);
              if (!producer.drive(env, env.getListener())) {
                return null;
              }
              assertThat(producer.hasResult()).isTrue();
              assertThat(producer.hasValue()).isFalse();
              assertThat(producer.hasException()).isTrue();
              assertThrows(SomeErrorException.class, producer::getValueOrThrow);
              assertThat(producer.getException()).isNotNull();
              return DONE_VALUE;
            });
    var result = eval(ROOT_KEY, keepGoing);
    if (keepGoing) {
      assertThat(result.get(ROOT_KEY)).isEqualTo(DONE_VALUE);
      assertThat(result.hasError()).isFalse();
    } else {
      assertThat(result.get(ROOT_KEY)).isNull();
      assertThatEvaluationResult(result).hasSingletonErrorThat(KEY_A1);
    }
  }

  private static class StringOrException2Producer
      extends ValueOrException2Producer.WithDriver<
          StringValue, SomeErrorException, ExecutionException>
      implements SkyKeyComputeState {
    @Override
    public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
      tasks.lookUp(
          KEY_A1,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException1(e);
            }
          });
      tasks.lookUp(
          KEY_B1,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException2(new ExecutionException(e));
            }
          });
      return (t, l) -> {
        if (!hasResult()) {
          setValue(SUCCESS_VALUE);
        }
        return null;
      };
    }
  }

  @Test
  public void valueOrException2Producer_propagatesValues() throws InterruptedException {
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrException2Producer::new);
              if (!producer.drive(env, env.getListener())) {
                return null;
              }
              assertThat(producer.hasResult()).isTrue();
              assertThat(producer.hasValue()).isTrue();
              assertThat(producer.getValue()).isEqualTo(SUCCESS_VALUE);
              assertThat(producer.hasException1()).isFalse();
              assertThat(producer.hasException2()).isFalse();
              try {
                assertThat(producer.getValueOrThrow()).isEqualTo(SUCCESS_VALUE);
              } catch (SomeErrorException | ExecutionException e) {
                fail("Unexpecteded exception: " + e);
              }
              return DONE_VALUE;
            });
    assertThat(eval(ROOT_KEY, /* keepGoing= */ false).get(ROOT_KEY)).isEqualTo(DONE_VALUE);
  }

  @Test
  public void valueOrException2Producer_propagatesExceptions(
      @TestParameter boolean trueForException1, @TestParameter boolean keepGoing)
      throws InterruptedException {
    SkyKey errorKey = trueForException1 ? KEY_A1 : KEY_B1;
    tester.getOrCreate(errorKey).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrException2Producer::new);
              if (!producer.drive(env, env.getListener())) {
                return null;
              }
              assertThat(producer.hasResult()).isTrue();
              assertThat(producer.hasValue()).isFalse();
              if (trueForException1) {
                assertThat(producer.hasException1()).isTrue();
                assertThrows(SomeErrorException.class, producer::getValueOrThrow);
                assertThat(producer.getException1()).isNotNull();
                assertThat(producer.hasException2()).isFalse();
              } else {
                assertThat(producer.hasException1()).isFalse();
                assertThat(producer.hasException2()).isTrue();
                assertThrows(ExecutionException.class, producer::getValueOrThrow);
                assertThat(producer.getException2()).isNotNull();
              }
              return DONE_VALUE;
            });
    var result = eval(ROOT_KEY, keepGoing);
    if (keepGoing) {
      assertThat(result.get(ROOT_KEY)).isEqualTo(DONE_VALUE);
      assertThat(result.hasError()).isFalse();
    } else {
      assertThat(result.get(ROOT_KEY)).isNull();
      assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
    }
  }

  /**
   * Sink for {@link SkyValue}s.
   *
   * <p>Verifies that the value is set no more than once.
   */
  private static class SkyValueSink implements Consumer<SkyValue> {
    private SkyValue value;

    @Override
    public void accept(SkyValue value) {
      assertThat(this.value).isNull();
      this.value = value;
    }

    @Nullable
    private SkyValue get() {
      return value;
    }
  }
}
