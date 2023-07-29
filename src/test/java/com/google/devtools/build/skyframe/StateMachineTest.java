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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.state.Driver;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachineEvaluatorForTesting;
import com.google.devtools.build.skyframe.state.ValueOrException2Producer;
import com.google.devtools.build.skyframe.state.ValueOrException3Producer;
import com.google.devtools.build.skyframe.state.ValueOrExceptionProducer;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.concurrent.atomic.AtomicBoolean;
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
  private static final int TEST_PARALLELISM = 5;

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
            new EmittedEventState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            keepGoing,
            revalidationReceiver,
            GraphInconsistencyReceiver.THROWING,
            AbstractQueueVisitor.create(
                "test-pool", TEST_PARALLELISM, ParallelEvaluatorErrorClassifier.instance()),
            new SimpleCycleDetector(),
            /* mergingSkyframeAnalysisExecutionPhases= */ false,
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

    private boolean drive(Environment env) throws InterruptedException {
      return driver.drive(env);
    }
  }

  /**
   * Defines a {@link SkyFunction} that executes the gives state machine.
   *
   * <p>The function always has key {@link ROOT_KEY} and value {@link DONE_VALUE}. State machine
   * internals can be observed with consumers.
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
                  .drive(env)) {
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

  private boolean runMachine(StateMachine root) throws InterruptedException {
    return !StateMachineEvaluatorForTesting.run(
            root,
            new InMemoryMemoizingEvaluator(
                tester.getSkyFunctionMap(), new SequencedRecordingDifferencer()),
            EvaluationContext.newBuilder()
                .setKeepGoing(true)
                .setParallelism(TEST_PARALLELISM)
                .setEventHandler(reportedEvents)
                .build())
        .hasError();
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
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(KEY_A1, sink1);
      return this::step2;
    }

    public StateMachine step2(Tasks tasks) {
      tasks.lookUp(KEY_A2, sink2);
      return DONE;
    }
  }

  @Test
  public void smoke(@TestParameter boolean useTestingEvaluator) throws InterruptedException {
    var v1Sink = new SkyValueSink();
    var v2Sink = new SkyValueSink();
    Supplier<StateMachine> factory = () -> new TwoStepMachine(v1Sink, v2Sink);
    if (useTestingEvaluator) {
      assertThat(runMachine(factory.get())).isTrue();
    } else {
      assertThat(evalMachine(factory)).isEqualTo(2);
    }
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
    public StateMachine step(Tasks tasks) {
      // Starts submachines in parallel.
      tasks.enqueue(this::stepA1);
      tasks.enqueue(this::stepB1);
      return DONE;
    }

    private StateMachine stepA1(Tasks tasks) {
      tasks.lookUp(KEY_A1, sinkA1);
      return this::stepA2;
    }

    private StateMachine stepA2(Tasks tasks) {
      tasks.lookUp(KEY_A2, sinkA2);
      return this::stepA3;
    }

    private StateMachine stepA3(Tasks tasks) {
      tasks.lookUp(KEY_A3, sinkA3);
      return DONE;
    }

    private StateMachine stepB1(Tasks tasks) {
      tasks.lookUp(KEY_B1, sinkB1);
      return this::stepB2;
    }

    private StateMachine stepB2(Tasks tasks) {
      tasks.lookUp(KEY_B2, sinkB2);
      return this::stepB3;
    }

    private StateMachine stepB3(Tasks tasks) {
      tasks.lookUp(KEY_B3, sinkB3);
      return DONE;
    }
  }

  @Test
  public void parallelSubmachines_runInParallel(@TestParameter boolean useTestingEvaluator)
      throws InterruptedException {
    var a1Sink = new SkyValueSink();
    var a2Sink = new SkyValueSink();
    var a3Sink = new SkyValueSink();
    var b1Sink = new SkyValueSink();
    var b2Sink = new SkyValueSink();
    var b3Sink = new SkyValueSink();

    Supplier<StateMachine> factory =
        () -> new ExampleWithSubmachines(a1Sink, a2Sink, a3Sink, b1Sink, b2Sink, b3Sink);
    if (useTestingEvaluator) {
      assertThat(runMachine(factory.get())).isTrue();
    } else {
      assertThat(evalMachine(factory)).isEqualTo(3);
    }

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
                tasks -> {
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
                  return StateMachine.DONE;
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
      extends ValueOrExceptionProducer<StringValue, SomeErrorException>
      implements SkyKeyComputeState {
    @Override
    public StateMachine step(Tasks tasks) {
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
      return DONE;
    }
  }

  @Test
  public void valueOrExceptionProducer_propagatesValues() throws InterruptedException {
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrExceptionProducer::new);

              SkyValue value;
              try {
                if ((value = producer.tryProduceValue(env)) == null) {
                  return null;
                }
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
    var hasRestarted = new AtomicBoolean(false);
    tester.getOrCreate(KEY_A1).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrExceptionProducer::new);
              if (!hasRestarted.getAndSet(true)) {
                try {
                  // The first call returns null because a restart is needed to compute the
                  // requested key.
                  assertThat(producer.tryProduceValue(env)).isNull();
                } catch (SomeErrorException e) {
                  fail("Unexpecteded exception: " + e);
                }
                return null;
              }
              assertThrows(SomeErrorException.class, () -> producer.tryProduceValue(env));
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

  /**
   * This producer performs two concurrent lookups.
   *
   * <p>It is used to test the case where one of the two lookups succeeds with exception but the
   * other value is not available. The expected result is the exception propagates.
   *
   * <p>This scenario may occur during error bubbling.
   */
  private static class TwoLookupProducer
      extends ValueOrExceptionProducer<StringValue, SomeErrorException>
      implements SkyKeyComputeState {
    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(KEY_A1, unusedValue -> fail("should not be reachable"));
      tasks.lookUp(
          KEY_A2,
          SomeErrorException.class,
          (v, e) -> {
            if (v != null) {
              setValue((StringValue) v);
              return;
            }
            setException(e);
          });
      return DONE;
    }
  }

  @Test
  public void valueOrExceptionProducer_throwsExceptionsEvenWithIncompleteDeps()
      throws InterruptedException {
    var hasRestarted = new AtomicBoolean(false);
    var gotError = new AtomicBoolean(false);
    tester.getOrCreate(KEY_A2).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (unusedKey, env) -> {
              // Primes KEY_A2, making the error available.
              if (!hasRestarted.getAndSet(true)) {
                assertThat(env.getValue(KEY_A2)).isNull();
                return null;
              }
              var producer = env.getState(TwoLookupProducer::new);
              // At this point, KEY_A2 is available but KEY_A1 is not. The state machine is in an
              // incomplete state, but throws the exception anyway.
              var error =
                  assertThrows(SomeErrorException.class, () -> producer.tryProduceValue(env));
              gotError.set(true);
              throw new GenericFunctionException(error);
            });
    // keepGoing must be false below, otherwise the state machine will be run a second time when
    // KEY_A1 becomes available.
    var result = eval(ROOT_KEY, /* keepGoing= */ false);
    assertThat(gotError.get()).isTrue();
    assertThat(result.get(ROOT_KEY)).isNull();
    assertThatEvaluationResult(result).hasSingletonErrorThat(KEY_A2);
  }

  private static class SomeErrorException1 extends SomeErrorException {
    public SomeErrorException1(String msg) {
      super(msg);
    }
  }

  private static class SomeErrorException2 extends SomeErrorException {
    public SomeErrorException2(String msg) {
      super(msg);
    }
  }

  private static class SomeErrorException3 extends SomeErrorException {
    public SomeErrorException3(String msg) {
      super(msg);
    }
  }

  private static class StringOrException2Producer
      extends ValueOrException2Producer<StringValue, SomeErrorException1, SomeErrorException2>
      implements SkyKeyComputeState {
    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(
          KEY_A1,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException1(new SomeErrorException1(e.getMessage()));
            }
          });
      tasks.lookUp(
          KEY_B1,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException2(new SomeErrorException2(e.getMessage()));
            }
          });
      return t -> {
        if (getException1() == null && getException2() == null) {
          setValue(SUCCESS_VALUE);
        }
        return DONE;
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
              SkyValue value;
              try {
                if ((value = producer.tryProduceValue(env)) == null) {
                  return null;
                }
                assertThat(value).isEqualTo(SUCCESS_VALUE);
              } catch (SomeErrorException e) {
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
    var hasRestarted = new AtomicBoolean(false);
    SkyKey errorKey = trueForException1 ? KEY_A1 : KEY_B1;
    tester.getOrCreate(errorKey).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrException2Producer::new);
              if (!hasRestarted.getAndSet(true)) {
                try {
                  assertThat(producer.tryProduceValue(env)).isNull();
                } catch (SomeErrorException e) {
                  fail("Unexpecteded exception: " + e);
                }
                return null;
              }
              if (trueForException1) {
                assertThrows(SomeErrorException1.class, () -> producer.tryProduceValue(env));
              } else {
                assertThrows(SomeErrorException2.class, () -> producer.tryProduceValue(env));
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

  private static class StringOrException3Producer
      extends ValueOrException3Producer<
          StringValue, SomeErrorException1, SomeErrorException2, SomeErrorException3>
      implements SkyKeyComputeState {
    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(
          KEY_A1,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException1(new SomeErrorException1(e.getMessage()));
            }
          });
      tasks.lookUp(
          KEY_A2,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException2(new SomeErrorException2(e.getMessage()));
            }
          });
      tasks.lookUp(
          KEY_A3,
          SomeErrorException.class,
          (v, e) -> {
            if (e != null) {
              setException3(new SomeErrorException3(e.getMessage()));
            }
          });
      return t -> {
        if (getException1() == null && getException2() == null && getException3() == null) {
          setValue(SUCCESS_VALUE);
        }
        return DONE;
      };
    }
  }

  @Test
  public void valueOrException3Producer_propagatesValues() throws InterruptedException {
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrException3Producer::new);
              SkyValue value;
              try {
                if ((value = producer.tryProduceValue(env)) == null) {
                  return null;
                }
                assertThat(value).isEqualTo(SUCCESS_VALUE);
              } catch (SomeErrorException e) {
                fail("Unexpecteded exception: " + e);
              }
              return DONE_VALUE;
            });
    assertThat(eval(ROOT_KEY, /* keepGoing= */ false).get(ROOT_KEY)).isEqualTo(DONE_VALUE);
  }

  enum ValueOrException3ExceptionCase {
    ONE {
      @Override
      SkyKey errorKey() {
        return KEY_A1;
      }
    },
    TWO {
      @Override
      SkyKey errorKey() {
        return KEY_A2;
      }
    },
    THREE {
      @Override
      SkyKey errorKey() {
        return KEY_A3;
      }
    };

    abstract SkyKey errorKey();
  }

  @Test
  public void valueOrException3Producer_propagatesExceptions(
      @TestParameter ValueOrException3ExceptionCase exceptionCase, @TestParameter boolean keepGoing)
      throws InterruptedException {
    var hasRestarted = new AtomicBoolean(false);
    SkyKey errorKey = exceptionCase.errorKey();
    tester.getOrCreate(errorKey).unsetConstantValue().setHasError(true);
    tester
        .getOrCreate(ROOT_KEY)
        .setBuilder(
            (k, env) -> {
              var producer = env.getState(StringOrException3Producer::new);
              if (!hasRestarted.getAndSet(true)) {
                try {
                  assertThat(producer.tryProduceValue(env)).isNull();
                } catch (SomeErrorException e) {
                  fail("Unexpecteded exception: " + e);
                }
                return null;
              }
              switch (exceptionCase) {
                case ONE:
                  assertThrows(SomeErrorException1.class, () -> producer.tryProduceValue(env));
                  break;
                case TWO:
                  assertThrows(SomeErrorException2.class, () -> producer.tryProduceValue(env));
                  break;
                case THREE:
                  assertThrows(SomeErrorException3.class, () -> producer.tryProduceValue(env));
                  break;
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

  @Test
  public void lookupValue_matrix(
      @TestParameter LookupType lookupType,
      @TestParameter boolean useBatch,
      @TestParameter boolean useTestingEvaluator)
      throws InterruptedException {
    var sink = new OmniSink();
    Supplier<StateMachine> rootSupplier =
        () -> {
          var lookup = lookupType.newLookup(KEY_A1, sink);
          if (!useBatch) {
            return lookup;
          }
          return new BatchPair(lookup);
        };
    if (useTestingEvaluator) {
      assertThat(runMachine(rootSupplier.get())).isTrue();
    } else {
      var unused = defineRootMachine(rootSupplier);
      // There are no errors in this test so the keepGoing value is arbitrary.
      assertThat(eval(ROOT_KEY, /* keepGoing= */ true).get(ROOT_KEY)).isEqualTo(DONE_VALUE);
    }
    assertThat(sink.value).isEqualTo(VALUE_A1);
    assertThat(sink.exception).isNull();
  }

  enum EvaluationMode {
    NO_KEEP_GOING,
    KEEP_GOING,
    TEST_EVALUATOR,
  }

  @Test
  public void lookupErrors_matrix(
      @TestParameter LookupType lookupType,
      @TestParameter ExceptionCase exceptionCase,
      @TestParameter boolean useBatch,
      @TestParameter EvaluationMode evaluationMode)
      throws InterruptedException {
    var exception = exceptionCase.getException();
    tester
        .getOrCreate(KEY_A1)
        .unsetConstantValue()
        .setBuilder(
            (k, env) -> {
              throw new ExceptionWrapper(exception);
            });
    var sink = new OmniSink();
    Supplier<StateMachine> rootSupplier =
        () -> {
          var lookup = lookupType.newLookup(KEY_A1, sink);
          if (!useBatch) {
            return lookup;
          }
          return new BatchPair(lookup);
        };

    boolean keepGoing = false;
    switch (evaluationMode) {
      case TEST_EVALUATOR:
        assertThat(runMachine(rootSupplier.get())).isFalse();
        if (exceptionCase.exceptionOrdinal() > lookupType.exceptionCount()) {
          // Undeclared exception is not handled.
          assertThat(sink.exception).isNull();
        } else {
          // Declared exception is captured.
          assertThat(sink.exception).isEqualTo(exception);
        }
        return;
      case KEEP_GOING:
        keepGoing = true;
        break;
      case NO_KEEP_GOING:
        break;
    }

    var unused = defineRootMachine(rootSupplier);
    var result = eval(ROOT_KEY, keepGoing);
    assertThat(sink.value).isNull();
    if (exceptionCase.exceptionOrdinal() > lookupType.exceptionCount()) {
      // The exception was not handled.
      assertThat(sink.exception).isNull();
      assertThat(result.get(ROOT_KEY)).isNull();
      assertThatEvaluationResult(result).hasSingletonErrorThat(KEY_A1);
      return;
    }
    assertThat(sink.exception).isEqualTo(exception);
    if (keepGoing) {
      // The error is completely handled.
      assertThat(result.get(ROOT_KEY)).isEqualTo(DONE_VALUE);
      return;
    }
    assertThatEvaluationResult(result).hasSingletonErrorThat(KEY_A1);
    assertThat(result.get(ROOT_KEY)).isNull();
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

  // -------------------- Helpers for lookupErrors_matrix --------------------
  private static class Exception1 extends Exception {}

  private static class Exception2 extends Exception {}

  private static class Exception3 extends Exception {}

  private static class Exception4 extends Exception {}

  private static class ExceptionWrapper extends SkyFunctionException {
    private ExceptionWrapper(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }

  /**
   * Adds a secondary lookup in parallel with a given {@link StateMachine}.
   *
   * <p>This causes the {@link Environment#getValuesAndExceptions} codepath in {@link Driver#drive}
   * to be used instead of the {@link Lookup#doLookup} when there is a single lookup.
   */
  private static class BatchPair implements StateMachine {
    private final StateMachine other;

    private BatchPair(StateMachine other) {
      this.other = other;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      tasks.enqueue(other);
      tasks.lookUp(KEY_B1, v -> assertThat(v).isEqualTo(VALUE_B1));
      return DONE;
    }
  }

  private static class Lookup0 implements StateMachine {
    private final SkyKey key;
    private final Consumer<SkyValue> sink;

    private Lookup0(SkyKey key, Consumer<SkyValue> sink) {
      this.key = key;
      this.sink = sink;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(key, sink);
      return DONE;
    }
  }

  private static class Lookup1 implements StateMachine {
    private final SkyKey key;
    private final ValueOrExceptionSink<Exception1> sink;

    private Lookup1(SkyKey key, ValueOrExceptionSink<Exception1> sink) {
      this.key = key;
      this.sink = sink;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(key, Exception1.class, sink);
      return DONE;
    }
  }

  private static class Lookup2 implements StateMachine {
    private final SkyKey key;
    private final ValueOrException2Sink<Exception1, Exception2> sink;

    private Lookup2(SkyKey key, ValueOrException2Sink<Exception1, Exception2> sink) {
      this.key = key;
      this.sink = sink;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(key, Exception1.class, Exception2.class, sink);
      return DONE;
    }
  }

  private static class Lookup3 implements StateMachine {
    private final SkyKey key;
    private final ValueOrException3Sink<Exception1, Exception2, Exception3> sink;

    private Lookup3(SkyKey key, ValueOrException3Sink<Exception1, Exception2, Exception3> sink) {
      this.key = key;
      this.sink = sink;
    }

    @Override
    public StateMachine step(Tasks tasks) {
      tasks.lookUp(key, Exception1.class, Exception2.class, Exception3.class, sink);
      return DONE;
    }
  }

  private static class OmniSink
      implements Consumer<SkyValue>,
          StateMachine.ValueOrExceptionSink<Exception1>,
          StateMachine.ValueOrException2Sink<Exception1, Exception2>,
          StateMachine.ValueOrException3Sink<Exception1, Exception2, Exception3> {
    private SkyValue value;
    private Exception exception;

    @Override
    public void accept(SkyValue value) {
      checkState(this.value == null && exception == null);
      this.value = checkNotNull(value);
    }

    @Override
    public void acceptValueOrException(@Nullable SkyValue value, @Nullable Exception1 exception1) {
      checkState(this.value == null && exception == null);
      if (value != null) {
        this.value = value;
        return;
      }
      if (exception1 != null) {
        checkState(value == null);
        this.exception = exception1;
      }
    }

    @Override
    public void acceptValueOrException2(
        @Nullable SkyValue value,
        @Nullable Exception1 exception1,
        @Nullable Exception2 exception2) {
      checkState(this.value == null && exception == null);
      if (value != null) {
        checkState(exception1 == null && exception2 == null);
        this.value = value;
        return;
      }
      if (exception1 != null) {
        checkState(value == null && exception2 == null);
        this.exception = exception1;
        return;
      }
      if (exception2 != null) {
        checkState(value == null && exception1 == null);
        this.exception = exception2;
      }
    }

    @Override
    public void acceptValueOrException3(
        @Nullable SkyValue value,
        @Nullable Exception1 exception1,
        @Nullable Exception2 exception2,
        @Nullable Exception3 exception3) {
      checkState(this.value == null && exception == null);
      if (value != null) {
        checkState(exception1 == null && exception2 == null && exception3 == null);
        this.value = value;
        return;
      }
      if (exception1 != null) {
        checkState(value == null && exception2 == null && exception3 == null);
        this.exception = exception1;
        return;
      }
      if (exception2 != null) {
        checkState(value == null && exception1 == null && exception3 == null);
        this.exception = exception2;
        return;
      }
      if (exception3 != null) {
        checkState(value == null && exception1 == null && exception2 == null);
        this.exception = exception3;
      }
    }
  }

  private enum LookupType {
    LOOKUP0 {
      @Override
      StateMachine newLookup(SkyKey key, OmniSink sink) {
        return new Lookup0(key, sink);
      }

      @Override
      int exceptionCount() {
        return 0;
      }
    },
    LOOKUP1 {
      @Override
      StateMachine newLookup(SkyKey key, OmniSink sink) {
        return new Lookup1(key, sink);
      }

      @Override
      int exceptionCount() {
        return 1;
      }
    },
    LOOKUP2 {
      @Override
      StateMachine newLookup(SkyKey key, OmniSink sink) {
        return new Lookup2(key, sink);
      }

      @Override
      int exceptionCount() {
        return 2;
      }
    },
    LOOKUP3 {
      @Override
      StateMachine newLookup(SkyKey key, OmniSink sink) {
        return new Lookup3(key, sink);
      }

      @Override
      int exceptionCount() {
        return 3;
      }
    };

    abstract StateMachine newLookup(SkyKey key, OmniSink sink);

    abstract int exceptionCount();
  }

  private enum ExceptionCase {
    EXCEPTION1 {
      @Override
      Exception getException() {
        return new Exception1();
      }

      @Override
      int exceptionOrdinal() {
        return 1;
      }
    },
    EXCEPTION2 {
      @Override
      Exception getException() {
        return new Exception2();
      }

      @Override
      int exceptionOrdinal() {
        return 2;
      }
    },
    EXCEPTION3 {
      @Override
      Exception getException() {
        return new Exception3();
      }

      @Override
      int exceptionOrdinal() {
        return 3;
      }
    },
    EXCEPTION4 {
      @Override
      Exception getException() {
        return new Exception4();
      }

      @Override
      int exceptionOrdinal() {
        return 4;
      }
    };

    abstract Exception getException();

    abstract int exceptionOrdinal();
  }
}
