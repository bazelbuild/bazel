// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent.safeexecutor;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Ticker;
import com.google.common.util.concurrent.ForwardingExecutorService;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import java.time.Duration;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SafeExecutor}, {@link SafeDirectExecutor}, and {@link SafeExecutorOwner}. */
@RunWith(JUnit4.class)
public class SafeExecutorTest {

  @Test
  public void safeDirectExecutor_executesInline() {
    AtomicBoolean ran = new AtomicBoolean(false);
    RejectionHandlingRunnable task =
        new TestTask(
            () -> ran.set(true),
            t -> {
              throw new AssertionError(t);
            });

    SafeExecutor.safeDirectExecutor().execute(task);
    assertThat(ran.get()).isTrue();
  }

  @Test
  public void safeDirectExecutor_interruptedThread_executesInlineUnconditionally() {
    AtomicBoolean ran = new AtomicBoolean(false);
    AtomicBoolean rejected = new AtomicBoolean(false);
    RejectionHandlingRunnable task = new TestTask(() -> ran.set(true), t -> rejected.set(true));

    Thread.currentThread().interrupt();
    try {
      SafeExecutor.safeDirectExecutor().execute(task);
    } finally {
      // Clear interrupt status for subsequent tests
      Thread.interrupted();
    }

    assertThat(ran.get()).isTrue();
    assertThat(rejected.get()).isFalse();
  }

  @Test
  public void safeDirectExecutor_addCallback_success() {
    SettableFuture<String> future = SettableFuture.create();
    AtomicReference<String> resultRef = new AtomicReference<>();

    SafeExecutor.safeDirectExecutor()
        .addCallback(
            future,
            new FutureCallback<String>() {
              @Override
              public void onSuccess(String result) {
                resultRef.set(result);
              }

              @Override
              public void onFailure(Throwable t) {}
            });

    future.set("direct_result");
    assertThat(resultRef.get()).isEqualTo("direct_result");
  }

  @Test
  public void safeExecutorOwner_execute_normalTask() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    CountDownLatch latch = new CountDownLatch(1);
    AtomicBoolean ran = new AtomicBoolean(false);

    RejectionHandlingRunnable task =
        new TestTask(
            () -> {
              ran.set(true);
              latch.countDown();
            },
            t -> {
              throw new AssertionError(t);
            });

    owner.execute(task);
    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(ran.get()).isTrue();

    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_execute_rejectedTask_invokesHandleRejectionOffThread()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);
    owner.shutdownNow();

    CountDownLatch rejectedLatch = new CountDownLatch(1);
    AtomicReference<Throwable> caught = new AtomicReference<>();

    RejectionHandlingRunnable task =
        new TestTask(
            () -> {
              throw new AssertionError("Task should be rejected");
            },
            t -> {
              caught.set(t);
              rejectedLatch.countDown();
            });

    owner.execute(task);
    assertThat(rejectedLatch.await(5, SECONDS)).isTrue();
    assertThat(caught.get()).isInstanceOf(RejectedExecutionException.class);

    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_shutdownAndTerminationForTesting() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    assertThat(owner.isShutdownForTesting()).isFalse();
    assertThat(owner.isTerminatedForTesting()).isFalse();

    owner.shutdownNow();
    assertThat(owner.isShutdownForTesting()).isTrue();

    owner.awaitTermination(Duration.ofSeconds(5));
    assertThat(owner.isTerminatedForTesting()).isTrue();
  }

  @Test
  public void safeExecutorOwner_addCallback_success() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    SettableFuture<String> future = SettableFuture.create();
    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<String> resultRef = new AtomicReference<>();

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {
            resultRef.set(result);
            latch.countDown();
          }

          @Override
          public void onFailure(Throwable t) {
            latch.countDown();
          }
        });

    future.set("hello");
    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(resultRef.get()).isEqualTo("hello");

    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_addCallback_onSuccessExceptionDoesNotTriggerOnFailure()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    SettableFuture<String> future = SettableFuture.create();
    CountDownLatch latch = new CountDownLatch(1);
    AtomicBoolean failureCalled = new AtomicBoolean(false);

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {
            latch.countDown();
            throw new RuntimeException("onSuccess throw exception test");
          }

          @Override
          public void onFailure(Throwable t) {
            failureCalled.set(true);
          }
        });

    future.set("success");
    assertThat(latch.await(5, SECONDS)).isTrue();
    // Wait a brief moment to ensure onFailure is never invoked
    Thread.sleep(100);
    assertThat(failureCalled.get()).isFalse();

    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_addCallback_futureFailure() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    SettableFuture<String> future = SettableFuture.create();
    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Throwable> failureRef = new AtomicReference<>();

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {}

          @Override
          public void onFailure(Throwable t) {
            failureRef.set(t);
            latch.countDown();
          }
        });

    Exception expectedException = new IllegalArgumentException("bad argument");
    future.setException(expectedException);
    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(failureRef.get()).isEqualTo(expectedException);

    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_addCallback_rejectionPreservesRootCauseDirectly() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    SettableFuture<String> future = SettableFuture.create();
    Exception rootCause = new IllegalStateException("root failure");
    future.setException(rootCause);

    // Shutdown delegate so callback execution is rejected
    owner.shutdownNow();

    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Throwable> failureRef = new AtomicReference<>();

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {}

          @Override
          public void onFailure(Throwable t) {
            failureRef.set(t);
            latch.countDown();
          }
        });

    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(failureRef.get()).isEqualTo(rootCause);

    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_addCallback_successfulFuture_rejectionReportsRejectionException()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    ListenableFuture<String> future = immediateFuture("success_value");

    // Shutdown delegate so callback execution is rejected
    owner.shutdownNow();

    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Throwable> failureRef = new AtomicReference<>();

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {}

          @Override
          public void onFailure(Throwable t) {
            failureRef.set(t);
            latch.countDown();
          }
        });

    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(failureRef.get()).isInstanceOf(RejectedExecutionException.class);

    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_addCallback_cancelledFuture_rejectionReportsCancellationException()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    SettableFuture<String> future = SettableFuture.create();
    future.cancel(/* mayInterruptIfRunning= */ false);

    // Shutdown delegate so callback execution is rejected
    owner.shutdownNow();

    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Throwable> failureRef = new AtomicReference<>();

    owner.addCallback(
        future,
        new FutureCallback<String>() {
          @Override
          public void onSuccess(String result) {}

          @Override
          public void onFailure(Throwable t) {
            failureRef.set(t);
            latch.countDown();
          }
        });

    assertThat(latch.await(5, SECONDS)).isTrue();
    assertThat(failureRef.get()).isInstanceOf(CancellationException.class);

    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeCallbackListener_handleRejection_pendingFuture_returnsRejectionException() {
    SettableFuture<String> pendingFuture = SettableFuture.create();
    AtomicReference<Throwable> failureRef = new AtomicReference<>();
    var listener =
        new SafeCallbackListener<>(
            pendingFuture,
            new FutureCallback<String>() {
              @Override
              public void onSuccess(String result) {}

              @Override
              public void onFailure(Throwable t) {
                failureRef.set(t);
              }
            },
            SafeExecutor.safeDirectExecutor());

    var rejection = new RejectedExecutionException("rejected while pending");
    listener.handleRejection(rejection);

    assertThat(failureRef.get()).isSameInstanceAs(rejection);
  }

  @Test
  public void safeExecutorOwner_shutdownNow_executesPlainRunnableUnderInterrupt() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    // Block the single thread so subsequent tasks are queued
    CountDownLatch blockLatch = new CountDownLatch(1);
    delegate.execute(
        () -> {
          try {
            blockLatch.await();
          } catch (InterruptedException e) {
            // expected
          }
        });

    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    CountDownLatch taskRunLatch = new CountDownLatch(1);
    AtomicBoolean wasInterrupted = new AtomicBoolean(false);

    // Enqueue a plain Runnable (NOT RejectionHandlingRunnable)
    Runnable plainTask =
        () -> {
          wasInterrupted.set(Thread.currentThread().isInterrupted());
          taskRunLatch.countDown();
        };
    delegate.execute(plainTask);

    owner.shutdownNow();
    blockLatch.countDown();

    assertThat(taskRunLatch.await(5, SECONDS)).isTrue();
    assertThat(wasInterrupted.get()).isTrue();

    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_nullArgs_throwsNullPointerException() {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    assertThrows(NullPointerException.class, () -> owner.execute(null));
    DummyCallback callback = new DummyCallback();
    assertThrows(NullPointerException.class, () -> owner.addCallback(null, callback));
    SettableFuture<Object> future = SettableFuture.create();
    assertThrows(NullPointerException.class, () -> owner.addCallback(future, null));
    SafeExecutor safeExecutor = SafeExecutor.safeDirectExecutor();
    assertThrows(NullPointerException.class, () -> safeExecutor.execute(null));
    SafeExecutor safeExecutor2 = SafeExecutor.safeDirectExecutor();
    DummyCallback callback2 = new DummyCallback();
    assertThrows(NullPointerException.class, () -> safeExecutor2.addCallback(null, callback2));
    SafeExecutor safeExecutor3 = SafeExecutor.safeDirectExecutor();
    SettableFuture<Object> future2 = SettableFuture.create();
    assertThrows(NullPointerException.class, () -> safeExecutor3.addCallback(future2, null));

    owner.shutdownNow();
  }

  @Test
  public void safeDirectExecutor_thinWrappers() throws Exception {
    SafeExecutor executor = SafeExecutor.safeDirectExecutor();

    ListenableFuture<String> submitFut = SafeFutures.submit(() -> "hello", executor);
    assertThat(submitFut.get()).isEqualTo("hello");

    ListenableFuture<String> submitAsyncFut =
        SafeFutures.submitAsync(() -> immediateFuture("async_hello"), executor);
    assertThat(submitAsyncFut.get()).isEqualTo("async_hello");

    ListenableFuture<String> transformFut =
        SafeFutures.transform(immediateFuture("input"), val -> val + "_transformed", executor);
    assertThat(transformFut.get()).isEqualTo("input_transformed");

    ListenableFuture<String> transformAsyncFut =
        SafeFutures.transformAsync(
            immediateFuture("input"), val -> immediateFuture(val + "_async_transformed"), executor);
    assertThat(transformAsyncFut.get()).isEqualTo("input_async_transformed");

    ListenableFuture<String> f1 = immediateFuture("part1");
    ListenableFuture<String> f2 = immediateFuture("part2");

    ListenableFuture<String> callFut =
        SafeFutures.call(
            Futures.whenAllComplete(f1, f2), () -> f1.get() + "_" + f2.get(), executor);
    assertThat(callFut.get()).isEqualTo("part1_part2");

    ListenableFuture<String> callAsyncFut =
        SafeFutures.callAsync(
            Futures.whenAllComplete(f1, f2),
            () -> immediateFuture(f1.get() + "_" + f2.get() + "_async"),
            executor);
    assertThat(callAsyncFut.get()).isEqualTo("part1_part2_async");
  }

  @Test
  public void safeExecutorOwner_thinWrappers() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    ListenableFuture<String> submitFut = SafeFutures.submit(() -> "hello", owner);
    assertThat(submitFut.get()).isEqualTo("hello");

    ListenableFuture<String> submitAsyncFut =
        SafeFutures.submitAsync(() -> immediateFuture("async_hello"), owner);
    assertThat(submitAsyncFut.get()).isEqualTo("async_hello");

    ListenableFuture<String> transformFut =
        SafeFutures.transform(immediateFuture("input"), val -> val + "_transformed", owner);
    assertThat(transformFut.get()).isEqualTo("input_transformed");

    ListenableFuture<String> transformAsyncFut =
        SafeFutures.transformAsync(
            immediateFuture("input"), val -> immediateFuture(val + "_async_transformed"), owner);
    assertThat(transformAsyncFut.get()).isEqualTo("input_async_transformed");

    ListenableFuture<String> f1 = immediateFuture("part1");
    ListenableFuture<String> f2 = immediateFuture("part2");

    ListenableFuture<String> callFut =
        SafeFutures.call(Futures.whenAllComplete(f1, f2), () -> f1.get() + "_" + f2.get(), owner);
    assertThat(callFut.get()).isEqualTo("part1_part2");

    ListenableFuture<String> callAsyncFut =
        SafeFutures.callAsync(
            Futures.whenAllComplete(f1, f2),
            () -> immediateFuture(f1.get() + "_" + f2.get() + "_async"),
            owner);
    assertThat(callAsyncFut.get()).isEqualTo("part1_part2_async");

    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));
  }

  @Test
  public void safeExecutorOwner_submit_rejectedExecution_returnsImmediateFailedFuture()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);
    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));

    ListenableFuture<String> fut = SafeFutures.submit(() -> "should_reject", owner);
    ExecutionException e = assertThrows(ExecutionException.class, fut::get);
    assertThat(e).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
  }

  @Test
  public void safeExecutorOwner_submitAsync_rejectedExecution_returnsImmediateFailedFuture()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);
    owner.shutdownNow();
    owner.awaitTermination(Duration.ofSeconds(5));

    ListenableFuture<String> fut =
        SafeFutures.submitAsync(() -> immediateFuture("should_reject"), owner);
    ExecutionException e = assertThrows(ExecutionException.class, fut::get);
    assertThat(e).hasCauseThat().isInstanceOf(RejectedExecutionException.class);
  }

  @Test
  public void safeExecutorOwner_awaitTermination_returnsTrueWhenTerminated() {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);
    owner.shutdownNow();

    assertThat(owner.awaitTermination(Duration.ofSeconds(5))).isTrue();
  }

  @Test
  public void safeExecutorOwner_awaitTermination_waitsForInFlightRejectionDispatcherTask()
      throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);
    owner.shutdownNow();

    CountDownLatch taskStarted = new CountDownLatch(1);
    CountDownLatch taskRelease = new CountDownLatch(1);
    CountDownLatch taskDone = new CountDownLatch(1);

    // Trigger dispatchRejection by submitting a task to the shut down delegate
    owner.execute(
        new TestTask(
            () -> {},
            t -> {
              taskStarted.countDown();
              try {
                taskRelease.await();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              } finally {
                taskDone.countDown();
              }
            }));

    assertThat(taskStarted.await(5, SECONDS)).isTrue();

    // Release the in-flight rejection task after 50ms in a background thread
    new Thread(
            () -> {
              try {
                Thread.sleep(50);
              } catch (InterruptedException interrupted) {
                throw new IllegalStateException(interrupted);
              }
              taskRelease.countDown();
            })
        .start();

    // awaitTermination should wait for the in-flight task on rejectionDispatcher and return true
    assertThat(owner.awaitTermination(Duration.ofSeconds(5))).isTrue();
    assertThat(taskDone.await(5, SECONDS)).isTrue();
  }

  @Test
  public void safeExecutorOwner_awaitTermination_passesExactRemainingDurationToDispatcher() {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    AtomicInteger readCount = new AtomicInteger(0);
    Ticker ticker =
        new Ticker() {
          @Override
          public long read() {
            int count = readCount.getAndIncrement();
            // Call 0 (startTimeNanos): 1s. Call 1 (Phase 3 elapsed): 3s (2s elapsed).
            return count == 0 ? 1_000_000_000L : 3_000_000_000L;
          }
        };

    RecordingExecutorService recordingDispatcher =
        new RecordingExecutorService(Executors.newVirtualThreadPerTaskExecutor());
    SafeExecutorOwner owner = new SafeExecutorOwner(delegate, recordingDispatcher, ticker);
    owner.shutdownNow();

    assertThat(owner.awaitTermination(Duration.ofSeconds(5))).isTrue();
    assertThat(recordingDispatcher.shutdownCalled.get()).isTrue();
    // Exactly 5s - 2s = 3s (3_000_000_000 ns). Kills any off-by-one / arithmetic mutants on
    // elapsedNanos.
    assertThat(recordingDispatcher.recordedAwaitTimeout.get()).isEqualTo(Duration.ofSeconds(3));
  }

  @Test
  public void safeExecutorOwner_awaitTermination_dispatcherFailsToTerminate_returnsFalse() {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    RecordingExecutorService failingDispatcher =
        new RecordingExecutorService(Executors.newVirtualThreadPerTaskExecutor());
    failingDispatcher.awaitTerminationResult = false;

    SafeExecutorOwner owner =
        new SafeExecutorOwner(delegate, failingDispatcher, Ticker.systemTicker());
    owner.shutdownNow();

    assertThat(owner.awaitTermination(Duration.ofSeconds(5))).isFalse();
    assertThat(failingDispatcher.shutdownCalled.get()).isTrue();
  }

  @Test
  public void safeExecutorOwner_awaitTermination_delegateTimeout_returnsFalse() throws Exception {
    ExecutorService delegate = Executors.newSingleThreadExecutor();
    CountDownLatch blockLatch = new CountDownLatch(1);
    CountDownLatch taskRunning = new CountDownLatch(1);
    delegate.execute(
        () -> {
          taskRunning.countDown();
          try {
            blockLatch.await();
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
          }
        });

    SafeExecutorOwner owner = new SafeExecutorOwner(delegate);

    try {
      assertThat(taskRunning.await(5, SECONDS)).isTrue();
      // Delegate cannot terminate within 50ms, awaitTermination should return false early
      assertThat(owner.awaitTermination(Duration.ofMillis(50))).isFalse();
    } finally {
      blockLatch.countDown();
      owner.shutdownNow();
      owner.awaitTermination(Duration.ofSeconds(5));
    }
  }

  private static final class RecordingExecutorService extends ForwardingExecutorService {
    private final ExecutorService delegate;
    final AtomicBoolean shutdownCalled = new AtomicBoolean(false);
    final AtomicReference<Duration> recordedAwaitTimeout = new AtomicReference<>();
    boolean awaitTerminationResult = true;

    RecordingExecutorService(ExecutorService delegate) {
      this.delegate = delegate;
    }

    @Override
    protected ExecutorService delegate() {
      return delegate;
    }

    @Override
    public void shutdown() {
      shutdownCalled.set(true);
      super.shutdown();
    }

    @Override
    public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
      recordedAwaitTimeout.set(Duration.ofNanos(unit.toNanos(timeout)));
      return awaitTerminationResult && super.awaitTermination(timeout, unit);
    }
  }

  private static final class TestTask implements RejectionHandlingRunnable {
    private final Runnable runAction;
    private final Consumer<Throwable> rejectionAction;

    TestTask(Runnable runAction, Consumer<Throwable> rejectionAction) {
      this.runAction = runAction;
      this.rejectionAction = rejectionAction;
    }

    @Override
    public void run() {
      runAction.run();
    }

    @Override
    public void handleRejection(Throwable t) {
      rejectionAction.accept(t);
    }
  }

  private static final class DummyCallback implements FutureCallback<Object> {
    @Override
    public void onSuccess(Object result) {}

    @Override
    public void onFailure(Throwable t) {}
  }
}
