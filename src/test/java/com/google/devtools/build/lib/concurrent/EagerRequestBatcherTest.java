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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.RequestBatching.CallbackMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.FutureMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.Multiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.Operation;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class EagerRequestBatcherTest {

  @Test
  public void simpleSubmit_executes() throws Exception {
    var batcher =
        EagerRequestBatcher.<Request, Response>create(
            requests -> immediateFuture(respondTo(requests)),
            directExecutor(),
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 1,
            directExecutor());
    ListenableFuture<Response> response = batcher.submit(new Request(1));
    assertThat(response.get()).isEqualTo(new Response(1));
  }

  @Test
  public void verifyEagerSendingBatchingAndCompletionFlows() throws Exception {
    var multiplexer = new SettableMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 2,
            directExecutor());

    // Scenario A: Eager sending due to low concurrency.
    // State established:
    // - targetConcurrentRequests = 2, maxBatchSize = 10.
    // - R1 and R2 are submitted and eagerly executed immediately as single-item batches
    //   because inFlightCount < targetConcurrentRequests.
    // - Active batches: [R1], [R2] -> inFlightCount = 2.
    // - R3 and R4 are submitted but queued because inFlightCount (2)
    //   >= targetConcurrentRequests (2) and queue size (2) < maxBatchSize (10).
    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(0);
    BatchedOperations batch1 = multiplexer.queue.take();
    assertThat(batch1.requests()).containsExactly(new Request(1));

    ListenableFuture<Response> r2 = batcher.submit(new Request(2));
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(0);
    BatchedOperations batch2 = multiplexer.queue.take();
    assertThat(batch2.requests()).containsExactly(new Request(2));

    ListenableFuture<Response> r3 = batcher.submit(new Request(3));
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(1);
    assertThat(multiplexer.queue).isEmpty();

    ListenableFuture<Response> r4 = batcher.submit(new Request(4));
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(2);
    assertThat(multiplexer.queue).isEmpty();

    // Scenario B: Batching due to high concurrency (Max Batch Size trigger).
    // State carried over from A:
    // - Active batches: [R1], [R2] -> inFlightCount = 2.
    // - Queued requests: [R3, R4] -> queueSize = 2.
    // Action: Submit 8 more requests (R5 to R12) to reach maxBatchSize (10).
    // State established:
    // - The queue reaches maxBatchSize (10) and is flushed immediately as a batch [R3-R12].
    // - Active batches: [R1], [R2], [R3-R12] -> inFlightCount = 3.
    // - R13 is submitted and queued because inFlightCount (3) >= targetConcurrentRequests (2)
    //   and queue size (1) < maxBatchSize (10).
    List<ListenableFuture<Response>> queuedResponses = new ArrayList<>();
    queuedResponses.add(r3);
    queuedResponses.add(r4);
    for (int i = 5; i <= 12; i++) {
      queuedResponses.add(batcher.submit(new Request(i)));
    }
    assertThat(batcher.getInFlightCount()).isEqualTo(3);
    assertThat(batcher.getQueueSize()).isEqualTo(0);
    BatchedOperations batch3 = multiplexer.queue.take();
    assertThat(batch3.requests()).hasSize(10);
    assertThat(batch3.requests().stream().map(Request::x).collect(toImmutableList()))
        .containsExactly(3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        .inOrder();

    ListenableFuture<Response> r13 = batcher.submit(new Request(13));
    assertThat(batcher.getInFlightCount()).isEqualTo(3);
    assertThat(batcher.getQueueSize()).isEqualTo(1);

    // Scenario C: Completion triggering queued work.
    // State carried over from B:
    // - Active batches: [R1], [R2], [R3-R12] -> inFlightCount = 3.
    // - Queued requests: [R13] -> queueSize = 1.
    // Action: Complete active batches and observe queue draining.
    // State transitions:
    // 1. Complete [R1] -> inFlightCount decrements to 2. R13 remains queued because
    //    inFlightCount (2) is not < targetConcurrentRequests (2).
    // 2. Complete [R2] -> inFlightCount decrements to 1. Since inFlightCount (1) <
    //    targetConcurrentRequests (2), the queued R13 is eagerly flushed and executed.
    // - Active batches: [R3-R12], [R13] -> inFlightCount = 2.
    // - Queued requests: none.
    batch1.setSimpleResponses();
    assertThat(r1.get()).isEqualTo(new Response(1));
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(1);
    assertThat(multiplexer.queue).isEmpty();

    batch2.setSimpleResponses();
    assertThat(r2.get()).isEqualTo(new Response(2));
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(0);
    BatchedOperations batch4 = multiplexer.queue.take();
    assertThat(batch4.requests()).containsExactly(new Request(13));

    batch3.setSimpleResponses();
    batch4.setSimpleResponses();
    assertThat(r13.get()).isEqualTo(new Response(13));
    for (int i = 0; i < queuedResponses.size(); i++) {
      assertThat(queuedResponses.get(i).get()).isEqualTo(new Response(i + 3));
    }
  }

  @Test
  public void synchronousException_decrementsInFlightAndFailsFutures() throws Exception {
    var failure = new RuntimeException("Sync Failure");
    Multiplexer<Request, Response> faultyMultiplexer =
        requests -> {
          throw failure;
        };
    var strategy =
        RequestBatching.createBatchExecutionStrategy(faultyMultiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 1,
            directExecutor());

    ListenableFuture<Response> response = batcher.submit(new Request(1));

    assertThat(batcher.getInFlightCount()).isEqualTo(0);
    var thrown = assertThrows(ExecutionException.class, response::get);
    assertThat(thrown).hasCauseThat().isEqualTo(failure);

    // Verify we can still submit after failure
    var multiplexer = new SettableMultiplexer();
    var goodStrategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var goodBatcher =
        new EagerRequestBatcher<>(
            goodStrategy,
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 1,
            directExecutor());

    ListenableFuture<Response> goodResponse = goodBatcher.submit(new Request(2));
    assertThat(goodBatcher.getInFlightCount()).isEqualTo(1);
    multiplexer.queue.take().setSimpleResponses();
    assertThat(goodResponse.get()).isEqualTo(new Response(2));
  }

  @Test
  public void executor_runsCallbacksOnInjectedExecutor() throws Exception {
    var multiplexer = new SettableMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var executorThreads = new ConcurrentLinkedQueue<Thread>();
    Executor recordingExecutor =
        command ->
            new Thread(
                    () -> {
                      executorThreads.add(Thread.currentThread());
                      command.run();
                    })
                .start();

    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 1,
            recordingExecutor);

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    BatchedOperations batch1 = multiplexer.queue.take();

    // Queue a second request
    ListenableFuture<Response> r2 = batcher.submit(new Request(2));
    assertThat(batcher.getQueueSize()).isEqualTo(1);

    // Complete the first batch. This should trigger onBatchComplete on the recordingExecutor.
    batch1.setSimpleResponses();
    r1.get(); // Wait for completion

    // Wait for the second batch to be executed (it should be triggered by onBatchComplete)
    BatchedOperations batch2 = multiplexer.queue.take();
    assertThat(batch2.requests()).containsExactly(new Request(2));

    // Verify that onBatchComplete ran on a thread from recordingExecutor
    assertThat(executorThreads).isNotEmpty();
    Thread callbackThread = executorThreads.peek();
    assertThat(callbackThread).isNotEqualTo(Thread.currentThread());

    batch2.setSimpleResponses();
    assertThat(r2.get()).isEqualTo(new Response(2));
  }

  @Test
  public void queuePool_safety_nestedSubmissions() throws Exception {
    var multiplexer = new SettableMultiplexer();
    var batcherRef = new AtomicReference<EagerRequestBatcher<Request, Response>>();
    var nestedResponseRef = new AtomicReference<ListenableFuture<Response>>();

    var interceptingMultiplexer =
        new Multiplexer<Request, Response>() {
          private boolean submittedNested = false;

          @Override
          public ListenableFuture<List<Response>> execute(List<Request> requests) {
            if (!submittedNested) {
              submittedNested = true;
              // Submit a nested request. This will run on the same thread.
              nestedResponseRef.set(batchRefRef(batcherRef).submit(new Request(99)));
            }
            return multiplexer.execute(requests);
          }
        };

    var goodStrategy =
        RequestBatching.createBatchExecutionStrategy(interceptingMultiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            goodStrategy,
            new QueuePool<Request, Response>(10),
            /* targetConcurrentRequests= */ 1,
            directExecutor());
    batcherRef.set(batcher);

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));

    // At this point, interceptingMultiplexer should have run, and submitted R99.
    // R1 triggered immediate execution, so it called execute().
    // Inside execute(), R99 was submitted.
    // Since targetConcurrentRequests is 1, and inFlightCount is 1 (for R1), R99 should be queued.
    assertThat(batcher.getQueueSize()).isEqualTo(1);
    assertThat(batcher.getInFlightCount()).isEqualTo(1);

    BatchedOperations batch1 = multiplexer.queue.take();
    assertThat(batch1.requests()).containsExactly(new Request(1));

    // Complete batch 1. This should trigger execution of R99.
    batch1.setSimpleResponses();
    r1.get();

    BatchedOperations batch2 = multiplexer.queue.take();
    assertThat(batch2.requests()).containsExactly(new Request(99));
    batch2.setSimpleResponses();

    assertThat(nestedResponseRef.get().get()).isEqualTo(new Response(99));
  }

  private static <T, R> EagerRequestBatcher<T, R> batchRefRef(
      AtomicReference<EagerRequestBatcher<T, R>> ref) {
    return ref.get();
  }

  @Test
  public void callbackMultiplexer_integration() throws Exception {
    var events = new LinkedBlockingQueue<String>();
    CallbackMultiplexer<Request, Response> callbackMultiplexer =
        (requests, sinks) -> {
          events.add("execute");
          for (int i = 0; i < requests.size(); i++) {
            sinks.get(i).acceptResponse(new Response(requests.get(i).x()));
          }
          return () -> events.add("cleanup");
        };

    var batcher =
        EagerRequestBatcher.<Request, Response>createWithCallbackMultiplexer(
            callbackMultiplexer,
            new QueuePool<Request, Response>(2),
            /* targetConcurrentRequests= */ 1,
            directExecutor());

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(r1.get()).isEqualTo(new Response(1));
    assertThat(events.take()).isEqualTo("execute");
    assertThat(events.take()).isEqualTo("cleanup");
  }

  @Test
  public void futureMultiplexer_integration() throws Exception {
    FutureMultiplexer<Request, Response> futureMultiplexer =
        (requests, sinks) -> {
          for (int i = 0; i < requests.size(); i++) {
            sinks.get(i).acceptFuture(immediateFuture(new Response(requests.get(i).x())));
          }
        };

    var batcher =
        EagerRequestBatcher.<Request, Response>createWithFutureMultiplexer(
            futureMultiplexer,
            new QueuePool<Request, Response>(2),
            /* targetConcurrentRequests= */ 1,
            directExecutor());

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(r1.get()).isEqualTo(new Response(1));
  }

  @Test
  public void sharedQueuePool_worksWithoutIssues() throws Exception {
    var pool = new QueuePool<Request, Response>(10);
    var multiplexer1 = new SettableMultiplexer();
    var multiplexer2 = new SettableMultiplexer();

    var strategy1 = RequestBatching.createBatchExecutionStrategy(multiplexer1, directExecutor());
    var strategy2 = RequestBatching.createBatchExecutionStrategy(multiplexer2, directExecutor());

    var batcher1 =
        new EagerRequestBatcher<>(
            strategy1, pool, /* targetConcurrentRequests= */ 1, directExecutor());
    var batcher2 =
        new EagerRequestBatcher<>(
            strategy2, pool, /* targetConcurrentRequests= */ 1, directExecutor());

    var testThread =
        new Thread(
            () -> {
              try {
                ListenableFuture<Response> r1 = batcher1.submit(new Request(1));
                BatchedOperations batch1 = multiplexer1.queue.take();
                batch1.setSimpleResponses();
                r1.get();

                ListenableFuture<Response> r2 = batcher2.submit(new Request(2));
                BatchedOperations batch2 = multiplexer2.queue.take();
                batch2.setSimpleResponses();
                r2.get();
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
              } catch (ExecutionException e) {
                throw new RuntimeException(e);
              }
            });
    testThread.start();
    testThread.join();
  }

  @Test
  public void parameterValidation() {
    assertThrows(IllegalArgumentException.class, () -> new QueuePool<Object, Object>(0));
    assertThrows(IllegalArgumentException.class, () -> new QueuePool<Object, Object>(-1));

    var pool = new QueuePool<Request, Response>(10);
    var strategy =
        RequestBatching.createBatchExecutionStrategy(new SettableMultiplexer(), directExecutor());

    assertThrows(
        IllegalArgumentException.class,
        () -> new EagerRequestBatcher<>(strategy, pool, 0, directExecutor()));
    assertThrows(
        IllegalArgumentException.class,
        () -> new EagerRequestBatcher<>(strategy, pool, -1, directExecutor()));
  }

  private static class SettableMultiplexer implements Multiplexer<Request, Response> {
    private final LinkedBlockingQueue<BatchedOperations> queue = new LinkedBlockingQueue<>();

    @Override
    public ListenableFuture<List<Response>> execute(List<Request> requests) {
      var responses = SettableFuture.<List<Response>>create();
      queue.add(new BatchedOperations(requests, responses));
      return responses;
    }
  }

  private record BatchedOperations(
      List<Request> requests, SettableFuture<List<Response>> responses) {
    private void setSimpleResponses() {
      responses().set(respondTo(requests()));
    }
  }

  @Test
  public void targetConcurrencyStrictness_underHighContention() throws Exception {
    final int targetConcurrency = 4;
    final int maxBatchSize = 10_000;
    // With these parameters, there are at most 6_400 requests sent, which is less than the
    // maxBatchSize of 10_000 so concurrency should never exceed targetConcurrency.
    final int numThreads = 32;
    final int submissionsPerThread = 200;
    // Under high concurrent load, we need to artificially introduce execution latency
    // at the multiplexer level to "stretch" the execution window. This allows concurrent
    // executions to pile up to the targetConcurrency (4) so we can verify that the batcher
    // strictly bounds this overlap and does not exceed it.
    // (Note: The baseline concurrent execution of >= 2 is guaranteed by active coordination
    // inside LatencyMultiplexer, but this latency is still required to reach peak concurrency).
    final int latencyMs = 20;

    ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
    ExecutorService multiplexerExecutor = Executors.newCachedThreadPool();

    try {
      LatencyMultiplexer multiplexer = new LatencyMultiplexer(latencyMs, multiplexerExecutor);
      var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
      var batcher =
          new EagerRequestBatcher<>(
              strategy,
              new QueuePool<Request, Response>(maxBatchSize),
              targetConcurrency,
              directExecutor());

      CyclicBarrier barrier = new CyclicBarrier(numThreads);
      CountDownLatch latch = new CountDownLatch(numThreads);
      ConcurrentLinkedQueue<ListenableFuture<Response>> futures = new ConcurrentLinkedQueue<>();

      for (int i = 0; i < numThreads; i++) {
        final int threadId = i;
        executorService.execute(
            () -> {
              try {
                barrier.await();
                for (int j = 0; j < submissionsPerThread; j++) {
                  futures.add(batcher.submit(new Request(threadId * 1000 + j)));
                }
              } catch (InterruptedException e) {
                multiplexer.errors.add(e);
                Thread.currentThread().interrupt();
              } catch (Exception e) {
                multiplexer.errors.add(e);
              } finally {
                latch.countDown();
              }
            });
      }

      latch.await();

      for (ListenableFuture<Response> future : futures) {
        future.get();
      }

      assertThat(multiplexer.errors).isEmpty();
      assertThat(multiplexer.maxConcurrentExecutions.get()).isAtMost(targetConcurrency);
      // Under remote scheduling delays, we might not always reach peak target concurrency
      // concurrently, but we must at least verify concurrent execution (>= 2) without exceeding
      // target concurrency limit.
      assertThat(multiplexer.maxConcurrentExecutions.get()).isAtLeast(2);

    } finally {
      executorService.shutdown();
      multiplexerExecutor.shutdown();
    }
  }

  @Test
  public void saturationFlush_bypassesConcurrencyLimit() throws Exception {
    int targetConcurrency = 1;
    int maxBatchSize = 10;
    var multiplexer = new SettableMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(maxBatchSize),
            targetConcurrency,
            directExecutor());

    List<ListenableFuture<Response>> futures = new ArrayList<>();
    for (int i = 1; i <= 50; i++) {
      futures.add(batcher.submit(new Request(i)));
    }

    assertThat(multiplexer.queue).hasSize(5);

    BatchedOperations b1 = multiplexer.queue.take();
    assertThat(b1.requests()).containsExactly(new Request(1));

    BatchedOperations b2 = multiplexer.queue.take();
    assertThat(b2.requests()).hasSize(10);
    assertThat(b2.requests().stream().map(Request::x).collect(toImmutableList()))
        .containsExactly(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

    BatchedOperations b3 = multiplexer.queue.take();
    assertThat(b3.requests()).hasSize(10);

    BatchedOperations b4 = multiplexer.queue.take();
    assertThat(b4.requests()).hasSize(10);

    BatchedOperations b5 = multiplexer.queue.take();
    assertThat(b5.requests()).hasSize(10);

    assertThat(multiplexer.queue).isEmpty();

    assertThat(batcher.getQueueSize()).isEqualTo(9);
    assertThat(batcher.getInFlightCount()).isEqualTo(5);

    // Complete Batch 1. inFlightCount should go 5 -> 4.
    b1.setSimpleResponses();
    futures.get(0).get();
    assertThat(batcher.getInFlightCount()).isEqualTo(4);
    assertThat(batcher.getQueueSize()).isEqualTo(9);

    // Complete Batch 2. inFlightCount should go 4 -> 3.
    b2.setSimpleResponses();
    futures.get(1).get();
    assertThat(batcher.getInFlightCount()).isEqualTo(3);
    assertThat(batcher.getQueueSize()).isEqualTo(9);

    // Complete Batch 3. inFlightCount should go 3 -> 2.
    b3.setSimpleResponses();
    futures.get(11).get();
    assertThat(batcher.getInFlightCount()).isEqualTo(2);
    assertThat(batcher.getQueueSize()).isEqualTo(9);

    // Complete Batch 4. inFlightCount should go 2 -> 1.
    b4.setSimpleResponses();
    futures.get(21).get();
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(9);

    // Complete Batch 5. This should reopen capacity (1 -> 0) and trigger flush of active buffer (0
    // -> 1).
    b5.setSimpleResponses();
    futures.get(31).get();

    BatchedOperations b6 = multiplexer.queue.take();
    assertThat(b6.requests()).hasSize(9);
    assertThat(b6.requests().stream().map(Request::x).collect(toImmutableList()))
        .containsExactly(42, 43, 44, 45, 46, 47, 48, 49, 50);

    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(0);

    b6.setSimpleResponses();
    for (ListenableFuture<Response> future : futures) {
      future.get();
    }
    assertThat(batcher.getInFlightCount()).isEqualTo(0);
  }

  @Test
  public void exceptionHandling_preventsSlotLeaking() throws Exception {
    int targetConcurrency = 1;
    int maxBatchSize = 10;
    var multiplexer = new MockExceptionMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(maxBatchSize),
            targetConcurrency,
            directExecutor());

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(batcher.getInFlightCount()).isEqualTo(0);
    var thrownSync = assertThrows(ExecutionException.class, r1::get);
    assertThat(thrownSync).hasCauseThat().isEqualTo(multiplexer.syncException);

    ListenableFuture<Response> r2 = batcher.submit(new Request(2));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(multiplexer.futures).hasSize(1);
    SettableFuture<List<Response>> f2 = multiplexer.futures.get(0);

    ListenableFuture<Response> r3 = batcher.submit(new Request(3));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(1);

    var asyncException = new RuntimeException("Async Failure");
    f2.setException(asyncException);

    var thrownAsync = assertThrows(ExecutionException.class, r2::get);
    assertThat(thrownAsync).hasCauseThat().isEqualTo(asyncException);

    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(0);
    assertThat(multiplexer.futures).hasSize(2);
    SettableFuture<List<Response>> f3 = multiplexer.futures.get(1);

    f3.set(respondTo(ImmutableList.of(new Request(3))));

    assertThat(r3.get()).isEqualTo(new Response(3));
    assertThat(batcher.getInFlightCount()).isEqualTo(0);
  }

  @Test
  public void testStalledWriterProgressGracefulDegradation() throws Exception {
    int targetConcurrency = 1;
    int maxBatchSize = 3;
    var multiplexer = new SettableMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(maxBatchSize),
            targetConcurrency,
            directExecutor());

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    BatchedOperations batch1 = multiplexer.queue.take();
    assertThat(batch1.requests()).containsExactly(new Request(1));

    // Get the active buffer which is currently empty (size=0, refs=1).
    Object bStalled = batcher.getActiveBufferForTesting();
    VarHandle sizeAndRefsVarHandle = EagerRequestBatcher.getBufferSizeAndRefsVarHandleForTesting();

    long initial = (long) sizeAndRefsVarHandle.get(bStalled);
    assertThat(EagerRequestBatcher.getBufferSizeAndRefsVarHandleForTesting()).isNotNull();

    // Simulates a "stalled writer" thread (Thread S).
    // We set size=1, refs=2.
    // This simulates that Thread S has successfully reserved slot 0 (incrementing size to 1 and
    // refs to 2), but it has got stalled *before* writing the element to the array, and *before*
    // decrementing refs back. So the buffer is locked in a state where it has 1 element (not yet
    // written) and 1 active writer (plus 1 barrier).
    long stalledValue = 0x0000_0001_0000_0002L;
    boolean success = sizeAndRefsVarHandle.compareAndSet(bStalled, initial, stalledValue);
    assertThat(success).isTrue();

    // Submit a normal request. It will see size=1, reserve slot 1, write elements[1], and decrement
    // refs. Buffer state becomes size=2, refs=2 (1 for stalled thread, 1 for barrier).
    ListenableFuture<Response> r2 = batcher.submit(new Request(2));
    assertThat(batcher.getInFlightCount())
        .isEqualTo(1); // Still 1 because r2 is not sent (over capacity)

    // Submit another request. It will see size=2 (which is maxBatchSize-1).
    // This triggers Rule R2 (Saturation Flush).
    // It detaches the buffer (allocating a new active buffer with size=0, refs=1), writes
    // Request(3) to elements[2], and decrements refs (size becomes 3, refs becomes 1). Since refs
    // is 1 (stalled thread still has a reference), it does NOT execute the buffer. The buffer
    // remains detached but unexecuted.
    ListenableFuture<Response> r3 = batcher.submit(new Request(3));
    assertThat(batcher.getInFlightCount())
        .isEqualTo(2); // inFlight count increases because we detached the buffer
    assertThat(batcher.getQueueSize()).isEqualTo(0); // Active buffer is now a fresh, empty one

    assertThat(multiplexer.queue).isEmpty(); // bStalled is NOT sent yet

    // Submit more requests to the NEW active buffer.
    // Since targetConcurrency=1 and inFlight=2, eager sending is disabled.
    // However, saturation flush (Rule R2) is still active.
    // Submitting 3 more requests will fill the new buffer (size 3) and trigger a saturation flush,
    // proving GRACEFUL DEGRADATION: the stalled thread in the first buffer does NOT block
    // progress of subsequent buffers.
    ListenableFuture<Response> r4 = batcher.submit(new Request(4));
    ListenableFuture<Response> r5 = batcher.submit(new Request(5));
    ListenableFuture<Response> r6 = batcher.submit(new Request(6));

    assertThat(batcher.getInFlightCount())
        .isEqualTo(3); // inFlight increases to 3 (batch1, bStalled, and batchNext)

    BatchedOperations batchNext = multiplexer.queue.take();
    assertThat(batchNext.requests())
        .containsExactly(new Request(4), new Request(5), new Request(6));

    // Complete the executed batches to clean up capacity.
    batch1.setSimpleResponses();
    batchNext.setSimpleResponses();
    r1.get();
    r4.get();
    r5.get();
    r6.get();

    assertThat(batcher.getInFlightCount()).isEqualTo(1); // Only bStalled remains in-flight

    // Now simulate the stalled thread resuming.
    // 1. It finally writes its element to slot 0.
    var stalledOp = new Operation<Request, Response>(new Request(99));
    Field elementsField = bStalled.getClass().getDeclaredField("elements");
    elementsField.setAccessible(true);
    Object[] elements = (Object[]) elementsField.get(bStalled);
    elements[0] = stalledOp;

    // 2. It decrements its reference count (size remains 3, refs becomes 0).
    long postDecrement = (long) sizeAndRefsVarHandle.getAndAdd(bStalled, -1L) - 1L;
    assertThat((int) postDecrement).isEqualTo(0); // We are the last reference!

    // 3. Since we are the last reference (refs=0), we trigger execution.
    // In production, the thread doing the decrement would call execute(buffer).
    // Here we invoke it manually via reflection to complete the simulation.
    Method executeMethod = batcher.getClass().getDeclaredMethod("execute", bStalled.getClass());
    executeMethod.setAccessible(true);
    executeMethod.invoke(batcher, bStalled);

    // Verify that the stalled batch is successfully executed and contains all elements in correct
    // order.
    BatchedOperations batchStalled = multiplexer.queue.take();
    assertThat(batchStalled.requests())
        .containsExactly(new Request(99), new Request(2), new Request(3));

    batchStalled.setSimpleResponses();
    stalledOp.get();
    assertThat(r2.get()).isEqualTo(new Response(2));
    assertThat(r3.get()).isEqualTo(new Response(3));

    assertThat(batcher.getInFlightCount()).isEqualTo(0);
  }

  @Test
  public void highConcurrency_exactlyOnceDispatch() throws Exception {
    int targetConcurrency = 8;
    int maxBatchSize = 50;
    int numThreads = 32;
    int submissionsPerThread = 5000;

    ExecutorService submitExecutor = Executors.newFixedThreadPool(numThreads);
    ExecutorService multiplexerExecutor = Executors.newCachedThreadPool();

    try {
      AsyncCountingMultiplexer multiplexer = new AsyncCountingMultiplexer(multiplexerExecutor);
      var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
      var batcher =
          new EagerRequestBatcher<>(
              strategy,
              new QueuePool<Request, Response>(maxBatchSize),
              targetConcurrency,
              directExecutor());

      CyclicBarrier barrier = new CyclicBarrier(numThreads);
      CountDownLatch latch = new CountDownLatch(numThreads);
      ConcurrentLinkedQueue<ListenableFuture<Response>> futures = new ConcurrentLinkedQueue<>();

      for (int i = 0; i < numThreads; i++) {
        final int threadId = i;
        submitExecutor.execute(
            () -> {
              try {
                barrier.await();
                for (int j = 0; j < submissionsPerThread; j++) {
                  futures.add(batcher.submit(new Request(threadId * 100000 + j)));
                }
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              } catch (Exception e) {
                // Ignore other exceptions
              } finally {
                latch.countDown();
              }
            });
      }

      latch.await();

      for (ListenableFuture<Response> future : futures) {
        future.get();
      }

      int totalExpected = numThreads * submissionsPerThread;
      assertThat(multiplexer.totalExecutions.get()).isEqualTo(totalExpected);
      assertThat(multiplexer.executionCounts).hasSize(totalExpected);

      for (AtomicInteger count : multiplexer.executionCounts.values()) {
        assertThat(count.get()).isEqualTo(1);
      }

      assertThat(batcher.getInFlightCount()).isEqualTo(0);
      assertThat(batcher.getQueueSize()).isEqualTo(0);

    } finally {
      submitExecutor.shutdown();
      multiplexerExecutor.shutdown();
    }
  }

  @Test
  public void capacityRecoveryEagerlySendsAccumulatedRequests() throws Exception {
    int targetConcurrency = 1;
    int maxBatchSize = 10;
    var multiplexer = new SettableMultiplexer();
    var strategy = RequestBatching.createBatchExecutionStrategy(multiplexer, directExecutor());
    var batcher =
        new EagerRequestBatcher<>(
            strategy,
            new QueuePool<Request, Response>(maxBatchSize),
            targetConcurrency,
            directExecutor());

    ListenableFuture<Response> r1 = batcher.submit(new Request(1));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(0);

    BatchedOperations batch1 = multiplexer.queue.take();
    assertThat(batch1.requests()).containsExactly(new Request(1));

    ListenableFuture<Response> r2 = batcher.submit(new Request(2));
    ListenableFuture<Response> r3 = batcher.submit(new Request(3));
    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(2);
    assertThat(multiplexer.queue).isEmpty();

    batch1.setSimpleResponses();
    r1.get();

    BatchedOperations batch2 = multiplexer.queue.take();
    assertThat(batch2.requests()).containsExactly(new Request(2), new Request(3));

    assertThat(batcher.getInFlightCount()).isEqualTo(1);
    assertThat(batcher.getQueueSize()).isEqualTo(0);

    batch2.setSimpleResponses();
    assertThat(r2.get()).isEqualTo(new Response(2));
    assertThat(r3.get()).isEqualTo(new Response(3));
    assertThat(batcher.getInFlightCount()).isEqualTo(0);
  }

  private static class LatencyMultiplexer implements Multiplexer<Request, Response> {
    private final int latencyMs;
    private final Executor executor;
    private final AtomicInteger activeExecutions = new AtomicInteger(0);
    private final AtomicInteger maxConcurrentExecutions = new AtomicInteger(0);
    private final ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();

    // Coordinated latch to force overlap
    private final CountDownLatch concurrencyLatch = new CountDownLatch(1);
    private final AtomicBoolean coordinationFailed = new AtomicBoolean(false);

    private LatencyMultiplexer(int latencyMs, Executor executor) {
      this.latencyMs = latencyMs;
      this.executor = executor;
    }

    @Override
    public ListenableFuture<List<Response>> execute(List<Request> requests) {
      SettableFuture<List<Response>> future = SettableFuture.create();
      executor.execute(
          () -> {
            int active = activeExecutions.incrementAndGet();
            maxConcurrentExecutions.accumulateAndGet(active, Math::max);

            if (active >= 2) {
              // Signal that we have reached concurrency >= 2
              concurrencyLatch.countDown();
            } else {
              // We are the first one. If we haven't failed coordination yet,
              // wait for a second one to join to guarantee overlap.
              if (!coordinationFailed.get()) {
                try {
                  // Wait up to 500ms. If no other task joins, we time out and proceed.
                  // This avoids deadlock if the batcher is broken and limits concurrency to 1.
                  if (!concurrencyLatch.await(500, MILLISECONDS)) {
                    // Mark coordination as failed so subsequent tasks don't keep waiting.
                    coordinationFailed.set(true);
                  }
                } catch (InterruptedException e) {
                  errors.add(e);
                  future.setException(e);
                  Thread.currentThread().interrupt();
                  activeExecutions.decrementAndGet();
                  return;
                }
              }
            }

            try {
              Thread.sleep(latencyMs);
            } catch (InterruptedException e) {
              errors.add(e);
              future.setException(e);
              Thread.currentThread().interrupt();
              activeExecutions.decrementAndGet();
              return;
            }
            activeExecutions.decrementAndGet();
            future.set(respondTo(requests));
          });
      return future;
    }
  }

  private static class MockExceptionMultiplexer implements Multiplexer<Request, Response> {
    private final List<SettableFuture<List<Response>>> futures = new ArrayList<>();
    private int executeCount = 0;
    private final RuntimeException syncException = new RuntimeException("Sync Failure");

    @Override
    public ListenableFuture<List<Response>> execute(List<Request> requests) {
      executeCount++;
      if (executeCount == 1) {
        throw syncException;
      }
      var future = SettableFuture.<List<Response>>create();
      futures.add(future);
      return future;
    }
  }

  private static class AsyncCountingMultiplexer implements Multiplexer<Request, Response> {
    private final ConcurrentHashMap<Request, AtomicInteger> executionCounts =
        new ConcurrentHashMap<>();
    private final AtomicInteger totalExecutions = new AtomicInteger(0);
    private final Executor executor;

    private AsyncCountingMultiplexer(Executor executor) {
      this.executor = executor;
    }

    @Override
    public ListenableFuture<List<Response>> execute(List<Request> requests) {
      SettableFuture<List<Response>> future = SettableFuture.create();
      executor.execute(
          () -> {
            totalExecutions.addAndGet(requests.size());
            for (Request request : requests) {
              executionCounts.computeIfAbsent(request, r -> new AtomicInteger(0)).incrementAndGet();
            }
            future.set(respondTo(requests));
          });
      return future;
    }
  }

  private record Request(int x) {}

  private record Response(int x) {}

  private static List<Response> respondTo(List<Request> requests) {
    return Lists.transform(requests, request -> new Response(request.x()));
  }
}
