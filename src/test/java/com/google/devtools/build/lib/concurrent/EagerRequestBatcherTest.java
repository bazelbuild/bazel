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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.RequestBatching.CallbackMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.FutureMultiplexer;
import com.google.devtools.build.lib.concurrent.RequestBatching.Multiplexer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
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

  private record Request(int x) {}

  private record Response(int x) {}

  private static List<Response> respondTo(List<Request> requests) {
    return Lists.transform(requests, request -> new Response(request.x()));
  }
}
