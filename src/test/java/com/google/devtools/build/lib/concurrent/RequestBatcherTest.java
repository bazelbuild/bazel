// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.createPaddedBaseAddress;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.getAlignedAddress;
import static com.google.devtools.build.lib.concurrent.RequestBatcher.createBatchExecutionStrategy;
import static com.google.devtools.build.lib.concurrent.RequestBatcher.createCallbackBatchExecutionStrategy;
import static com.google.devtools.build.lib.concurrent.RequestBatcher.createPerResponseBatchExecutionStrategy;
import static java.util.concurrent.ForkJoinPool.commonPool;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.RequestBatcher.Operation;
import com.google.devtools.build.lib.concurrent.RequestBatcher.ResponseSink;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;

import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sun.misc.Unsafe;

@RunWith(JUnit4.class)
@SuppressWarnings("SunApi") // TODO: b/359688989 - clean this up
public final class RequestBatcherTest {
  private static final Cleaner cleaner = Cleaner.create();

  @Test
  public void simpleSubmit_executes() throws Exception {
    // This covers Step 1A in the documentation.
    var batcher =
        RequestBatcher.<Request, Response>create(
            createBatchExecutionStrategy(
                requests -> immediateFuture(respondTo(requests)), commonPool()),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1);
    ListenableFuture<Response> response = batcher.submit(new Request(1));
    assertThat(response.get()).isEqualTo(new Response(1));
  }

  @Test
  public void queueOverflow_sleeps() throws Exception {
    // This covers the overflow case of Step 1B in the documentation.
    int batchSize = 256;

    var multiplexer = new SettableMultiplexer();
    var batcher =
        RequestBatcher.<Request, Response>create(
            createBatchExecutionStrategy(multiplexer, commonPool()),
            /* maxBatchSize= */ batchSize - 1,
            /* maxConcurrentRequests= */ 1);
    ListenableFuture<Response> response0 = batcher.submit(new Request(0));
    BatchedOperations requestResponses0 = multiplexer.queue.take();
    // The first worker is busy until requestResponse0 is populated.

    var responses = new ArrayList<ListenableFuture<Response>>();
    // With the single available worker being busy, we can completely fill the queue.
    for (int i = 0; i < ConcurrentFifo.MAX_ELEMENTS; i++) {
      responses.add(batcher.submit(new Request(i + 1)));
    }

    // The next request triggers a queue overflow. Since this ends up blocking, we do it in another
    // thread.
    var overflowStarting = new CountDownLatch(1);
    var overflowAdded = new CountDownLatch(1);
    // A new thread needs must used here instead of commonPool because there are test environments
    // where commonPool has only a single thread.
    new Thread(
            () -> {
              overflowStarting.countDown();
              responses.add(batcher.submit(new Request(ConcurrentFifo.MAX_ELEMENTS + 1)));
              overflowAdded.countDown();
            })
        .start();

    // The following assertion will occasionally fail if the overflow submit above does not block.
    overflowStarting.await();
    assertThat(responses).hasSize(ConcurrentFifo.MAX_ELEMENTS);

    // Responding to the first batch enables the overflowing element to enter the queue.
    requestResponses0.setSimpleResponses();
    assertThat(response0.get()).isEqualTo(new Response(0));
    overflowAdded.await();
    assertThat(responses).hasSize(ConcurrentFifo.MAX_ELEMENTS + 1);

    // Responds to all remaining batches.
    int batchCount = responses.size() / batchSize;
    assertThat(responses).hasSize(batchCount * batchSize);

    for (int i = 0; i < batchCount; i++) {
      multiplexer.queue.take().setSimpleResponses();
    }

    for (int i = 0; i < ConcurrentFifo.MAX_ELEMENTS + 1; i++) {
      assertThat(responses.get(i).get()).isEqualTo(new Response(i + 1));
    }
  }

  @Test
  public void submitWithWorkersFull_enqueuesThenExecutes() throws Exception {
    // This covers Step 1B, Step 2A and Step 3 of the documentation.

    var multiplexer = new SettableMultiplexer();
    var batcher =
        RequestBatcher.<Request, Response>create(
            createBatchExecutionStrategy(multiplexer, commonPool()),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1);
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    BatchedOperations requestResponses1 = multiplexer.queue.take();

    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    // The first batch is not yet complete. The 2nd request waits in an internal queue. Ideally, we
    // could make a stronger assertion here, that the 2nd batch executes only after the first one is
    // done.
    assertThat(multiplexer.queue).isEmpty();

    requestResponses1.setSimpleResponses();

    // With the first batch done, the worker picks up the enqueued 2nd request and executes it.
    BatchedOperations requestResponses2 = multiplexer.queue.take();
    requestResponses2.setSimpleResponses();

    assertThat(response1.get()).isEqualTo(new Response(1));
    assertThat(response2.get()).isEqualTo(new Response(2));
  }

  // TODO: b/386384684 - remove Unsafe usage
  @Test
  public void concurrentWorkCompletion_startsNewWorker() throws Exception {
    // This covers Step 1B and Step 2B of the documentation.

    // This test uses fakes to achieve the narrow set of conditions needed to reach this code path.
    long baseAddress = createPaddedBaseAddress(4);

    var queueDrainingExecutor = new FakeExecutor();
    var fifo =
        new FakeConcurrentFifo(
            getAlignedAddress(baseAddress, /* offset= */ 1),
            getAlignedAddress(baseAddress, /* offset= */ 2),
            getAlignedAddress(baseAddress, /* offset= */ 3));
    long countersAddress = getAlignedAddress(baseAddress, /* offset= */ 0);
    var batcher =
        new RequestBatcher<Request, Response>(
            /* queueDrainingExecutor= */ queueDrainingExecutor,
            createBatchExecutionStrategy(
                requests -> immediateFuture(respondTo(requests)), commonPool()),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1,
            countersAddress,
            fifo);
    cleaner.register(batcher, new AddressFreer(baseAddress));

    // Submits a request. This starts a worker to run the batch, but it gets blocked on
    // `queueDrainingExecutor` and can't continue.
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));

    // Submits a 2nd request. This request observes that there are enough active workers so it tries
    // to enqueue an element. It gets blocked at the queue.
    @SuppressWarnings("FutureReturnValueIgnored")
    var response2 = SettableFuture.<ListenableFuture<Response>>create();
    commonPool().execute(() -> response2.set(batcher.submit(new Request(2))));
    // Waits until the 2nd request starts enqueuing.
    fifo.tryAppendTokens.acquireUninterruptibly();

    // Allows the 1st worker to continue. This calls an enqueued `continueToNextBatchOrBecomeIdle`
    // invocation that will cause the 1st worker to go idle.
    queueDrainingExecutor.queue.take().run();

    assertThat(response1.get()).isEqualTo(new Response(1));

    // Verifies that there's absolutely nothing inflight in the batcher.
    assertThat(UNSAFE.getIntVolatile(null, countersAddress)).isEqualTo(0);

    // Allows the 2nd request to enqueue and complete processing.
    fifo.appendPermits.release();
    queueDrainingExecutor.queue.take().run();

    assertThat(response2.get().get()).isEqualTo(new Response(2));
  }

  @Test
  public void randomRaces_executeCorrectly() throws Exception {
    var batcher =
        RequestBatcher.<Request, Response>create(
            createBatchExecutionStrategy(
                requests -> immediateFuture(respondTo(requests)), commonPool()),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 4);

    var results = new ConcurrentLinkedQueue<ListenableFuture<Void>>();
    final int requestCount = 4_000_000;
    var allStarted = new CountDownLatch(requestCount);
    for (int i = 0; i < requestCount; i++) {
      final int iForCapture = i;
      commonPool()
          .execute(
              () -> {
                results.add(
                    Futures.transformAsync(
                        batcher.submit(new Request(iForCapture)),
                        response ->
                            response.x() == iForCapture
                                ? immediateVoidFuture()
                                : immediateFailedFuture(
                                    new AssertionError(
                                        String.format(
                                            "expected %d got %s", iForCapture, response))),
                        directExecutor()));
                allStarted.countDown();
              });
    }
    allStarted.await();
    // Throws ExecutionException if there are any errors.
    var unused = Futures.whenAllSucceed(results).call(() -> null, directExecutor()).get();
  }

  private static class FakeExecutor implements Executor {
    private final LinkedBlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();

    @Override
    public void execute(Runnable runnable) {
      queue.add(runnable);
    }
  }

  @Test
  public void perResponseMultiplexer_simpleSubmit_executes() throws Exception {
    var batcher =
        RequestBatcher.<Request, Response>create(
            createPerResponseBatchExecutionStrategy(
                (requests, sinks) -> {
                  assertThat(requests).hasSize(1);
                  assertThat(sinks).hasSize(1);
                  sinks.get(0).acceptFutureResponse(immediateFuture(new Response(1)));
                }),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1);

    ListenableFuture<Response> response = batcher.submit(new Request(1));

    assertThat(response.get()).isEqualTo(new Response(1));
  }

  @Test
  public void perResponseMultiplexer_batching_succeeds() throws Exception {
    var multiplexer = new PerResponseSettableMultiplexer();
    var batcher =
        RequestBatcher.<Request, Response>create(
            createPerResponseBatchExecutionStrategy(multiplexer),
            /* maxBatchSize= */ 1, // actual batch size is 2
            /* maxConcurrentRequests= */ 1);

    // Block the first worker
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    BatchedPerResponseRequests batch1 = multiplexer.queue.take();
    assertThat(batch1.requests()).hasSize(1);

    // These will get enqueued because the worker is busy
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    ListenableFuture<Response> response3 = batcher.submit(new Request(3));
    ListenableFuture<Response> response4 = batcher.submit(new Request(4));

    // Unblock the first worker, allowing the next batch to be processed.
    batch1.setSimpleSuccessResponses();
    assertThat(response1.get()).isEqualTo(new Response(1));

    // The next batch should contain requests 2 and 3.
    BatchedPerResponseRequests batch2 = multiplexer.queue.take();
    assertThat(batch2.requests()).hasSize(2);
    assertThat(batch2.requests().stream().map(Request::x)).containsExactly(2, 3).inOrder();
    batch2.setSimpleSuccessResponses();

    // The final batch should contain request 4.
    BatchedPerResponseRequests batch3 = multiplexer.queue.take();
    assertThat(batch3.requests()).hasSize(1);
    assertThat(batch3.requests().get(0).x()).isEqualTo(4);
    batch3.setSimpleSuccessResponses();

    // Verify all responses
    assertThat(response2.get()).isEqualTo(new Response(2));
    assertThat(response3.get()).isEqualTo(new Response(3));
    assertThat(response4.get()).isEqualTo(new Response(4));
  }

  @Test
  public void perResponseMultiplexer_individualResponseFailure_isIsolated() throws Exception {
    var multiplexer = new PerResponseSettableMultiplexer();
    var batcher =
        RequestBatcher.<Request, Response>create(
            createPerResponseBatchExecutionStrategy(multiplexer),
            /* maxBatchSize= */ 1, // actual batch size is 2
            /* maxConcurrentRequests= */ 1);

    // Blocks the first worker to allow a batch to form.
    ListenableFuture<Response> response0 = batcher.submit(new Request(0));
    BatchedPerResponseRequests batch0 = multiplexer.queue.take();
    assertThat(batch0.requests()).hasSize(1);

    // These two will be enqueued and batched together because the worker is busy.
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));

    // Unblocks the first worker, allowing the next batch to be processed.
    batch0.setSimpleSuccessResponses();
    assertThat(response0.get()).isEqualTo(new Response(0));

    // Wait for the batch to be sent to the multiplexer.
    BatchedPerResponseRequests batch = multiplexer.queue.take();
    assertThat(batch.requests()).hasSize(2);

    // Fulfill the first future with success and the second with failure.
    var failure = new IllegalStateException("Individual failure");
    batch.settableFutures().get(0).set(new Response(1));
    batch.settableFutures().get(1).setException(failure);

    // Assert the first future succeeded and the second failed correctly.
    assertThat(response1.get()).isEqualTo(new Response(1));
    var thrown = assertThrows(ExecutionException.class, response2::get);
    assertThat(thrown).hasCauseThat().isEqualTo(failure);
  }

  @Test
  public void perResponseMultiplexer_missingFuture_throwsIllegalState() throws Exception {
    var futureResponses = new LinkedBlockingQueue<SettableFuture<Response>>();
    var multiplexer =
        new RequestBatcher.PerResponseMultiplexer<Request, Response>() {
          @Override
          public void execute(
              List<Request> requests,
              ImmutableList<? extends RequestBatcher.FutureResponseSink<Response>> sinks) {
            // Faulty implementation: only sets the first future in the batch, and "forgets" to set
            // the rest.
            var futureResponse = SettableFuture.<Response>create();
            futureResponses.add(futureResponse);
            sinks.get(0).acceptFutureResponse(futureResponse);
          }
        };

    var batcher =
        RequestBatcher.<Request, Response>create(
            createPerResponseBatchExecutionStrategy(multiplexer),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1);

    // Blocks the first worker to allow a batch to form.
    ListenableFuture<Response> response0 = batcher.submit(new Request(0));
    SettableFuture<Response> responseSetter0 = futureResponses.take();

    // These two will be batched together.
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));

    // Unblocks the first worker, allowing the next batch to be processed.
    responseSetter0.set(new Response(0));
    assertThat(response0.get()).isEqualTo(new Response(0));

    // The multiplexer will set the future for request 1, but not for 2.
    futureResponses.take().set(new Response(1));
    assertThat(response1.get()).isEqualTo(new Response(1));
    assertThat(futureResponses).isEmpty();

    var thrown = assertThrows(ExecutionException.class, response2::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(thrown)
        .hasCauseThat()
        .hasMessageThat()
        .contains("Future for Request[x=2] is unexpectedly not set");
  }

  @Test
  public void cancelledRequest_doesNotCrash() throws Exception {
    var uncaughtException = new AtomicReference<Throwable>();
    var crashDetectingExecutor =
        new Executor() {
          @Override
          public void execute(Runnable command) {
            try {
              command.run();
            } catch (Throwable t) {
              uncaughtException.set(t);
            }
          }
        };
    var multiplexer = new SettableMultiplexer();
    var batcher =
        RequestBatcher.<Request, Response>create(
            createBatchExecutionStrategy(multiplexer, crashDetectingExecutor),
            /* maxBatchSize= */ 255,
            /* maxConcurrentRequests= */ 1);

    ListenableFuture<Response> response = batcher.submit(new Request(1));
    response.cancel(true);

    BatchedOperations requestResponses = multiplexer.queue.take();
    requestResponses.setSimpleResponses();

    assertThat(response.isCancelled()).isTrue();
    assertThat(uncaughtException.get()).isNull();
  }

  @Test
  public void callbackMultiplexer_allSucceed() throws Exception {
    var events = new LinkedBlockingQueue<BatcherEvent>();
    RequestBatcher<Request, Response> batcher = createCallbackMultiplexerBatcher(events);

    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    // A batch begins executing immediately with response1.
    var firstBatch = (BatchOperation) events.poll();

    // The next 2 requests are enqueued.
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    ListenableFuture<Response> response3 = batcher.submit(new Request(3));

    // No other batches have started and no callbacks have been called.
    assertThat(events.poll()).isNull();

    firstBatch.defaultReplyAll();
    assertThat(response1.get()).isEqualTo(new Response(1));

    // The done callback is always called first.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);

    var secondBatch = (BatchOperation) events.take();
    secondBatch.defaultReplyAll();
    assertThat(response2.get()).isEqualTo(new Response(2));
    assertThat(response3.get()).isEqualTo(new Response(3));

    // The done callback is called after the 2nd batch completes.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);
  }

  private sealed interface BatcherEvent {}

  private enum DoneCallbackCalled implements BatcherEvent {
    INSTANCE
  }

  private record BatchOperation(
      List<Request> requests, ImmutableList<? extends ResponseSink<Response>> sinks)
      implements BatcherEvent {
    private void defaultReplyAll() {
      for (int i = 0; i < requests.size(); i++) {
        sinks.get(i).acceptResponse(new Response(requests.get(i).x()));
      }
    }

    private void failAll(Throwable t) {
      for (ResponseSink<Response> sink : sinks) {
        sink.acceptFailure(t);
      }
    }
  }

  private static RequestBatcher<Request, Response> createCallbackMultiplexerBatcher(
      LinkedBlockingQueue<BatcherEvent> events) {
    var multiplexer =
        new RequestBatcher.CallbackMultiplexer<Request, Response>() {
          @Override
          public Runnable execute(
              List<Request> requests, ImmutableList<? extends ResponseSink<Response>> sinks) {
            assertThat(requests).hasSize(sinks.size());
            events.offer(new BatchOperation(requests, sinks));
            return () -> events.offer(DoneCallbackCalled.INSTANCE);
          }
        };

    return RequestBatcher.<Request, Response>create(
        createCallbackBatchExecutionStrategy(multiplexer),
        /* maxBatchSize= */ 1,
        /* maxConcurrentRequests= */ 1);
  }

  @Test
  public void callbackMultiplexer_allFail() throws Exception {
    var events = new LinkedBlockingQueue<BatcherEvent>();
    RequestBatcher<Request, Response> batcher = createCallbackMultiplexerBatcher(events);

    var failure = new RuntimeException("Test Failure");

    ListenableFuture<Response> response1 = batcher.submit(new Request(1));

    // A batch begins executing immediately with response1.
    var firstBatch = (BatchOperation) events.poll();

    // The next 2 requests are enqueued.
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    ListenableFuture<Response> response3 = batcher.submit(new Request(3));

    assertThat(events.poll()).isNull(); // No new events have occurred.

    firstBatch.failAll(failure);
    var e1 = assertThrows(ExecutionException.class, response1::get);
    assertThat(e1).hasCauseThat().isEqualTo(failure);

    // The done callback is always called before the next batch starts.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);

    var secondBatch = (BatchOperation) events.take();
    secondBatch.failAll(failure);

    var e2 = assertThrows(ExecutionException.class, response2::get);
    assertThat(e2).hasCauseThat().isEqualTo(failure);
    var e3 = assertThrows(ExecutionException.class, response3::get);
    assertThat(e3).hasCauseThat().isEqualTo(failure);

    // The done callback is called after the 2nd batch completes.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);
  }

  @Test
  public void callbackMultiplexer_mixedSuccessFailure() throws Exception {
    var events = new LinkedBlockingQueue<BatcherEvent>();
    RequestBatcher<Request, Response> batcher = createCallbackMultiplexerBatcher(events);

    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    // A batch begins executing immediately with response1.
    var firstBatch = (BatchOperation) events.poll();

    // The next 2 requests are enqueued.
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    ListenableFuture<Response> response3 = batcher.submit(new Request(3));

    assertThat(events.poll()).isNull(); // No new events have occurred.

    firstBatch.defaultReplyAll();
    assertThat(response1.get()).isEqualTo(new Response(1));

    // The done callback is always called before the next batch starts.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);

    var secondBatch = (BatchOperation) events.take();

    var failure = new IllegalArgumentException("Bad Request");

    assertThat(secondBatch.sinks).hasSize(2);

    secondBatch.sinks.get(0).acceptResponse(new Response(secondBatch.requests.get(0).x()));
    secondBatch.sinks.get(1).acceptFailure(failure);

    assertThat(response2.get()).isEqualTo(new Response(2));
    var e3 = assertThrows(ExecutionException.class, response3::get);
    assertThat(e3).hasCauseThat().isEqualTo(failure);

    // The done callback is called after the 2nd batch completes.
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);
  }

  @Test
  public void callbackMultiplexer_nullResponse() throws Exception {
    var events = new LinkedBlockingQueue<BatcherEvent>();
    RequestBatcher<Request, Response> batcher = createCallbackMultiplexerBatcher(events);

    ListenableFuture<Response> response1 = batcher.submit(new Request(1));

    var batch = (BatchOperation) events.poll();
    batch.sinks.get(0).acceptResponse(null);

    assertThat(response1.get()).isNull();

    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);
  }

  @Test
  public void callbackMultiplexer_batching() throws Exception {
    var events = new LinkedBlockingQueue<BatcherEvent>();
    RequestBatcher<Request, Response> batcher = createCallbackMultiplexerBatcher(events);

    // These should form three batches: (1), (2, 3), (4, 5)
    ListenableFuture<Response> response1 = batcher.submit(new Request(1));
    ListenableFuture<Response> response2 = batcher.submit(new Request(2));
    ListenableFuture<Response> response3 = batcher.submit(new Request(3));
    ListenableFuture<Response> response4 = batcher.submit(new Request(4));
    ListenableFuture<Response> response5 = batcher.submit(new Request(5));

    var batch1 = (BatchOperation) events.take();

    assertThat(batch1.requests()).containsExactly(new Request(1));
    batch1.defaultReplyAll();

    assertThat(response1.get()).isEqualTo(new Response(1));
    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);

    var batch2 = (BatchOperation) events.take();
    assertThat(batch2.requests()).containsExactly(new Request(2), new Request(3)).inOrder();
    batch2.defaultReplyAll();

    assertThat(response2.get()).isEqualTo(new Response(2));
    assertThat(response3.get()).isEqualTo(new Response(3));

    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);

    var batch3 = (BatchOperation) events.take();
    assertThat(batch3.requests()).containsExactly(new Request(4), new Request(5)).inOrder();
    batch3.defaultReplyAll();

    assertThat(response4.get()).isEqualTo(new Response(4));
    assertThat(response5.get()).isEqualTo(new Response(5));

    assertThat(events.take()).isEqualTo(DoneCallbackCalled.INSTANCE);
  }

  private static class FakeConcurrentFifo extends ConcurrentFifo<Operation<Request, Response>> {
    private final ConcurrentLinkedQueue<Operation<Request, Response>> queue =
        new ConcurrentLinkedQueue<>();

    private final Semaphore tryAppendTokens = new Semaphore(0);
    private final Semaphore appendPermits = new Semaphore(0);

    private FakeConcurrentFifo(long sizeAddress, long appendIndexAddress, long takeIndexAddress) {
      super(Operation.class, sizeAddress, appendIndexAddress, takeIndexAddress);
    }

    @Override
    boolean tryAppend(Operation<Request, Response> task) {
      tryAppendTokens.release();
      appendPermits.acquireUninterruptibly();
      queue.add(task);
      return true;
    }

    @Override
    Operation<Request, Response> take() {
      return queue.poll();
    }
  }

  private static class SettableMultiplexer
      implements RequestBatcher.Multiplexer<Request, Response> {
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

  private static class PerResponseSettableMultiplexer
      implements RequestBatcher.PerResponseMultiplexer<Request, Response> {
    private final LinkedBlockingQueue<BatchedPerResponseRequests> queue =
        new LinkedBlockingQueue<>();

    @Override
    public void execute(
        List<Request> requests,
        ImmutableList<? extends RequestBatcher.FutureResponseSink<Response>> sinks) {
      assertThat(requests).hasSize(sinks.size());

      List<SettableFuture<Response>> settableFutures = new ArrayList<>();
      for (int i = 0; i < sinks.size(); i++) {
        SettableFuture<Response> settableFuture = SettableFuture.create();
        settableFutures.add(settableFuture);
        // This links the batcher's internal future to our controllable future.
        sinks.get(i).acceptFutureResponse(settableFuture);
      }
      queue.add(new BatchedPerResponseRequests(requests, settableFutures));
    }
  }

  private record BatchedPerResponseRequests(
      List<Request> requests, List<SettableFuture<Response>> settableFutures) {

    private void setSimpleSuccessResponses() {
      assertThat(requests).hasSize(settableFutures.size());
      for (int i = 0; i < requests.size(); i++) {
        settableFutures.get(i).set(new Response(requests.get(i).x()));
      }
    }
  }

  private static final Unsafe UNSAFE = UnsafeProvider.unsafe();
}
