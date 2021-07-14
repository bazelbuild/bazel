// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.worker.WorkRequestHandler.RequestInfo;
import com.google.devtools.build.lib.worker.WorkRequestHandler.WorkRequestCallback;
import com.google.devtools.build.lib.worker.WorkRequestHandler.WorkRequestHandlerBuilder;
import com.google.devtools.build.lib.worker.WorkRequestHandler.WorkerMessageProcessor;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for the WorkRequestHandler */
@RunWith(JUnit4.class)
public class WorkRequestHandlerTest {

  @Before
  public void init() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testNormalWorkRequest() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> 1,
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo(null));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testMultiplexWorkRequest() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> 0,
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).setRequestId(42).build();
    handler.respondToRequest(request, new RequestInfo(null));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getExitCode()).isEqualTo(0);
    assertThat(response.getOutput()).isEmpty();
  }

  @Test
  public void testOutput() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> {
              err.println("Failed!");
              return 1;
            },
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo(null));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).contains("Failed!");
  }

  @Test
  public void testException() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestHandler handler =
        new WorkRequestHandler(
            (args, err) -> {
              throw new RuntimeException("Exploded!");
            },
            new PrintStream(new ByteArrayOutputStream()),
            new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out));

    List<String> args = Arrays.asList("--sources", "A.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo(null));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(1);
    assertThat(response.getOutput()).startsWith("java.lang.RuntimeException: Exploded!");
  }

  @Test
  public void testCancelRequest_exactlyOneResponseSent() throws IOException, InterruptedException {
    boolean[] handlerCalled = new boolean[] {false};
    boolean[] cancelCalled = new boolean[] {false};
    PipedOutputStream src = new PipedOutputStream();
    PipedInputStream dest = new PipedInputStream();
    Semaphore done = new Semaphore(0);
    Semaphore finish = new Semaphore(0);
    List<String> failures = new ArrayList<>();

    WorkRequestHandler handler =
        new WorkRequestHandlerBuilder(
                (args, err) -> {
                  handlerCalled[0] = true;
                  err.println("Such work! Much progress! Wow!");
                  return 1;
                },
                new PrintStream(new ByteArrayOutputStream()),
                new LimitedWorkerMessageProcessor(
                    new ProtoWorkerMessageProcessor(
                        new PipedInputStream(src), new PipedOutputStream(dest)),
                    2))
            .setCancelCallback(
                (i, t) -> {
                  cancelCalled[0] = true;
                })
            .build();

    runRequestHandlerThread(done, handler, finish, failures);
    WorkRequest.newBuilder().setRequestId(42).build().writeDelimitedTo(src);
    WorkRequest.newBuilder().setRequestId(42).setCancel(true).build().writeDelimitedTo(src);
    WorkResponse response = WorkResponse.parseDelimitedFrom(dest);
    done.acquire();

    assertThat(handlerCalled[0] || cancelCalled[0]).isTrue();
    assertThat(response.getRequestId()).isEqualTo(42);
    if (response.getWasCancelled()) {
      assertThat(response.getOutput()).isEmpty();
      assertThat(response.getExitCode()).isEqualTo(0);
    } else {
      assertThat(response.getOutput()).isEqualTo("Such work! Much progress! Wow!\n");
      assertThat(response.getExitCode()).isEqualTo(1);
    }

    // Checks that nothing more was sent.
    assertThat(dest.available()).isEqualTo(0);
    finish.release();

    // Checks that there weren't other unexpected failures.
    assertThat(failures).isEmpty();
  }

  @Test
  public void testCancelRequest_sendsResponseWhenNotAlreadySent()
      throws IOException, InterruptedException {
    Semaphore waitForCancel = new Semaphore(0);
    Semaphore handlerCalled = new Semaphore(0);
    Semaphore cancelCalled = new Semaphore(0);
    PipedOutputStream src = new PipedOutputStream();
    PipedInputStream dest = new PipedInputStream();
    Semaphore done = new Semaphore(0);
    Semaphore finish = new Semaphore(0);
    List<String> failures = new ArrayList<>();

    // We force the regular handling to not finish until after we have read the cancel response,
    // to avoid flakiness.
    WorkRequestHandler handler =
        new WorkRequestHandlerBuilder(
                (args, err) -> {
                  // This handler waits until the main thread has sent a cancel request.
                  handlerCalled.release(2);
                  try {
                    waitForCancel.acquire();
                  } catch (InterruptedException e) {
                    failures.add("Unexpected interrupt waiting for cancel request");
                    e.printStackTrace();
                  }
                  return 0;
                },
                new PrintStream(new ByteArrayOutputStream()),
                new LimitedWorkerMessageProcessor(
                    new ProtoWorkerMessageProcessor(
                        new PipedInputStream(src), new PipedOutputStream(dest)),
                    2))
            .setCancelCallback(
                (i, t) -> {
                  cancelCalled.release();
                })
            .build();

    runRequestHandlerThread(done, handler, finish, failures);
    WorkRequest.newBuilder().setRequestId(42).build().writeDelimitedTo(src);
    // Make sure the handler is called before sending the cancel request, or we might process
    // the cancellation entirely before that.
    handlerCalled.acquire();
    WorkRequest.newBuilder().setRequestId(42).setCancel(true).build().writeDelimitedTo(src);
    WorkResponse response = WorkResponse.parseDelimitedFrom(dest);
    waitForCancel.release();
    // Give the other request a chance to process, so we can check that no other response is sent
    done.acquire();

    assertThat(handlerCalled.availablePermits()).isEqualTo(1); // Released 2, one was acquired
    assertThat(cancelCalled.availablePermits()).isEqualTo(1);
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getOutput()).isEmpty();
    assertThat(response.getWasCancelled()).isTrue();

    // Checks that nothing more was sent.
    assertThat(dest.available()).isEqualTo(0);
    src.close();
    finish.release();

    // Checks that there weren't other unexpected failures.
    assertThat(failures).isEmpty();
  }

  @Test
  public void testCancelRequest_noDoubleCancelResponse() throws IOException, InterruptedException {
    Semaphore waitForCancel = new Semaphore(0);
    Semaphore cancelCalled = new Semaphore(0);
    PipedOutputStream src = new PipedOutputStream();
    PipedInputStream dest = new PipedInputStream();
    Semaphore done = new Semaphore(0);
    Semaphore finish = new Semaphore(0);
    List<String> failures = new ArrayList<>();

    // We force the regular handling to not finish until after we have read the cancel response,
    // to avoid flakiness.
    WorkRequestHandler handler =
        new WorkRequestHandlerBuilder(
                (args, err) -> {
                  try {
                    waitForCancel.acquire();
                  } catch (InterruptedException e) {
                    failures.add("Unexpected interrupt waiting for cancel request");
                    e.printStackTrace();
                  }
                  return 0;
                },
                new PrintStream(new ByteArrayOutputStream()),
                new LimitedWorkerMessageProcessor(
                    new ProtoWorkerMessageProcessor(
                        new PipedInputStream(src), new PipedOutputStream(dest)),
                    3))
            .setCancelCallback(
                (i, t) -> {
                  cancelCalled.release();
                })
            .build();

    runRequestHandlerThread(done, handler, finish, failures);
    WorkRequest.newBuilder().setRequestId(42).build().writeDelimitedTo(src);
    WorkRequest.newBuilder().setRequestId(42).setCancel(true).build().writeDelimitedTo(src);
    WorkRequest.newBuilder().setRequestId(42).setCancel(true).build().writeDelimitedTo(src);
    WorkResponse response = WorkResponse.parseDelimitedFrom(dest);
    waitForCancel.release();
    done.acquire();

    assertThat(cancelCalled.availablePermits()).isLessThan(2);
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getOutput()).isEmpty();
    assertThat(response.getWasCancelled()).isTrue();

    // Checks that nothing more was sent.
    assertThat(dest.available()).isEqualTo(0);
    src.close();
    finish.release();

    // Checks that there weren't other unexpected failures.
    assertThat(failures).isEmpty();
  }

  @Test
  public void testCancelRequest_sendsNoResponseWhenAlreadySent()
      throws IOException, InterruptedException {
    Semaphore handlerCalled = new Semaphore(0);
    PipedOutputStream src = new PipedOutputStream();
    PipedInputStream dest = new PipedInputStream();
    Semaphore done = new Semaphore(0);
    Semaphore finish = new Semaphore(0);
    List<String> failures = new ArrayList<>();

    // We force the cancel request to not happen until after we have read the normal response,
    // to avoid flakiness.
    WorkRequestHandler handler =
        new WorkRequestHandlerBuilder(
                (args, err) -> {
                  handlerCalled.release();
                  err.println("Such work! Much progress! Wow!");
                  return 2;
                },
                new PrintStream(new ByteArrayOutputStream()),
                new LimitedWorkerMessageProcessor(
                    new ProtoWorkerMessageProcessor(
                        new PipedInputStream(src), new PipedOutputStream(dest)),
                    2))
            .setCancelCallback((i, t) -> {})
            .build();

    runRequestHandlerThread(done, handler, finish, failures);
    WorkRequest.newBuilder().setRequestId(42).build().writeDelimitedTo(src);
    WorkResponse response = WorkResponse.parseDelimitedFrom(dest);
    WorkRequest.newBuilder().setRequestId(42).setCancel(true).build().writeDelimitedTo(src);
    done.acquire();

    assertThat(response).isNotNull();

    assertThat(handlerCalled.availablePermits()).isEqualTo(1);
    assertThat(response.getRequestId()).isEqualTo(42);
    assertThat(response.getWasCancelled()).isFalse();
    assertThat(response.getExitCode()).isEqualTo(2);
    assertThat(response.getOutput()).isEqualTo("Such work! Much progress! Wow!\n");

    // Checks that nothing more was sent.
    assertThat(dest.available()).isEqualTo(0);
    src.close();
    finish.release();

    // Checks that there weren't other unexpected failures.
    assertThat(failures).isEmpty();
  }

  private void runRequestHandlerThread(
      Semaphore done, WorkRequestHandler handler, Semaphore finish, List<String> failures) {
    // This thread just makes sure the WorkRequestHandler does work asynchronously.
    new Thread(
            () -> {
              try {
                handler.processRequests();
                while (!handler.activeRequests.isEmpty()) {
                  Thread.sleep(1);
                }
                done.release();
                finish.acquire();
              } catch (IOException | InterruptedException e) {
                failures.add("Unexpected I/O error talking to worker thread");
                e.printStackTrace();
              }
            })
        .start();
  }

  /**
   * A wrapper around a WorkerMessageProcessor that stops after a given number of requests have been
   * read. It stops by making readWorkRequest() return null.
   */
  private static class LimitedWorkerMessageProcessor implements WorkerMessageProcessor {
    private final WorkerMessageProcessor delegate;
    private final int maxMessages;
    private int messages;

    public LimitedWorkerMessageProcessor(WorkerMessageProcessor delegate, int maxMessages) {
      this.delegate = delegate;
      this.maxMessages = maxMessages;
    }

    @Override
    public WorkRequest readWorkRequest() throws IOException {
      System.out.println("Handling request #" + messages);
      if (++messages > maxMessages) {
        return null;
      } else {
        return delegate.readWorkRequest();
      }
    }

    @Override
    public void writeWorkResponse(WorkResponse workResponse) throws IOException {
      delegate.writeWorkResponse(workResponse);
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }
  }

  @Test
  public void testWorkRequestHandler_withWorkRequestCallback() throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    WorkRequestCallback callback =
        new WorkRequestCallback((request, err) -> request.getArgumentsCount());
    WorkRequestHandler handler =
        new WorkRequestHandlerBuilder(
                callback,
                new PrintStream(new ByteArrayOutputStream()),
                new ProtoWorkerMessageProcessor(new ByteArrayInputStream(new byte[0]), out))
            .build();

    List<String> args = Arrays.asList("--sources", "B.java");
    WorkRequest request = WorkRequest.newBuilder().addAllArguments(args).build();
    handler.respondToRequest(request, new RequestInfo(null));

    WorkResponse response =
        WorkResponse.parseDelimitedFrom(new ByteArrayInputStream(out.toByteArray()));
    assertThat(response.getRequestId()).isEqualTo(0);
    assertThat(response.getExitCode()).isEqualTo(2);
    assertThat(response.getOutput()).isEmpty();
  }
}
