// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.build.lib.worker.WorkerTestUtils.FakeSubprocess;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.lang.Thread.State;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for WorkerMultiplexer */
@RunWith(JUnit4.class)
public class WorkerMultiplexerTest {
  private FileSystem fileSystem;
  private Path logPath;

  @Before
  public void setUp() throws IOException {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    logPath = fileSystem.getPath("/tmp/logs4");
    logPath.createDirectoryAndParents();
  }

  @After
  public void tearDown() {
    WorkerMultiplexerManager.resetForTesting();
  }

  @Test
  public void testGetResponse_noOutstandingRequests() throws IOException, InterruptedException {
    WorkerKey workerKey = WorkerTestUtils.createWorkerKey(fileSystem, "test1", true, "fakeBinary");
    WorkerMultiplexer multiplexer = WorkerMultiplexerManager.getInstance(workerKey, logPath);

    PipedInputStream serverInputStream = new PipedInputStream();
    OutputStream workerOutputStream = new PipedOutputStream(serverInputStream);
    multiplexer.setProcessFactory(params -> new FakeSubprocess(serverInputStream));

    WorkRequest request1 = WorkRequest.newBuilder().setRequestId(1).build();
    WorkerProxy worker =
        new WorkerProxy(workerKey, 2, logPath, multiplexer, workerKey.getExecRoot());
    worker.prepareExecution(null, null, null);
    worker.putRequest(request1);
    WorkResponse response1 = WorkResponse.newBuilder().setRequestId(1).build();
    response1.writeDelimitedTo(workerOutputStream);
    workerOutputStream.flush();
    WorkResponse response = worker.getResponse(1);
    assertThat(response.getRequestId()).isEqualTo(1);
    // Can't get the same response twice - but the responseChecker is gone, so it just returns null
    assertThat(multiplexer.getResponse(1)).isNull();
    assertThat(multiplexer.noOutstandingRequests()).isTrue();
  }

  @Test
  public void testGetResponse_basicConcurrency()
      throws IOException, InterruptedException, ExecutionException {
    WorkerKey workerKey = WorkerTestUtils.createWorkerKey(fileSystem, "test2", true, "fakeBinary");
    WorkerMultiplexer multiplexer = WorkerMultiplexerManager.getInstance(workerKey, logPath);

    PipedInputStream serverInputStream = new PipedInputStream();
    OutputStream workerOutputStream = new PipedOutputStream(serverInputStream);
    multiplexer.setProcessFactory(params -> new FakeSubprocess(serverInputStream));

    WorkerProxy worker1 =
        new WorkerProxy(workerKey, 1, logPath, multiplexer, workerKey.getExecRoot());
    worker1.prepareExecution(null, null, null);
    WorkRequest request1 = WorkRequest.newBuilder().setRequestId(3).build();
    worker1.putRequest(request1);

    WorkerProxy worker2 =
        new WorkerProxy(workerKey, 2, logPath, multiplexer, workerKey.getExecRoot());
    worker2.prepareExecution(null, null, null);
    WorkRequest request2 = WorkRequest.newBuilder().setRequestId(42).build();
    worker2.putRequest(request2);

    Executor executor = Executors.newFixedThreadPool(2);
    Future<WorkResponse> response1 = Futures.submit(() -> worker1.getResponse(3), executor);
    Future<WorkResponse> response2 = Futures.submit(() -> worker2.getResponse(42), executor);

    WorkResponse fakedResponse1 = WorkResponse.newBuilder().setRequestId(3).build();
    WorkResponse fakedResponse2 = WorkResponse.newBuilder().setRequestId(42).build();
    // Responses can arrive out of order
    fakedResponse2.writeDelimitedTo(workerOutputStream);
    fakedResponse1.writeDelimitedTo(workerOutputStream);
    workerOutputStream.flush();

    assertThat(response1.get().getRequestId()).isEqualTo(3);
    assertThat(response2.get().getRequestId()).isEqualTo(42);
    assertThat(multiplexer.noOutstandingRequests()).isTrue();
  }

  @Test
  public void testGetResponse_slowMultiplexer()
      throws IOException, InterruptedException, ExecutionException {
    WorkerKey workerKey = WorkerTestUtils.createWorkerKey(fileSystem, "test3", true, "fakeBinary");
    WorkerMultiplexer multiplexer = WorkerMultiplexerManager.getInstance(workerKey, logPath);

    PipedInputStream serverInputStream = new PipedInputStream();
    OutputStream workerOutputStream = new PipedOutputStream(serverInputStream);
    multiplexer.setProcessFactory(params -> new FakeSubprocess(serverInputStream));

    WorkerProxy worker1 =
        new WorkerProxy(workerKey, 1, logPath, multiplexer, workerKey.getExecRoot());
    worker1.prepareExecution(null, null, null);
    WorkRequest request1 = WorkRequest.newBuilder().setRequestId(3).build();
    worker1.putRequest(request1);

    WorkerProxy worker2 =
        new WorkerProxy(workerKey, 2, logPath, multiplexer, workerKey.getExecRoot());
    worker2.prepareExecution(null, null, null);
    WorkRequest request2 = WorkRequest.newBuilder().setRequestId(42).build();
    worker2.putRequest(request2);

    Thread[] proxyThreads = new Thread[2];
    Executor executor = Executors.newFixedThreadPool(2);
    Future<WorkResponse> response1 =
        Futures.submit(
            () -> {
              synchronized (this) {
                proxyThreads[0] = Thread.currentThread();
              }

              return worker1.getResponse(3);
            },
            executor);
    Future<WorkResponse> response2 =
        Futures.submit(
            () -> {
              synchronized (this) {
                proxyThreads[1] = Thread.currentThread();
              }
              return worker2.getResponse(42);
            },
            executor);

    // Makes sure both workers are waiting for responses before the multiplexer processes anything.
    while (threadsAreNotWaiting(proxyThreads)) {
      Thread.sleep(1);
    }

    WorkResponse fakedResponse1 = WorkResponse.newBuilder().setRequestId(3).build();
    WorkResponse fakedResponse2 = WorkResponse.newBuilder().setRequestId(42).build();
    // Responses can arrive out of order
    fakedResponse2.writeDelimitedTo(workerOutputStream);
    fakedResponse1.writeDelimitedTo(workerOutputStream);
    workerOutputStream.flush();

    assertThat(response1.get().getRequestId()).isEqualTo(3);
    assertThat(response2.get().getRequestId()).isEqualTo(42);
    assertThat(multiplexer.noOutstandingRequests()).isTrue();
  }

  synchronized boolean threadsAreNotWaiting(Thread[] threads) {
    for (Thread thread : threads) {
      if (thread == null || thread.getState() != State.WAITING) {
        return true;
      }
    }
    return false;
  }

  @Test
  public void testGetResponse_slowProxy()
      throws IOException, InterruptedException, ExecutionException {
    WorkerKey workerKey = WorkerTestUtils.createWorkerKey(fileSystem, "test4", true, "fakeBinary");
    WorkerMultiplexer multiplexer = WorkerMultiplexerManager.getInstance(workerKey, logPath);

    PipedInputStream serverInputStream = new PipedInputStream();
    OutputStream workerOutputStream = new PipedOutputStream(serverInputStream);
    multiplexer.setProcessFactory(params -> new FakeSubprocess(serverInputStream));

    WorkerProxy worker1 =
        new WorkerProxy(workerKey, 1, logPath, multiplexer, workerKey.getExecRoot());
    worker1.prepareExecution(null, null, null);
    WorkRequest request1 = WorkRequest.newBuilder().setRequestId(3).build();
    worker1.putRequest(request1);

    WorkerProxy worker2 =
        new WorkerProxy(workerKey, 2, logPath, multiplexer, workerKey.getExecRoot());
    worker2.prepareExecution(null, null, null);
    WorkRequest request2 = WorkRequest.newBuilder().setRequestId(42).build();
    worker2.putRequest(request2);

    WorkResponse fakedResponse1 = WorkResponse.newBuilder().setRequestId(3).build();
    WorkResponse fakedResponse2 = WorkResponse.newBuilder().setRequestId(42).build();
    // Responses can arrive out of order, and before the workerproxies are ready to get them.
    fakedResponse2.writeDelimitedTo(workerOutputStream);
    fakedResponse1.writeDelimitedTo(workerOutputStream);
    workerOutputStream.flush();

    Executor executor = Executors.newFixedThreadPool(2);
    Future<WorkResponse> response1 = Futures.submit(() -> worker1.getResponse(3), executor);
    Future<WorkResponse> response2 = Futures.submit(() -> worker2.getResponse(42), executor);

    assertThat(response1.get().getRequestId()).isEqualTo(3);
    assertThat(response2.get().getRequestId()).isEqualTo(42);
    assertThat(multiplexer.noOutstandingRequests()).isTrue();
  }
}
