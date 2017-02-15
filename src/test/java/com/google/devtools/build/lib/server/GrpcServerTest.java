// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;
import static junit.framework.TestCase.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.base.Strings;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.server.CommandProtos.PingRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunRequest;
import com.google.devtools.build.lib.server.CommandProtos.RunResponse;
import com.google.devtools.build.lib.server.CommandServerGrpc.CommandServerBlockingStub;
import com.google.devtools.build.lib.server.GrpcServerImpl.StreamType;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.stub.ServerCallStreamObserver;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;

/**
 * Unit tests for the gRPC server.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class GrpcServerTest {

  /**
   * A little mock observer so that we can pretend we are talking to gRPC.
   */
  private static class MockObserver extends ServerCallStreamObserver<RunResponse> {
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean ready = new AtomicBoolean(true);
    private Runnable onCancelHandler;
    private Runnable onReadyHandler;
    private int sentMessages = 0;
    private int targetMessageCount = -1;
    private CountDownLatch targetMessageLatch = null;

    private void waitForMessages(int count, long l, TimeUnit unit) {
      synchronized (this) {
        Preconditions.checkState(targetMessageCount == -1);
        if (sentMessages >= count) {
          return;
        }

        targetMessageLatch = new CountDownLatch(1);
        this.targetMessageCount = count;
      }
      assertThat(Uninterruptibles.awaitUninterruptibly(targetMessageLatch, l, unit)).isTrue();
    }

    private synchronized int getMessageCount() {
      return sentMessages;
    }

    @Override
    public boolean isCancelled() {
      return true;
    }

    @Override
    public void setOnCancelHandler(Runnable onCancelHandler) {
      this.onCancelHandler = onCancelHandler;
    }

    @Override
    public void setCompression(String compression) {
    }

    @Override
    public boolean isReady() {
      return ready.get();
    }

    @Override
    public void setOnReadyHandler(Runnable onReadyHandler) {
      this.onReadyHandler = onReadyHandler;
    }

    @Override
    public void disableAutoInboundFlowControl() {
    }

    @Override
    public void request(int count) {
    }

    @Override
    public void setMessageCompression(boolean enable) {
    }

    @Override
    public void onNext(RunResponse value) {
      synchronized (this) {
        sentMessages += 1;
        if (sentMessages == targetMessageCount) {
          targetMessageLatch.countDown();
          targetMessageLatch = null;
          targetMessageCount = -1;
        }
      }

      if (cancelled.get()) {
        throw new StatusRuntimeException(Status.CANCELLED);
      }
    }

    @Override
    public void onError(Throwable t) {
    }

    @Override
    public void onCompleted() {
      if (cancelled.get()) {
        throw new StatusRuntimeException(Status.CANCELLED);
      }
    }
  }

  private ExecutorService executor;

  @Before
  public void setUp() {
    executor = Executors.newSingleThreadExecutor(new ThreadFactoryBuilder()
        .setNameFormat("grpc-server-test-%d")
        .setDaemon(true)
        .build());
  }

  @After
  public void tearDown() {
    executor.shutdownNow();
  }

  private RunResponse runResponse() {
    return RunResponse.newBuilder().setStandardError(ByteString.copyFromUtf8("hello")).build();
  }

  @Test
  public void testSendingSimpleMessage() {
    MockObserver observer = new MockObserver();
    GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);

    assertThat(sink.offer(runResponse())).isTrue();
    assertThat(sink.finish()).isFalse();
    observer.waitForMessages(1, 100, TimeUnit.MILLISECONDS);
  }

  @Test
  public void testSurvivesLateOnCancelHandler() {
    MockObserver observer = new MockObserver();
    GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);
    // First make the observer cancelled...
    observer.cancelled.set(true);

    // send a message...
    sink.offer(runResponse());

    // Then call the oncancel handler.
    observer.onCancelHandler.run();

    // Now the sink should still be responsive to finish events.
    assertThat(sink.finish()).isTrue();
    observer.waitForMessages(1, 100, TimeUnit.MILLISECONDS);
  }

  @Test
  public void testCancellationTurnsSinkIntoBlackHole() {
    MockObserver observer = new MockObserver();
    GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);

    observer.cancelled.set(true);
    observer.onCancelHandler.run();
    assertThat(sink.offer(runResponse())).isFalse();
    assertThat(sink.finish()).isTrue();
    assertThat(observer.getMessageCount()).isEqualTo(0);  // The message shouldn't have been sent.
  }

  @Test
  public void testInterruptsCommandThreadOnCancellation() throws Exception {
    final CountDownLatch safety = new CountDownLatch(1);
    final AtomicBoolean interrupted = new AtomicBoolean(false);
    TestThread victim = new TestThread() {
      @Override
      public void runTest() throws Exception {
        try {
          safety.await();
          fail("Test thread finished unexpectedly");
        } catch (InterruptedException e) {
          interrupted.set(true);
        }
      }
    };

    victim.setDaemon(true);
    victim.start();

    MockObserver observer = new MockObserver();
    GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);
    sink.setCommandThread(victim);
    observer.cancelled.set(true);
    observer.onCancelHandler.run();
    assertThat(sink.offer(runResponse())).isFalse();
    assertThat(sink.finish()).isTrue();
    safety.countDown();
    victim.joinAndAssertState(1000);
    assertThat(interrupted.get()).isTrue();
  }

  @Test
  public void testObeysReadySignal() throws Exception {
    MockObserver observer = new MockObserver();
    final GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);

    // First check if we can send a simple message
    assertThat(sink.offer(runResponse())).isTrue();
    observer.waitForMessages(1, 100, TimeUnit.MILLISECONDS);

    observer.ready.set(false);
    TestThread sender = new TestThread() {
      @Override
      public void runTest() {
        assertThat(sink.offer(runResponse())).isTrue();
      }
    };

    sender.setDaemon(true);
    sender.start();
    // Give the sender a little time to actually send the message
    Uninterruptibles.sleepUninterruptibly(100, TimeUnit.MILLISECONDS);
    assertThat(observer.getMessageCount()).isEqualTo(1);

    // Let the sink loose again
    observer.ready.set(true);
    observer.onReadyHandler.run();

    // Wait until the sender thread finishes and verify that the message was sent.
    assertThat(sink.finish()).isFalse();
    sender.joinAndAssertState(1000);
    observer.waitForMessages(2, 100, TimeUnit.MILLISECONDS);
  }

  @Test
  public void testDeadlockWhenDisconnectedWithQueueFull() throws Exception {
    MockObserver observer = new MockObserver();
    final GrpcServerImpl.GrpcSink sink = new GrpcServerImpl.GrpcSink(observer, executor);

    observer.ready.set(false);
    TestThread sender = new TestThread() {
      @Override
      public void runTest() {
        // Should return false due to the disconnect
        assertThat(sink.offer(runResponse())).isFalse();
      }
    };

    sender.setDaemon(true);
    sender.start();

    // Wait until the sink thread has processed the SEND message from #offer()
    while (sink.getReceivedEventCount() < 1) {
      Thread.sleep(200);
    }

    // Disconnect while there is an item pending
    observer.onCancelHandler.run();

    // Make sure that both the sink and the sender thread finish
    assertThat(sink.finish()).isTrue();
    sender.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void testRpcOutputStreamChunksLargeResponses() throws Exception {
    GrpcServerImpl.GrpcSink mockSink = mock(GrpcServerImpl.GrpcSink.class);
    @SuppressWarnings("resource")
    GrpcServerImpl.RpcOutputStream underTest = new GrpcServerImpl.RpcOutputStream(
        "command_id", "cookie", StreamType.STDOUT, mockSink);

    when(mockSink.offer(any(RunResponse.class))).thenReturn(true);

    String chunk1 = Strings.repeat("a", 8192);
    String chunk2 = Strings.repeat("b", 8192);
    String chunk3 = Strings.repeat("c", 1024);

    underTest.write((chunk1 + chunk2 + chunk3).getBytes(StandardCharsets.ISO_8859_1));
    InOrder inOrder = inOrder(mockSink);
    inOrder.verify(mockSink).offer(
        RunResponse.newBuilder()
            .setCommandId("command_id")
            .setCookie("cookie")
            .setStandardOutput(ByteString.copyFrom(chunk1.getBytes(StandardCharsets.ISO_8859_1)))
            .build());
    inOrder.verify(mockSink).offer(
        RunResponse.newBuilder()
            .setCommandId("command_id")
            .setCookie("cookie")
            .setStandardOutput(ByteString.copyFrom(chunk2.getBytes(StandardCharsets.ISO_8859_1)))
            .build());
    inOrder.verify(mockSink).offer(
        RunResponse.newBuilder()
            .setCommandId("command_id")
            .setCookie("cookie")
            .setStandardOutput(ByteString.copyFrom(chunk3.getBytes(StandardCharsets.ISO_8859_1)))
            .build());
  }

  private static class Server {
    private final Path workspaceDirectory;
    private final Path serverDirectory;
    private final CommandExecutor commandExecutor;
    private final GrpcServerImpl impl;
    private final TestThread serverThread;
    private final CommandServerBlockingStub clientStub;

    private Server() throws Exception {
      FileSystem fs = new InMemoryFileSystem(BlazeClock.instance());
      workspaceDirectory = fs.getPath("/workspace");
      workspaceDirectory.createDirectory();
      serverDirectory = fs.getPath("/server");
      serverDirectory.createDirectory();
      Path pidfile = fs.getPath("/server/server.pid.txt");
      FileSystemUtils.writeContentAsLatin1(pidfile, "1234");
      commandExecutor = mock(CommandExecutor.class);
      impl = new GrpcServerImpl(
          commandExecutor, BlazeClock.instance(), 0, workspaceDirectory, serverDirectory, 1000);
      serverThread = new TestThread() {
        @Override
        public void runTest() throws Exception {
          impl.serve();
        }
      };

      serverThread.start();
      CommandServerBlockingStub stubCandidate =  null;
      boolean ok = false;

      // Wait until the server starts up. Should be reasonably quick.
      for (int i = 0; i < 50; i++) {
        Thread.sleep(200);
        PingRequest request = PingRequest.newBuilder()
            .setCookie(impl.getRequestCookie())
            .build();
        if (impl.getAddress() == null) {
          continue;
        }

        if (stubCandidate == null) {
          stubCandidate = CommandServerGrpc.newBlockingStub(
            NettyChannelBuilder.forAddress(impl.getAddress())
                .usePlaintext(true)
                .build());
        }

        try {
          stubCandidate.ping(request);
          ok = true;
          break;
        } catch (StatusRuntimeException e) {
          continue;
        }
      }

      if (!ok) {
        throw new IllegalStateException("Server did not start up in time");
      }

      clientStub = stubCandidate;
    }

    private void finish() throws Exception {
      impl.signalShutdown();
      serverThread.joinAndAssertState(1000);
    }
  }

  @Test
  public void testRunCommand() throws Exception {
    Server server = new Server();
    when(server.commandExecutor.exec(any(List.class), any(OutErr.class), any(LockingMode.class),
        any(String.class), any(Long.class)))
        .thenReturn(42);

    RunRequest request = RunRequest.newBuilder()
        .setClientDescription("client")
        .setCookie(server.impl.getRequestCookie())
        .build();

    Iterator<RunResponse> result = server.clientStub.run(request);
    int exitCode = -1;
    while (result.hasNext()) {
      exitCode = result.next().getExitCode();
    }
    server.finish();
    assertThat(exitCode).isEqualTo(42);
  }

  @Test
  public void testIdleShutdownWhenWorkspaceDeleted() throws Exception {
    Server server = new Server();
    server.workspaceDirectory.delete();
    server.serverThread.joinAndAssertState(10000);  // We check the workspace dir every five seconds
  }
}
