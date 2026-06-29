// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.ExecutionGrpc.ExecutionImplBase;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.rpc.Code;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class GrpcRemoteExecutorTest extends GrpcRemoteExecutorTestBase {
  private ListeningScheduledExecutorService retryService;

  @Override
  public void setUp() throws Exception {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
    super.setUp();
  }

  @Override
  protected RemoteExecutionClient createExecutionService(ReferenceCountedChannel channel) {
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions),
            RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
            retryService);

    return new GrpcRemoteExecutor(channel, CallCredentialsProvider.NO_CREDENTIALS, retrier);
  }

  @Override
  public void tearDown() throws Exception {
    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
    super.tearDown();
  }

  @Test
  public void executeRemotely_operationWithoutResult_crashes() {
    executionService.whenExecute(DUMMY_REQUEST).thenDone();

    assertThrows(
        IllegalStateException.class,
        () -> executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP));
    // Shouldn't retry in this case
    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_responseWithoutResult_crashes() {
    executionService.whenExecute(DUMMY_REQUEST).thenDone(ExecuteResponse.getDefaultInstance());

    assertThrows(
        IllegalStateException.class,
        () -> executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP));

    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_retryWaitExecutionWhenUnauthenticated()
      throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().finish();
    executionService.whenWaitExecution(DUMMY_REQUEST).thenError(Code.UNAUTHENTICATED);
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(2);
    assertThat(executionService.getWaitTimes()).isEqualTo(1);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_holdsChannelLeaseUntilExecuteStreamCompletes() throws Exception {
    BlockingExecutionService blockingExecutionService = new BlockingExecutionService();
    ExecutorService serverExecutor = Executors.newCachedThreadPool();
    String fakeServerName = "blocking fake server for " + getClass() + "#" + System.nanoTime();
    Server server =
        InProcessServerBuilder.forName(fakeServerName)
            .addService(blockingExecutionService)
            .executor(serverExecutor)
            .build()
            .start();
    AtomicInteger channelCreations = new AtomicInteger();
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(
            new ChannelConnectionWithServerCapabilitiesFactory() {
              @Override
              public Single<ChannelConnectionWithServerCapabilities> create() {
                channelCreations.incrementAndGet();
                ManagedChannel ch =
                    InProcessChannelBuilder.forName(fakeServerName)
                        .intercept(TracingMetadataUtils.newExecHeadersInterceptor(remoteOptions))
                        .build();
                ServerCapabilities caps =
                    ServerCapabilities.newBuilder()
                        .setExecutionCapabilities(
                            ExecutionCapabilities.newBuilder().setExecEnabled(true).build())
                        .build();
                return Single.just(
                    new ChannelConnectionWithServerCapabilities(ch, Single.just(caps)));
              }

              @Override
              public int maxConcurrency() {
                return 1;
              }
            });
    RemoteExecutionClient executor = createExecutionService(channel);
    ExecutorService clientExecutor = Executors.newFixedThreadPool(2);

    try {
      Future<ExecuteResponse> first =
          clientExecutor.submit(
              () -> executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP));
      assertThat(blockingExecutionService.firstExecuteStarted.await(1, SECONDS)).isTrue();
      assertThat(channelCreations.get()).isEqualTo(1);

      Future<ExecuteResponse> second =
          clientExecutor.submit(
              () -> executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP));
      assertThat(blockingExecutionService.secondExecuteStarted.await(1, SECONDS)).isTrue();

      assertThat(channelCreations.get()).isEqualTo(2);

      blockingExecutionService.completeFirstExecute.countDown();
      assertThat(second.get(1, SECONDS)).isEqualTo(DUMMY_RESPONSE);
      assertThat(first.get(1, SECONDS)).isEqualTo(DUMMY_RESPONSE);
    } finally {
      blockingExecutionService.completeFirstExecute.countDown();
      clientExecutor.shutdownNow();
      executor.close();
      server.shutdownNow();
      server.awaitTermination();
      serverExecutor.shutdownNow();
    }
  }

  private static final class BlockingExecutionService extends ExecutionImplBase {
    private final AtomicInteger executeTimes = new AtomicInteger();
    private final CountDownLatch firstExecuteStarted = new CountDownLatch(1);
    private final CountDownLatch secondExecuteStarted = new CountDownLatch(1);
    private final CountDownLatch completeFirstExecute = new CountDownLatch(1);

    @Override
    public void execute(ExecuteRequest request, StreamObserver<Operation> responseObserver) {
      int executeNumber = executeTimes.incrementAndGet();
      if (executeNumber == 1) {
        responseObserver.onNext(FakeExecutionService.ackOperation(request));
        firstExecuteStarted.countDown();
        try {
          completeFirstExecute.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          responseObserver.onError(Status.CANCELLED.asRuntimeException());
          return;
        }
      } else if (executeNumber == 2) {
        secondExecuteStarted.countDown();
      }

      responseObserver.onNext(FakeExecutionService.doneOperation(request, DUMMY_RESPONSE));
      responseObserver.onCompleted();
    }
  }
}
