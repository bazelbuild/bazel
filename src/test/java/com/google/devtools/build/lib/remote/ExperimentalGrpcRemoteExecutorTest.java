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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.common.options.Options;
import com.google.longrunning.Operation;
import com.google.rpc.Code;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExperimentalGrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class ExperimentalGrpcRemoteExecutorTest {

  private RemoteActionExecutionContext context;
  private FakeExecutionService executionService;
  private RemoteOptions remoteOptions;
  private Server fakeServer;
  private ListeningScheduledExecutorService retryService;
  ExperimentalGrpcRemoteExecutor executor;

  private static final int MAX_RETRY_ATTEMPTS = 5;

  private static final OutputFile DUMMY_OUTPUT =
      OutputFile.newBuilder()
          .setPath("dummy.txt")
          .setDigest(
              Digest.newBuilder()
                  .setHash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                  .setSizeBytes(0)
                  .build())
          .build();

  private static final ExecuteRequest DUMMY_REQUEST =
      ExecuteRequest.newBuilder()
          .setInstanceName("dummy")
          .setActionDigest(
              Digest.newBuilder()
                  .setHash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                  .setSizeBytes(0)
                  .build())
          .build();

  private static final ExecuteResponse DUMMY_RESPONSE =
      ExecuteResponse.newBuilder()
          .setResult(ActionResult.newBuilder().addOutputFiles(DUMMY_OUTPUT).build())
          .build();

  @Before
  public final void setUp() throws Exception {
    context = RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance());

    executionService = new FakeExecutionService();

    String fakeServerName = "fake server for " + getClass();
    // Use a mutable service registry for later registering the service impl for each test case.
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .addService(executionService)
            .directExecutor()
            .build()
            .start();

    remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteMaxRetryAttempts = MAX_RETRY_ATTEMPTS;

    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions),
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryService);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(
            new ChannelConnectionFactory() {
              @Override
              public Single<? extends ChannelConnection> create() {
                ManagedChannel ch =
                    InProcessChannelBuilder.forName(fakeServerName)
                        .intercept(TracingMetadataUtils.newExecHeadersInterceptor(remoteOptions))
                        .directExecutor()
                        .build();
                return Single.just(new ChannelConnection(ch));
              }

              @Override
              public int maxConcurrency() {
                return 100;
              }
            });

    ServerCapabilities caps =
        ServerCapabilities.newBuilder()
            .setExecutionCapabilities(
                ExecutionCapabilities.newBuilder().setExecEnabled(true).build())
            .build();

    executor =
        new ExperimentalGrpcRemoteExecutor(
            caps, remoteOptions, channel, CallCredentialsProvider.NO_CREDENTIALS, retrier);
  }

  @After
  public void tearDown() throws Exception {
    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);

    fakeServer.shutdownNow();
    fakeServer.awaitTermination();

    executor.close();
  }

  @Test
  public void executeRemotely_smoke() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(response).isEqualTo(DUMMY_RESPONSE);
    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_errorInOperation_retryExecute() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenError(new RuntimeException("Unavailable"));
    executionService.whenExecute(DUMMY_REQUEST).thenError(Code.UNAVAILABLE);
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(3);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_errorInResponse_retryExecute() throws Exception {
    executionService
        .whenExecute(DUMMY_REQUEST)
        .thenDone(
            ExecuteResponse.newBuilder()
                .setStatus(com.google.rpc.Status.newBuilder().setCode(Code.UNAVAILABLE_VALUE))
                .build());
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(2);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_unretriableErrorInResponse_reportError() {
    executionService
        .whenExecute(DUMMY_REQUEST)
        .thenDone(
            ExecuteResponse.newBuilder()
                .setStatus(com.google.rpc.Status.newBuilder().setCode(Code.INVALID_ARGUMENT_VALUE))
                .build());
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    IOException e =
        assertThrows(
            IOException.class,
            () -> {
              executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
            });

    assertThat(e).hasMessageThat().contains("INVALID_ARGUMENT");
    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_retryExecuteAndFail() {
    for (int i = 0; i <= MAX_RETRY_ATTEMPTS * 2; ++i) {
      executionService.whenExecute(DUMMY_REQUEST).thenError(Code.UNAVAILABLE);
    }

    IOException exception =
        assertThrows(
            IOException.class,
            () -> {
              executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
            });

    assertThat(executionService.getExecTimes()).isEqualTo(MAX_RETRY_ATTEMPTS + 1);
    assertThat(exception).hasMessageThat().contains("UNAVAILABLE");
  }

  @Test
  public void executeRemotely_executeAndWait() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    assertThat(executionService.getWaitTimes()).isEqualTo(1);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_executeAndRetryWait() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    assertThat(executionService.getWaitTimes()).isEqualTo(1);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_executeAndRetryWait_forever() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    int errorTimes = MAX_RETRY_ATTEMPTS * 2;
    for (int i = 0; i < errorTimes; ++i) {
      executionService.whenWaitExecution(DUMMY_REQUEST).thenAck().thenError(Code.DEADLINE_EXCEEDED);
    }
    executionService.whenWaitExecution(DUMMY_REQUEST).thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    assertThat(executionService.getWaitTimes()).isEqualTo(errorTimes + 1);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_executeAndRetryWait_failForConsecutiveErrors() {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    for (int i = 0; i < MAX_RETRY_ATTEMPTS * 2; ++i) {
      executionService.whenWaitExecution(DUMMY_REQUEST).thenError(Code.UNAVAILABLE);
    }

    assertThrows(
        IOException.class,
        () -> {
          executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
        });

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    // Implementation detail:
    //
    // The retry times is MAX_RETRY_ATTEMPTS + 2 instead of MAX_RETRY_ATTEMPTS + 1, because we reset
    // waitExecutionBackoff unconditionally when we receive a response that is not an error.
    //
    // For a ProgressiveBackoff, once a reset() is called, the next call to nextDelayMillis() will
    // not increase the internal counter. So there will be one more retry here.
    assertThat(executionService.getWaitTimes()).isEqualTo(MAX_RETRY_ATTEMPTS + 2);
  }

  @Test
  public void executeRemotely_operationWithoutResult_shouldNotCrash() {
    executionService.whenExecute(DUMMY_REQUEST).thenDone();

    IOException e =
        assertThrows(
            IOException.class,
            () -> {
              executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
            });

    assertThat(e).hasCauseThat().isInstanceOf(ExecutionStatusException.class);
    ExecutionStatusException executionStatusException = (ExecutionStatusException) e.getCause();
    assertThat(executionStatusException.getStatus().getCode()).isEqualTo(Status.Code.DATA_LOSS);
    // Shouldn't retry in this case
    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_responseWithoutResult_shouldNotCrash() {
    executionService.whenExecute(DUMMY_REQUEST).thenDone(ExecuteResponse.getDefaultInstance());

    IOException e =
        assertThrows(
            IOException.class,
            () -> {
              executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
            });

    assertThat(e).hasCauseThat().isInstanceOf(ExecutionStatusException.class);
    ExecutionStatusException executionStatusException = (ExecutionStatusException) e.getCause();
    assertThat(executionStatusException.getStatus().getCode()).isEqualTo(Status.Code.DATA_LOSS);
    // Shouldn't retry in this case
    assertThat(executionService.getExecTimes()).isEqualTo(1);
  }

  @Test
  public void executeRemotely_retryExecuteWhenUnauthenticated()
      throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenError(Code.UNAUTHENTICATED);
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(2);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_retryWaitExecutionWhenUnauthenticated()
      throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenAck().thenError(Code.UNAUTHENTICATED);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    assertThat(executionService.getWaitTimes()).isEqualTo(2);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_retryExecuteIfNotFound() throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenError(Code.NOT_FOUND);
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
    executionService.whenWaitExecution(DUMMY_REQUEST).thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(2);
    assertThat(executionService.getWaitTimes()).isEqualTo(2);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }

  @Test
  public void executeRemotely_notFoundLoop_reportError() {
    for (int i = 0; i <= MAX_RETRY_ATTEMPTS * 2; ++i) {
      executionService.whenExecute(DUMMY_REQUEST).thenAck().thenError(Code.UNAVAILABLE);
      executionService.whenWaitExecution(DUMMY_REQUEST).thenAck().thenError(Code.NOT_FOUND);
    }

    IOException e =
        assertThrows(
            IOException.class,
            () -> {
              executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);
            });

    assertThat(e).hasCauseThat().isInstanceOf(ExecutionStatusException.class);
    ExecutionStatusException executionStatusException = (ExecutionStatusException) e.getCause();
    assertThat(executionStatusException.getStatus().getCode()).isEqualTo(Status.Code.NOT_FOUND);
    assertThat(executionService.getExecTimes()).isEqualTo(MAX_RETRY_ATTEMPTS + 1);
    assertThat(executionService.getWaitTimes()).isEqualTo(MAX_RETRY_ATTEMPTS + 1);
  }

  @Test
  public void executeRemotely_notifyObserver() throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    List<Operation> notified = new ArrayList<>();
    executor.executeRemotely(context, DUMMY_REQUEST, notified::add);

    assertThat(notified)
        .containsExactly(
            FakeExecutionService.ackOperation(DUMMY_REQUEST),
            FakeExecutionService.doneOperation(DUMMY_REQUEST, DUMMY_RESPONSE));
  }
}
