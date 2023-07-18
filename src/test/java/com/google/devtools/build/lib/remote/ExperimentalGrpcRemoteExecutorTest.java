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

import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.util.TestUtils;
import io.grpc.Status;
import java.io.IOException;
import java.util.concurrent.Executors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExperimentalGrpcRemoteExecutor}. */
@RunWith(JUnit4.class)
public class ExperimentalGrpcRemoteExecutorTest extends GrpcRemoteExecutorTestBase {
  private ListeningScheduledExecutorService retryService;

  @Override
  protected RemoteExecutionClient createExecutionService(
      ServerCapabilities caps, ReferenceCountedChannel channel) throws Exception {
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions),
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryService);

    return new ExperimentalGrpcRemoteExecutor(
        caps, remoteOptions, channel, CallCredentialsProvider.NO_CREDENTIALS, retrier);
  }

  @Override
  public void setUp() throws Exception {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
    super.setUp();
  }

  @Override
  public void tearDown() throws Exception {
    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
    super.tearDown();
  }

  @Test
  public void executeRemotely_executeAndRetryWait_forever() throws Exception {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().finish();
    int errorTimes = MAX_RETRY_ATTEMPTS * 2;
    for (int i = 0; i < errorTimes; ++i) {
      executionService
          .whenWaitExecution(DUMMY_REQUEST)
          .thenAck()
          .thenError(Status.DEADLINE_EXCEEDED.asRuntimeException());
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
    executionService.whenExecute(DUMMY_REQUEST).thenAck().finish();
    for (int i = 0; i < MAX_RETRY_ATTEMPTS * 2; ++i) {
      executionService
          .whenWaitExecution(DUMMY_REQUEST)
          .thenError(Status.UNAVAILABLE.asRuntimeException());
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
  public void executeRemotely_retryWaitExecutionWhenUnauthenticated()
      throws IOException, InterruptedException {
    executionService.whenExecute(DUMMY_REQUEST).thenAck().finish();
    executionService
        .whenWaitExecution(DUMMY_REQUEST)
        .thenAck()
        .thenError(Status.UNAUTHENTICATED.asRuntimeException());
    executionService.whenWaitExecution(DUMMY_REQUEST).thenAck().thenDone(DUMMY_RESPONSE);

    ExecuteResponse response =
        executor.executeRemotely(context, DUMMY_REQUEST, OperationObserver.NO_OP);

    assertThat(executionService.getExecTimes()).isEqualTo(1);
    assertThat(executionService.getWaitTimes()).isEqualTo(2);
    assertThat(response).isEqualTo(DUMMY_RESPONSE);
  }
}
