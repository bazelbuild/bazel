// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.util.Utils;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for remote utility methods */
@RunWith(JUnit4.class)
public class UtilsTest {

  @Test
  public void testGrpcAwareErrorMessage() {
    IOException ioError = new IOException("io error");
    IOException wrappedGrpcError =
        new IOException(
            "wrapped error", Status.ABORTED.withDescription("grpc error").asRuntimeException());

    assertThat(Utils.grpcAwareErrorMessage(ioError, /* verboseFailures= */ false))
        .isEqualTo("io error");
    assertThat(Utils.grpcAwareErrorMessage(wrappedGrpcError, /* verboseFailures= */ false))
        .isEqualTo("ABORTED: grpc error");
  }

  @Test
  public void testGrpcAwareErrorMessage_verboseFailures() {
    IOException ioError = new IOException("io error");
    IOException wrappedGrpcError =
        new IOException(
            "wrapped error", Status.ABORTED.withDescription("grpc error").asRuntimeException());

    assertThat(Utils.grpcAwareErrorMessage(ioError, /* verboseFailures= */ true))
        .startsWith(
            "io error\n"
                + "java.io.IOException: io error\n"
                + "\tat com.google.devtools.build.lib.remote.UtilsTest.testGrpcAwareErrorMessage_verboseFailures");
    assertThat(Utils.grpcAwareErrorMessage(wrappedGrpcError, /* verboseFailures= */ true))
        .startsWith(
            "ABORTED: grpc error\n"
                + "java.io.IOException: wrapped error\n"
                + "\tat com.google.devtools.build.lib.remote.UtilsTest.testGrpcAwareErrorMessage_verboseFailures");
  }

  @Test
  public void refreshIfUnauthenticatedAsync_unauthenticated_shouldRefresh() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    Utils.refreshIfUnauthenticatedAsync(
            () -> {
              if (callTimes.getAndIncrement() == 0) {
                throw new StatusRuntimeException(Status.UNAUTHENTICATED);
              }
              return Futures.immediateFuture(null);
            },
            callCredentialsProvider)
        .get();

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticatedAsync_unauthenticatedFuture_shouldRefresh() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    Utils.refreshIfUnauthenticatedAsync(
            () -> {
              if (callTimes.getAndIncrement() == 0) {
                return Futures.immediateFailedFuture(
                    new StatusRuntimeException(Status.UNAUTHENTICATED));
              }
              return Futures.immediateFuture(null);
            },
            callCredentialsProvider)
        .get();

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticatedAsync_permissionDenied_shouldRefresh() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    Utils.refreshIfUnauthenticated(
            () -> {
              if (callTimes.getAndIncrement() == 0) {
                throw new StatusRuntimeException(Status.PERMISSION_DENIED);
              }
              return Futures.immediateFuture(null);
            },
            callCredentialsProvider)
        .get();

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticatedAsync_cantRefresh_shouldRefreshOnceAndFail()
      throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    assertThrows(
        ExecutionException.class,
        () -> {
          Utils.refreshIfUnauthenticatedAsync(
                  () -> {
                    callTimes.getAndIncrement();
                    throw new StatusRuntimeException(Status.UNAUTHENTICATED);
                  },
                  callCredentialsProvider)
              .get();
        });

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticated_unauthenticated_shouldRefresh() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    Utils.refreshIfUnauthenticated(
        () -> {
          if (callTimes.getAndIncrement() == 0) {
            throw new StatusRuntimeException(Status.UNAUTHENTICATED);
          }
          return null;
        },
        callCredentialsProvider);

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticated_permissionDenied_shouldRefresh() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    Utils.refreshIfUnauthenticated(
        () -> {
          if (callTimes.getAndIncrement() == 0) {
            throw new StatusRuntimeException(Status.PERMISSION_DENIED);
          }
          return null;
        },
        callCredentialsProvider);

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }

  @Test
  public void refreshIfUnauthenticated_cantRefresh_shouldRefreshOnceAndFail() throws Exception {
    CallCredentialsProvider callCredentialsProvider = mock(CallCredentialsProvider.class);
    AtomicInteger callTimes = new AtomicInteger();

    assertThrows(
        StatusRuntimeException.class,
        () -> {
          Utils.refreshIfUnauthenticated(
              () -> {
                callTimes.getAndIncrement();
                throw new StatusRuntimeException(Status.UNAUTHENTICATED);
              },
              callCredentialsProvider);
        });

    assertThat(callTimes.get()).isEqualTo(2);
    verify(callCredentialsProvider, times(1)).refresh();
  }
}
