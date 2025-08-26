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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.bytestream.ByteStreamGrpc.ByteStreamImplBase;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableScheduledFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.Retrier;
import com.google.devtools.build.lib.remote.Retrier.Backoff;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier.Result;
import com.google.protobuf.ByteString;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.stub.StreamObserver;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

/** Test utilities */
public class TestUtils {

  public static RemoteRetrier newRemoteRetrier(
      Supplier<Backoff> backoff,
      ResultClassifier resultClassifier,
      ListeningScheduledExecutorService retryScheduler) {
    ZeroDelayListeningScheduledExecutorService zeroDelayRetryScheduler =
        new ZeroDelayListeningScheduledExecutorService(retryScheduler);
    return new RemoteRetrier(
        backoff,
        (e) ->
            Status.fromThrowable(e).getCode() == Code.CANCELLED
                ? Result.SUCCESS
                : resultClassifier.test(e),
        zeroDelayRetryScheduler,
        Retrier.ALLOW_ALL_CALLS,
        (millis) -> {
          /* don't wait in tests */
        });
  }

  /**
   * Wraps around a {@link ListeningScheduledExecutorService} and schedules all tasks with zero
   * delay.
   */
  private static class ZeroDelayListeningScheduledExecutorService
      implements ListeningScheduledExecutorService {

    private final ListeningScheduledExecutorService delegate;

    ZeroDelayListeningScheduledExecutorService(ListeningScheduledExecutorService delegate) {
      this.delegate = delegate;
    }

    @Override
    public ListenableScheduledFuture<?> schedule(Runnable runnable, long l, TimeUnit timeUnit) {
      return delegate.schedule(runnable, 0, timeUnit);
    }

    @Override
    public <V> ListenableScheduledFuture<V> schedule(
        Callable<V> callable, long l, TimeUnit timeUnit) {
      return delegate.schedule(callable, 0, timeUnit);
    }

    @Override
    public ListenableScheduledFuture<?> scheduleAtFixedRate(
        Runnable runnable, long l, long l1, TimeUnit timeUnit) {
      return delegate.scheduleAtFixedRate(runnable, 0, 0, timeUnit);
    }

    @Override
    public ListenableScheduledFuture<?> scheduleWithFixedDelay(
        Runnable runnable, long l, long l1, TimeUnit timeUnit) {
      return delegate.scheduleWithFixedDelay(runnable, 0, 0, timeUnit);
    }

    @Override
    public void shutdown() {
      delegate.shutdown();
    }

    @Override
    public List<Runnable> shutdownNow() {
      return delegate.shutdownNow();
    }

    @Override
    public boolean isShutdown() {
      return delegate.isShutdown();
    }

    @Override
    public boolean isTerminated() {
      return delegate.isTerminated();
    }

    @Override
    public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
      return delegate.awaitTermination(timeout, unit);
    }

    @Override
    public <T> ListenableFuture<T> submit(Callable<T> callable) {
      return delegate.submit(callable);
    }

    @Override
    public ListenableFuture<?> submit(Runnable runnable) {
      return delegate.submit(runnable);
    }

    @Override
    public <T> ListenableFuture<T> submit(Runnable runnable, T t) {
      return delegate.submit(runnable, t);
    }

    @Override
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks)
        throws InterruptedException {
      return delegate.invokeAll(tasks);
    }

    @Override
    public <T> List<Future<T>> invokeAll(
        Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
        throws InterruptedException {
      return delegate.invokeAll(tasks, timeout, unit);
    }

    @Override
    public <T> T invokeAny(Collection<? extends Callable<T>> tasks)
        throws InterruptedException, ExecutionException {
      return delegate.invokeAny(tasks);
    }

    @Override
    public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
        throws InterruptedException, ExecutionException, TimeoutException {
      return delegate.invokeAny(tasks, timeout, unit);
    }

    @Override
    public void execute(Runnable command) {
      delegate.execute(command);
    }
  }

  public static final ByteStreamImplBase newNoErrorByteStreamService(byte[] blob) {
    return new ByteStreamImplBase() {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> streamObserver) {
        return new StreamObserver<WriteRequest>() {

          byte[] receivedData = new byte[blob.length];
          long nextOffset = 0;

          @Override
          public void onNext(WriteRequest writeRequest) {
            if (nextOffset == 0) {
              assertThat(writeRequest.getResourceName()).isNotEmpty();
              assertThat(writeRequest.getResourceName()).endsWith(String.valueOf(blob.length));
            } else {
              assertThat(writeRequest.getResourceName()).isEmpty();
            }

            assertThat(writeRequest.getWriteOffset()).isEqualTo(nextOffset);

            ByteString data = writeRequest.getData();

            System.arraycopy(data.toByteArray(), 0, receivedData, (int) nextOffset, data.size());

            nextOffset += data.size();
            boolean lastWrite = blob.length == nextOffset;
            assertThat(writeRequest.getFinishWrite()).isEqualTo(lastWrite);
          }

          @Override
          public void onError(Throwable throwable) {
            fail("onError should never be called.");
          }

          @Override
          public void onCompleted() {
            assertThat(nextOffset).isEqualTo(blob.length);
            assertThat(receivedData).isEqualTo(blob);

            WriteResponse response =
                WriteResponse.newBuilder().setCommittedSize(nextOffset).build();
            streamObserver.onNext(response);
            streamObserver.onCompleted();
          }
        };
      }
    };
  }
}
