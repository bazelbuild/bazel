// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.observers.TestObserver;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RxFutures}. */
@RunWith(JUnit4.class)
public class RxFuturesTest {
  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  @Test
  public void toCompletable_noSubscription_noExecution() {
    SettableFuture<Void> future = SettableFuture.create();
    AtomicBoolean executed = new AtomicBoolean(false);

    toCompletable(
        () -> {
          executed.set(true);
          return future;
        },
        MoreExecutors.directExecutor());

    assertThat(executed.get()).isFalse();
  }

  @Test
  public void toCompletable_futureOnSuccess_completableOnComplete() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    TestObserver<Void> observer = completable.test();
    observer.assertEmpty();
    future.set(null);

    observer.assertComplete();
  }

  @Test
  public void toCompletable_futureOnError_completableOnError() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    TestObserver<Void> observer = completable.test();
    observer.assertEmpty();
    Throwable error = new IllegalStateException("error");
    future.setException(error);

    observer.assertError(error);
  }

  @Test
  public void toCompletable_futureOnSuccessBeforeSubscription_completableOnComplete() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    future.set(null);
    TestObserver<Void> observer = completable.test();

    observer.assertComplete();
  }

  @Test
  public void toCompletable_futureOnErrorBeforeSubscription_completableOnError() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    Throwable error = new IllegalStateException("error");
    future.setException(error);
    TestObserver<Void> observer = completable.test();

    observer.assertError(error);
  }

  @Test
  public void toCompletable_futureCancelledBeforeSubscription_completableOnError() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    future.cancel(true);
    TestObserver<Void> observer = completable.test();

    observer.assertError(CancellationException.class);
  }

  @Test
  public void toCompletable_futureCancelled_completableOnError() {
    ListenableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, directExecutor());

    TestObserver<Void> observer = completable.test();
    observer.assertNotComplete();
    future.cancel(true);

    observer.assertError(CancellationException.class);
  }

  @Test
  public void toCompletable_disposeCompletable_cancelFuture() {
    SettableFuture<Void> future = SettableFuture.create();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());

    TestObserver<Void> observer = completable.test();
    observer.assertEmpty();
    observer.dispose();

    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void toCompletable_multipleSubscriptions_error() {
    ListenableFuture<Void> future = immediateVoidFuture();
    Completable completable = toCompletable(() -> future, MoreExecutors.directExecutor());
    completable.test().assertComplete();

    TestObserver<Void> observer = completable.test();

    observer.assertError(IllegalStateException.class);
  }

  @Test
  public void toListenableFutureFromCompletable_noEvents_waiting() {
    CompletableToListenableFutureSetup setup = CompletableToListenableFutureSetup.create();

    assertThat(setup.getEmitter()).isNotNull();
    assertThat(setup.getFuture().isDone()).isFalse();
    assertThat(setup.getFuture().isCancelled()).isFalse();
  }

  @Test
  public void toListenableFutureFromCompletable_completableOnComplete_futureOnSuccess() {
    CompletableToListenableFutureSetup setup = CompletableToListenableFutureSetup.create();

    setup.getEmitter().onComplete();

    assertThat(setup.isSuccess()).isTrue();
    assertThat(setup.getFailure()).isNull();
  }

  @Test
  public void toListenableFutureFromCompletable_completableOnError_futureOnFailure() {
    CompletableToListenableFutureSetup setup = CompletableToListenableFutureSetup.create();

    Throwable error = new IllegalStateException("error");
    setup.getEmitter().onError(error);

    assertThat(setup.isSuccess()).isFalse();
    assertThat(setup.getFailure()).isEqualTo(error);
  }

  @Test
  public void toListenableFutureFromCompletable_cancelled() {
    CompletableToListenableFutureSetup setup = CompletableToListenableFutureSetup.create();

    setup.getFuture().cancel(true);

    assertThat(setup.isSuccess()).isFalse();
    assertThat(setup.getFailure()).isInstanceOf(CancellationException.class);
    assertThat(setup.isDisposed()).isTrue();
  }

  @Test
  public void toListenableFutureFromCompletable_sourceFutureCancelled_cancelFuture() {
    SettableFuture<Void> source = SettableFuture.create();
    ListenableFuture<Void> future =
        toListenableFuture(toCompletable(() -> source, directExecutor()));

    source.cancel(true);

    assertThat(future.isCancelled()).isTrue();
  }

  private static class CompletableToListenableFutureSetup {
    public static CompletableToListenableFutureSetup create() {
      return new CompletableToListenableFutureSetup();
    }

    private final ListenableFuture<Void> future;

    private CompletableEmitter emitter;
    private boolean disposed;
    private boolean success;
    private Throwable failure;

    CompletableToListenableFutureSetup() {
      Completable completable =
          Completable.create(emitter -> this.emitter = emitter).doOnDispose(() -> disposed = true);
      future = toListenableFuture(completable);
      Futures.addCallback(
          future,
          new FutureCallback<Void>() {
            @Override
            public void onSuccess(@Nullable Void result) {
              success = true;
            }

            @Override
            public void onFailure(Throwable t) {
              failure = t;
            }
          },
          MoreExecutors.directExecutor());
    }

    public CompletableEmitter getEmitter() {
      return emitter;
    }

    public ListenableFuture<Void> getFuture() {
      return future;
    }

    public boolean isDisposed() {
      return disposed;
    }

    public boolean isSuccess() {
      return success;
    }

    public Throwable getFailure() {
      return failure;
    }
  }
}
