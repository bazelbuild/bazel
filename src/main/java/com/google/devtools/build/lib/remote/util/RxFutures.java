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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.core.CompletableObserver;
import io.reactivex.rxjava3.core.CompletableOnSubscribe;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.core.SingleEmitter;
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.core.SingleOnSubscribe;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.exceptions.Exceptions;
import io.reactivex.rxjava3.functions.Supplier;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** Methods for interoperating between Rx and ListenableFuture. */
public class RxFutures {

  private RxFutures() {}

  /**
   * Returns a {@link Completable} that is complete once the supplied {@link ListenableFuture} has
   * completed.
   *
   * <p>A {@link ListenableFuture} represents some computation that is already in progress. We use
   * {@link Supplier} here to defer the execution of the thing that produces ListenableFuture until
   * there is subscriber.
   *
   * <p>Errors are also propagated except for certain "fatal" exceptions defined by rxjava. Multiple
   * subscriptions are not allowed.
   *
   * <p>Disposes the Completable to cancel the underlying ListenableFuture.
   */
  public static Completable toCompletable(
      Supplier<ListenableFuture<Void>> supplier, Executor executor) {
    return Completable.create(new OnceCompletableOnSubscribe(supplier, executor));
  }

  private static class OnceCompletableOnSubscribe implements CompletableOnSubscribe {
    private final AtomicBoolean subscribed = new AtomicBoolean(false);

    private final Supplier<ListenableFuture<Void>> supplier;
    private final Executor executor;

    private OnceCompletableOnSubscribe(
        Supplier<ListenableFuture<Void>> supplier, Executor executor) {
      this.supplier = supplier;
      this.executor = executor;
    }

    @Override
    public void subscribe(@NonNull CompletableEmitter emitter) throws Throwable {
      try {
        checkState(!subscribed.getAndSet(true), "This completable cannot be subscribed to twice");
        ListenableFuture<Void> future = supplier.get();
        Futures.addCallback(
            future,
            new FutureCallback<Void>() {
              @Override
              public void onSuccess(@Nullable Void t) {
                emitter.onComplete();
              }

              @Override
              public void onFailure(Throwable throwable) {
                /*
                 * CancellationException can be thrown in two cases:
                 *   1. The ListenableFuture itself is cancelled.
                 *   2. Completable is disposed by downstream.
                 *
                 * This check is used to prevent propagating CancellationException to downstream
                 * when it has already disposed the Completable.
                 */
                if (throwable instanceof CancellationException && emitter.isDisposed()) {
                  return;
                }

                emitter.onError(throwable);
              }
            },
            executor);
        emitter.setCancellable(() -> future.cancel(true));
      } catch (Throwable t) {
        // We failed to construct and listen to the LF. Following RxJava's own behaviour, prefer
        // to pass RuntimeExceptions and Errors down to the subscriber except for certain
        // "fatal" exceptions.
        Exceptions.throwIfFatal(t);
        executor.execute(() -> emitter.onError(t));
      }
    }
  }

  /**
   * Returns a {@link Single} that is complete once the supplied {@link ListenableFuture} has
   * completed.
   *
   * <p>A {@link ListenableFuture} represents some computation that is already in progress. We use
   * {@link Supplier} here to defer the execution of the thing that produces ListenableFuture until
   * there is subscriber.
   *
   * <p>Errors are also propagated except for certain "fatal" exceptions defined by rxjava. Multiple
   * subscriptions are not allowed.
   *
   * <p>Disposes the Single to cancel the underlying ListenableFuture.
   */
  public static <T> Single<T> toSingle(Supplier<ListenableFuture<T>> supplier, Executor executor) {
    return Single.create(new OnceSingleOnSubscribe<>(supplier, executor));
  }

  private static class OnceSingleOnSubscribe<T> implements SingleOnSubscribe<T> {
    private final AtomicBoolean subscribed = new AtomicBoolean(false);

    private final Supplier<ListenableFuture<T>> supplier;
    private final Executor executor;

    private OnceSingleOnSubscribe(Supplier<ListenableFuture<T>> supplier, Executor executor) {
      this.supplier = supplier;
      this.executor = executor;
    }

    @Override
    public void subscribe(@NonNull SingleEmitter<T> emitter) throws Throwable {
      try {
        checkState(!subscribed.getAndSet(true), "This single cannot be subscribed to twice");
        ListenableFuture<T> future = supplier.get();
        Futures.addCallback(
            future,
            new FutureCallback<T>() {
              @Override
              public void onSuccess(@Nullable T t) {
                emitter.onSuccess(t);
              }

              @Override
              public void onFailure(Throwable throwable) {
                /*
                 * CancellationException can be thrown in two cases:
                 *   1. The ListenableFuture itself is cancelled.
                 *   2. Single is disposed by downstream.
                 *
                 * This check is used to prevent propagating CancellationException to downstream
                 * when it has already disposed the Single.
                 */
                if (throwable instanceof CancellationException && emitter.isDisposed()) {
                  return;
                }

                emitter.onError(throwable);
              }
            },
            executor);
        emitter.setCancellable(() -> future.cancel(true));
      } catch (Throwable t) {
        // We failed to construct and listen to the LF. Following RxJava's own behaviour, prefer
        // to pass RuntimeExceptions and Errors down to the subscriber except for certain
        // "fatal" exceptions.
        Exceptions.throwIfFatal(t);
        executor.execute(() -> emitter.onError(t));
      }
    }
  }

  /**
   * Returns a {@link ListenableFuture} that is complete once the {@link Completable} has completed.
   *
   * <p>Errors are also propagated. If the {@link ListenableFuture} is canceled, the subscription to
   * the {@link Completable} will automatically be cancelled.
   */
  public static ListenableFuture<Void> toListenableFuture(Completable completable) {
    SettableFuture<Void> future = SettableFuture.create();
    completable.subscribe(
        new CompletableObserver() {
          @Override
          public void onSubscribe(Disposable d) {
            future.addListener(
                () -> {
                  if (future.isCancelled()) {
                    d.dispose();
                  }
                },
                directExecutor());
          }

          @Override
          public void onComplete() {
            // Making the Completable as complete.
            future.set(null);
          }

          @Override
          public void onError(Throwable e) {
            if (e instanceof InterruptedException) {
              future.cancel(true);
            } else if (e instanceof CancellationException) {
              future.cancel(true);
            } else {
              future.setException(e);
            }
          }
        });
    return future;
  }

  /**
   * Returns a {@link ListenableFuture} that is complete once the {@link Single} has succeeded.
   *
   * <p>Errors are also propagated. If the {@link ListenableFuture} is canceled, the subscription to
   * the {@link Single} will automatically be cancelled.
   */
  public static <T> ListenableFuture<T> toListenableFuture(Single<T> single) {
    SettableFuture<T> future = SettableFuture.create();
    single.subscribe(
        new SingleObserver<T>() {
          @Override
          public void onSubscribe(Disposable d) {
            future.addListener(
                () -> {
                  if (future.isCancelled()) {
                    d.dispose();
                  }
                },
                directExecutor());
          }

          @Override
          public void onSuccess(@NonNull T t) {
            future.set(t);
          }

          @Override
          public void onError(Throwable e) {
            if (e instanceof CancellationException) {
              future.cancel(true);
            } else {
              future.setException(e);
            }
          }
        });
    return future;
  }

}
