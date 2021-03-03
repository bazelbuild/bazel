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

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.core.CompletableObserver;
import io.reactivex.rxjava3.core.CompletableOnSubscribe;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.exceptions.Exceptions;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** Methods for interoperating between Rx and ListenableFuture. */
public class RxFutures {

  private RxFutures() {}

  /**
   * Returns a {@link Completable} that is complete once the supplied {@link ListenableFuture} has
   * completed.
   *
   * <p>A {@link ListenableFuture>} represents some computation that is already in progress. We use
   * {@link Callable} here to defer the execution of the thing that produces ListenableFuture until
   * there is subscriber.
   *
   * <p>Errors are also propagated except for certain "fatal" exceptions defined by rxjava. Multiple
   * subscriptions are not allowed.
   *
   * <p>Disposes the Completable to cancel the underlying ListenableFuture.
   */
  public static Completable toCompletable(
      Callable<ListenableFuture<Void>> callable, Executor executor) {
    return Completable.create(new OnceCompletableOnSubscribe(callable, executor));
  }

  private static class OnceCompletableOnSubscribe implements CompletableOnSubscribe {
    private final AtomicBoolean subscribed = new AtomicBoolean(false);

    private final Callable<ListenableFuture<Void>> callable;
    private final Executor executor;

    private OnceCompletableOnSubscribe(
        Callable<ListenableFuture<Void>> callable, Executor executor) {
      this.callable = callable;
      this.executor = executor;
    }

    @Override
    public void subscribe(@NonNull CompletableEmitter emitter) throws Throwable {
      try {
        checkState(!subscribed.getAndSet(true), "This completable cannot be subscribed to twice");
        ListenableFuture<Void> future = callable.call();
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
   * Returns a {@link ListenableFuture} that is complete once the {@link Completable} has completed.
   *
   * <p>Errors are also propagated. If the {@link ListenableFuture} is canceled, the subscription to
   * the {@link Completable} will automatically be cancelled.
   */
  public static ListenableFuture<Void> toListenableFuture(Completable completable) {
    CompletableFuture future = new CompletableFuture();
    completable.subscribe(
        new CompletableObserver() {
          @Override
          public void onSubscribe(Disposable d) {
            future.setCancelCallback(d);
          }

          @Override
          public void onComplete() {
            // Making the Completable as complete.
            future.set(null);
          }

          @Override
          public void onError(Throwable e) {
            future.setException(e);
          }
        });
    return future;
  }

  private static final class CompletableFuture extends AbstractFuture<Void> {
    private final AtomicReference<Disposable> cancelCallback = new AtomicReference<>();

    private void setCancelCallback(Disposable cancelCallback) {
      this.cancelCallback.set(cancelCallback);
      // Just in case it was already canceled before we set the callback.
      doCancelIfCancelled();
    }

    private void doCancelIfCancelled() {
      if (isCancelled()) {
        Disposable callback = cancelCallback.getAndSet(null);
        if (callback != null) {
          callback.dispose();
        }
      }
    }

    @Override
    protected void afterDone() {
      doCancelIfCancelled();
    }

    // Allow set to be called by other members.
    @Override
    protected boolean set(@Nullable Void t) {
      return super.set(t);
    }

    // Allow setException to be called by other members.
    @Override
    protected boolean setException(Throwable throwable) {
      return super.setException(throwable);
    }
  }
}
