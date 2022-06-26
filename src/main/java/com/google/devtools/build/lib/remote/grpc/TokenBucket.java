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
package com.google.devtools.build.lib.remote.grpc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Observer;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.subjects.BehaviorSubject;
import java.io.Closeable;
import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.ConcurrentLinkedDeque;

/** A container for tokens which is used for rate limiting. */
@ThreadSafe
public class TokenBucket<T> implements Closeable {
  private final ConcurrentLinkedDeque<T> tokens;
  private final BehaviorSubject<T> tokenBehaviorSubject;

  public TokenBucket() {
    this(ImmutableList.of());
  }

  public TokenBucket(Collection<T> initialTokens) {
    tokens = new ConcurrentLinkedDeque<>(initialTokens);
    tokenBehaviorSubject = BehaviorSubject.create();

    if (!tokens.isEmpty()) {
      tokenBehaviorSubject.onNext(tokens.getFirst());
    }
  }

  /** Add a token to the bucket. */
  public void addToken(T token) {
    tokens.addLast(token);
    tokenBehaviorSubject.onNext(token);
  }

  /** Returns current number of tokens in the bucket. */
  public int size() {
    return tokens.size();
  }

  /**
   * Returns a cold {@link Single} which will start the token acquisition process upon subscription.
   */
  public Single<T> acquireToken() {
    return Single.create(
        downstream ->
            tokenBehaviorSubject.subscribe(
                new Observer<T>() {
                  Disposable upstream;

                  @Override
                  public void onSubscribe(@NonNull Disposable d) {
                    upstream = d;
                    downstream.setDisposable(d);
                  }

                  @Override
                  public void onNext(@NonNull T ignored) {
                    if (!downstream.isDisposed()) {
                      T token = tokens.pollFirst();
                      if (token != null) {
                        downstream.onSuccess(token);
                      }
                    }
                  }

                  @Override
                  public void onError(@NonNull Throwable e) {
                    downstream.onError(new IllegalStateException(e));
                  }

                  @Override
                  public void onComplete() {
                    if (!downstream.isDisposed()) {
                      downstream.onError(new IllegalStateException("closed"));
                    }
                  }
                }));
  }

  /**
   * Closes the bucket and release all the tokens.
   *
   * <p>Subscriptions after closed to the Single returned by {@link TokenBucket#acquireToken()} will
   * emit error.
   */
  @Override
  public void close() throws IOException {
    tokens.clear();
    tokenBehaviorSubject.onComplete();
  }
}
