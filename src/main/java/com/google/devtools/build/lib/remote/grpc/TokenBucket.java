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
import io.reactivex.rxjava3.core.SingleObserver;
import io.reactivex.rxjava3.core.Single;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.subjects.BehaviorSubject;
import io.reactivex.rxjava3.subjects.Subject;
import java.io.Closeable;
import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.ConcurrentLinkedDeque;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** A container for tokens which is used for rate limiting. */
@ThreadSafe
public class TokenBucket<T> implements Closeable {
  private final ConcurrentLinkedDeque<T> tokens;
  // Serialized, as tokens are added from the callbacks of concurrently closing connections.
  private final Subject<T> tokenSubject;

  public TokenBucket() {
    this(ImmutableList.of());
  }

  public TokenBucket(Collection<T> initialTokens) {
    tokens = new ConcurrentLinkedDeque<>(initialTokens);
    tokenSubject = BehaviorSubject.<T>create().toSerialized();
    if (!tokens.isEmpty()) {
      tokenSubject.onNext(tokens.getFirst());
    }
  }

  /** Add a token to the bucket. */
  public void addToken(T token) {
    tokens.addLast(token);
    tokenSubject.onNext(token);
  }

  /** Returns current number of tokens in the bucket. */
  public int size() {
    return tokens.size();
  }

  @Nullable
  public T tryAcquireToken() {
    return tokens.pollFirst();
  }

  /**
   * Returns a cold {@link Single} which will start the token acquisition process upon subscription.
   */
  public Single<T> acquireToken() {
    return new Single<T>() {
      @Override
      protected void subscribeActual(SingleObserver<? super T> observer) {
        var acquisition = new Acquisition(observer);
        observer.onSubscribe(acquisition);
        tokenSubject.subscribe(acquisition);
      }
    };
  }

  /**
   * A pending acquisition of a token, which decides under its own lock whether a token it takes
   * from the bucket is delivered: an emitter would silently discard a token delivered after the
   * downstream has been disposed.
   */
  private final class Acquisition implements Observer<T>, Disposable {
    private final SingleObserver<? super T> downstream;
    private final Object lock = new Object();

    @GuardedBy("lock")
    private boolean disposed;

    @GuardedBy("lock")
    private boolean done;

    @Nullable private volatile Disposable upstream;

    Acquisition(SingleObserver<? super T> downstream) {
      this.downstream = downstream;
    }

    @Override
    public void onSubscribe(@NonNull Disposable d) {
      upstream = d;
      synchronized (lock) {
        if (disposed || done) {
          d.dispose();
        }
      }
    }

    @Override
    public void onNext(@NonNull T ignored) {
      T token;
      synchronized (lock) {
        if (disposed || done) {
          return;
        }
        token = tokens.pollFirst();
        if (token == null) {
          return;
        }
        done = true;
      }
      disposeUpstream();
      downstream.onSuccess(token);
    }

    @Override
    public void onError(@NonNull Throwable e) {
      synchronized (lock) {
        if (disposed || done) {
          return;
        }
        done = true;
      }
      downstream.onError(new IllegalStateException(e));
    }

    @Override
    public void onComplete() {
      synchronized (lock) {
        if (disposed || done) {
          return;
        }
        done = true;
      }
      downstream.onError(new IllegalStateException("closed"));
    }

    @Override
    public void dispose() {
      synchronized (lock) {
        disposed = true;
      }
      disposeUpstream();
    }

    @Override
    public boolean isDisposed() {
      synchronized (lock) {
        return disposed;
      }
    }

    private void disposeUpstream() {
      Disposable d = upstream;
      if (d != null) {
        d.dispose();
      }
    }
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
    tokenSubject.onComplete();
  }
}
