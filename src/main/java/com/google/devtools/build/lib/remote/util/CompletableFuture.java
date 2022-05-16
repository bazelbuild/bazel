// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.AbstractFuture;
import io.reactivex.rxjava3.disposables.Disposable;
import io.reactivex.rxjava3.functions.Action;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * A {@link com.google.common.util.concurrent.ListenableFuture} whose result can be set by a {@link
 * #set(Object)} or {@link #setException(Throwable)}.
 *
 * <p>It differs from {@link com.google.common.util.concurrent.SettableFuture} that it provides
 * {@link #setCancelCallback(Disposable)} for callers to register a callback which is called when
 * the future is cancelled.
 */
public final class CompletableFuture<T> extends AbstractFuture<T> {

  public static <T> CompletableFuture<T> create() {
    return new CompletableFuture<>();
  }

  private final AtomicReference<Disposable> cancelCallback = new AtomicReference<>();

  public void setCancelCallback(Action action) {
    setCancelCallback(Disposable.fromAction(action));
  }

  public void setCancelCallback(Disposable cancelCallback) {
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
  public boolean set(@Nullable T t) {
    return super.set(t);
  }

  // Allow setException to be called by other members.
  @Override
  public boolean setException(Throwable throwable) {
    return super.setException(throwable);
  }
}
