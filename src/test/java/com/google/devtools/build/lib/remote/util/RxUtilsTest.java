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
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import io.reactivex.rxjava3.annotations.NonNull;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.CompletableEmitter;
import io.reactivex.rxjava3.core.CompletableObserver;
import io.reactivex.rxjava3.observers.TestObserver;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RxUtils}. */
@RunWith(JUnit4.class)
public class RxUtilsTest {
  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  static class SettableCompletable extends Completable {
    private final AtomicReference<CompletableEmitter> emitterRef = new AtomicReference<>(null);
    private final AtomicBoolean cancelled = new AtomicBoolean(false);
    private final AtomicBoolean completed = new AtomicBoolean(false);
    private final Completable completable =
        Completable.create(
            emitter -> {
              emitterRef.set(emitter);
              emitter.setCancellable(
                  () -> {
                    if (!completed.get()) {
                      cancelled.set(true);
                    }
                  });
            });

    public static SettableCompletable create() {
      return new SettableCompletable();
    }

    @Override
    protected void subscribeActual(@NonNull CompletableObserver observer) {
      completable.subscribe(observer);
    }

    public void setComplete() {
      completed.set(true);
      emitterRef.get().onComplete();
    }

    public void setError(Throwable error) {
      completed.set(true);
      emitterRef.get().onError(error);
    }

    public boolean cancelled() {
      return cancelled.get();
    }
  }

  @Test
  public void toTransferResult_onComplete_isOk() {
    SettableCompletable transfer = SettableCompletable.create();
    TestObserver<TransferResult> ob = toTransferResult(transfer).test();

    transfer.setComplete();

    ob.assertValue(
        result -> {
          assertThat(result.isOk()).isTrue();
          assertThat(result.isError()).isFalse();
          return true;
        });
  }

  @Test
  public void toTransferResult_onIOException_isError() {
    SettableCompletable transfer = SettableCompletable.create();
    TestObserver<TransferResult> ob = toTransferResult(transfer).test();
    IOException error = new IOException("IO error");

    transfer.setError(error);

    ob.assertValue(
        result -> {
          assertThat(result.isOk()).isFalse();
          assertThat(result.isError()).isTrue();
          assertThat(result.getError()).isEqualTo(error);
          return true;
        });
  }

  @Test
  public void toTransferResult_onOtherError_propagateError() {
    SettableCompletable transfer = SettableCompletable.create();
    TestObserver<TransferResult> ob = toTransferResult(transfer).test();
    Exception error = new Exception("other error");

    transfer.setError(error);

    ob.assertError(error);
  }

  @Test
  public void mergeBulkTransfer_allComplete_complete() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();
    TestObserver<Void> ob = mergeBulkTransfer(transfer1, transfer2, transfer3).test();

    transfer1.setComplete();
    transfer2.setComplete();
    transfer3.setComplete();

    ob.assertComplete();
  }

  @Test
  public void mergeBulkTransfer_hasPendingTransfer_pending() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();
    TestObserver<Void> ob = mergeBulkTransfer(transfer1, transfer2, transfer3).test();

    transfer1.setComplete();
    transfer2.setComplete();

    ob.assertNotComplete();
    ob.assertNoErrors();
  }

  @Test
  public void mergeBulkTransfer_onIOErrors_keepOtherTransfers() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();
    TestObserver<Void> ob = mergeBulkTransfer(transfer1, transfer2, transfer3).test();
    IOException error = new IOException("IO error");

    transfer1.setError(error);
    transfer2.setComplete();
    transfer3.setComplete();

    ob.assertError(BulkTransferException.class);
    assertThat(transfer2.cancelled()).isFalse();
    assertThat(transfer3.cancelled()).isFalse();
  }

  @Test
  public void mergeBulkTransfer_onIOErrors_wrapsIOErrorsInBulkTransferExceptions() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();
    TestObserver<Void> ob = mergeBulkTransfer(transfer1, transfer2, transfer3).test();
    IOException error1 = new IOException("IO error 1");
    IOException error2 = new IOException("IO error 2");

    transfer1.setError(error1);
    transfer2.setError(error2);
    transfer3.setComplete();

    ob.assertError(
        e -> {
          assertThat(e).isInstanceOf(BulkTransferException.class);
          assertThat(ImmutableList.copyOf(e.getSuppressed())).containsExactly(error1, error2);
          return true;
        });
  }

  @Test
  public void mergeBulkTransfer_onOtherError_cancelOtherTransfers() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();
    TestObserver<Void> ob = mergeBulkTransfer(transfer1, transfer2, transfer3).test();
    Exception error = new Exception("error");

    transfer1.setError(error);

    ob.assertError(error);
    assertThat(transfer2.cancelled()).isTrue();
    assertThat(transfer3.cancelled()).isTrue();
  }

  @Test
  public void mergeBulkTransfer_onInterruption_cancelOtherTransfers() {
    SettableCompletable transfer1 = SettableCompletable.create();
    SettableCompletable transfer2 = SettableCompletable.create();
    SettableCompletable transfer3 = SettableCompletable.create();

    Thread.currentThread().interrupt();
    RuntimeException error = null;
    try {
      mergeBulkTransfer(transfer1, transfer2, transfer3).blockingAwait();
    } catch (RuntimeException e) {
      error = e;
    }

    assertThat(error).hasCauseThat().isInstanceOf(InterruptedException.class);
    assertThat(transfer1.cancelled()).isTrue();
    assertThat(transfer2.cancelled()).isTrue();
    assertThat(transfer3.cancelled()).isTrue();
  }
}
