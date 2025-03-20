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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.remote.common.BulkTransferException;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import javax.annotation.Nullable;

/** Utility methods for the Rx. * */
public class RxUtils {
  private RxUtils() {}

  /** Result of an I/O operation to remote cache. */
  public static class TransferResult {
    private static final TransferResult OK = new TransferResult(null, false);

    private static final TransferResult INTERRUPTED = new TransferResult(null, true);

    public static TransferResult ok() {
      return OK;
    }

    public static TransferResult interrupted() {
      return INTERRUPTED;
    }

    public static TransferResult error(IOException error) {
      return new TransferResult(error, false);
    }

    @Nullable private final IOException error;

    private final boolean interrupted;

    TransferResult(@Nullable IOException error, boolean interrupted) {
      this.error = error;
      this.interrupted = interrupted;
    }

    /** Returns {@code true} if the operation succeed. */
    public boolean isOk() {
      return error == null && !interrupted;
    }

    /** Returns {@code true} if the operation failed. */
    public boolean isError() {
      return error != null;
    }

    public boolean isInterrupted() {
      return interrupted;
    }

    /** Returns the IO error if the operation failed. */
    @Nullable
    public IOException getError() {
      return error;
    }
  }

  /**
   * Converts the {@link Completable} to {@link Single} which will emit {@link TransferResult} on
   * complete or IO errors. Other errors will be propagated to downstream.
   */
  public static Single<TransferResult> toTransferResult(Completable completable) {
    return completable
        .toSingleDefault(TransferResult.ok())
        .onErrorResumeNext(
            error -> {
              if (error instanceof IOException ioException) {
                return Single.just(TransferResult.error(ioException));
              } else if (error instanceof InterruptedException) {
                return Single.just(TransferResult.interrupted());
              } else {
                return Single.error(error);
              }
            });
  }

  private static class BulkTransferExceptionCollector {
    private BulkTransferException bulkTransferException;
    private boolean interrupted = false;

    void onResult(TransferResult result) {
      if (result.isOk()) {
        return;
      }

      if (result.isInterrupted()) {
        interrupted = true;
        return;
      }

      IOException error = checkNotNull(result.getError());
      if (bulkTransferException == null) {
        bulkTransferException = new BulkTransferException();
      }

      bulkTransferException.add(error);
    }

    Completable toCompletable() {
      if (interrupted) {
        return Completable.error(new InterruptedException());
      }

      if (bulkTransferException != null) {
        return Completable.error(bulkTransferException);
      }

      return Completable.complete();
    }
  }

  /**
   * Returns a {@link Completable} which will complete when the {@link Flowable} complete.
   *
   * <p>Errors of {@link TransferResult#getError()} are wrapped in {@link BulkTransferException}.
   * Other errors are propagated to downstream.
   */
  public static Completable mergeBulkTransfer(Flowable<TransferResult> transfers) {
    return transfers
        .collectInto(new BulkTransferExceptionCollector(), BulkTransferExceptionCollector::onResult)
        .flatMapCompletable(BulkTransferExceptionCollector::toCompletable);
  }

  /**
   * Returns a {@link Completable} which will complete when all the passed in {@link Completable}s
   * complete.
   *
   * <p>{@link IOException}s emitted by the passed in {@link Completable}s are wrapped in {@link
   * BulkTransferException}. Other errors are propagated to downstream.
   */
  public static Completable mergeBulkTransfer(Completable... transfers) {
    Flowable<TransferResult> flowable =
        Flowable.fromArray(transfers).flatMapSingle(RxUtils::toTransferResult);
    return mergeBulkTransfer(flowable);
  }
}
