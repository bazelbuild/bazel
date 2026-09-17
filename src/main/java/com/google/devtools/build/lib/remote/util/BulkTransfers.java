// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;

/** Utility methods for bulk transfers. */
public final class BulkTransfers {

  /**
   * Waits for all transfers to finish.
   *
   * <p>If interrupted, all remaining transfers are canceled.
   */
  public static void waitForBulkTransfer(Iterable<? extends ListenableFuture<?>> transfers)
      throws BulkTransferException, InterruptedException {
    BulkTransferException bulkTransferException = null;
    InterruptedException interruptedException = null;
    boolean interrupted = Thread.currentThread().isInterrupted();
    for (ListenableFuture<?> transfer : transfers) {
      try {
        if (interruptedException == null) {
          // Wait for all transfers to finish.
          var unused = getFromFuture(transfer, /* cancelOnInterrupt= */ true);
        } else {
          transfer.cancel(true);
        }
      } catch (IOException e) {
        if (bulkTransferException == null) {
          bulkTransferException = new BulkTransferException();
        }
        bulkTransferException.add(e);
      } catch (InterruptedException e) {
        interrupted = Thread.interrupted() || interrupted;
        interruptedException = e;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    if (interruptedException != null) {
      if (bulkTransferException != null) {
        interruptedException.addSuppressed(bulkTransferException);
      }
      throw interruptedException;
    }
    if (bulkTransferException != null) {
      throw bulkTransferException;
    }
  }

  public static ListenableFuture<Void> mergeBulkTransfer(
      Iterable<ListenableFuture<Void>> transfers) {
    return Futures.whenAllComplete(transfers)
        .callAsync(
            () -> {
              BulkTransferException bulkTransferException = null;

              for (var transfer : transfers) {
                IOException error = null;
                try {
                  transfer.get();
                } catch (CancellationException e) {
                  return immediateFailedFuture(new InterruptedException());
                } catch (InterruptedException e) {
                  return immediateFailedFuture(e);
                } catch (ExecutionException e) {
                  var cause = e.getCause();
                  if (cause instanceof InterruptedException) {
                    return immediateFailedFuture(cause);
                  } else if (cause instanceof IOException ioException) {
                    error = ioException;
                  } else {
                    error = new IOException(cause);
                  }
                }

                if (error == null) {
                  continue;
                }

                if (bulkTransferException == null) {
                  bulkTransferException = new BulkTransferException();
                }
                bulkTransferException.add(error);
              }

              if (bulkTransferException != null) {
                return immediateFailedFuture(bulkTransferException);
              }

              return immediateVoidFuture();
            },
            directExecutor());
  }

  private BulkTransfers() {}
}
