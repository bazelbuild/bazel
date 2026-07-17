// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal.junit4;

import static com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory.CancellationRequest.HARD_STOP;
import static com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory.CancellationRequest.NOT_REQUESTED;
import static com.google.testing.junit.runner.internal.junit4.CancellableRequestFactory.CancellationRequest.ORDERLY_STOP;

import com.google.testing.junit.junit4.runner.RunNotifierWrapper;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.runner.Description;
import org.junit.runner.Request;
import org.junit.runner.Runner;
import org.junit.runner.notification.RunNotifier;
import org.junit.runner.notification.StoppedByUserException;

/**
 * Creates requests that can be cancelled.
 */
public class CancellableRequestFactory {
  private boolean requestCreated;
  private volatile ThreadSafeRunNotifier currentNotifier;
  private final AtomicReference<CancellationRequest> cancellationRequest =
      new AtomicReference<>(NOT_REQUESTED);

  public CancellableRequestFactory() {}

  /**
   * Creates a request that can be cancelled. Can only be called once.
   *
   * @param delegate request to wrap
   */
  public Request createRequest(Request delegate) {
    if (requestCreated) {
      throw new IllegalStateException("a request was already created");
    }
    requestCreated = true;

    return new MemoizingRequest(delegate) {
      @Override
      Runner createRunner(Request delegate) {
        return new CancellableRunner(delegate.getRunner());
      }
    };
  }

  /**
   * Cancels the request created by this request factory.
   */
  public void cancelRun() {
    stop(true);
  }

  /** Cancels the request created by this request factory as orderly as possible. */
  public void cancelRunOrderly() {
    stop(false);
  }

  private void stop(boolean hardStop) {
    if (cancellationRequest.compareAndSet(NOT_REQUESTED, hardStop ? HARD_STOP : ORDERLY_STOP)) {
      RunNotifier notifier = currentNotifier;
      if (notifier != null) {
        notifier.pleaseStop();
      }
    }
  }


  private class CancellableRunner extends Runner {
    private final Runner delegate;

    public CancellableRunner(Runner delegate) {
      this.delegate = delegate;
    }

    @Override
    public Description getDescription() {
      return delegate.getDescription();
    }

    @Override
    public void run(RunNotifier notifier) {
      currentNotifier = new ThreadSafeRunNotifier(notifier);
      if (cancellationRequest.get() != NOT_REQUESTED) {
        currentNotifier.pleaseStop();
      }
      if (cancellationRequest.get() == ORDERLY_STOP) {
        return;
      }

      try {
        delegate.run(currentNotifier);
      } catch (StoppedByUserException e) {
        if (cancellationRequest.get() == HARD_STOP) {
          throw new RuntimeException("Test run interrupted", e);
        } else if (cancellationRequest.get() == ORDERLY_STOP) {
          e.printStackTrace();
          return;
        }
        throw e;
      }
    }
  }


  private static class ThreadSafeRunNotifier extends RunNotifierWrapper {
    private volatile boolean stopRequested;

    public ThreadSafeRunNotifier(RunNotifier delegate) {
      super(delegate);
    }

    /**
     * {@inheritDoc}<p>
     *
     * The implementation is almost an exact copy of the version in
     * {@code RunNotifier} but is thread-safe.
     */
    @Override
    public void fireTestStarted(Description description) throws StoppedByUserException {
      if (stopRequested) {
        throw new StoppedByUserException();
      }
      getDelegate().fireTestStarted(description);
    }

    /**
     * {@inheritDoc}<p>
     *
     * This method is thread-safe.
     */
    @Override
    public void pleaseStop() {
      stopRequested = true;
    }
  }

  /** Cancellation request types of a {@link CancellableRequestFactory}. */
  enum CancellationRequest {
    NOT_REQUESTED, // Initial state of CancellableRequestFactory
    HARD_STOP, // Propagates StoppedByUserException
    ORDERLY_STOP // Catches StoppedByUserException and prevents further test runs
  }
}
