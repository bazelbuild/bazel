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

package com.google.testing.junit.runner.junit4;

import com.google.common.base.Preconditions;
import com.google.inject.Singleton;
import com.google.testing.junit.junit4.runner.MemoizingRequest;
import com.google.testing.junit.junit4.runner.RunNotifierWrapper;

import org.junit.runner.Description;
import org.junit.runner.Request;
import org.junit.runner.Runner;
import org.junit.runner.notification.RunNotifier;
import org.junit.runner.notification.StoppedByUserException;

/**
 * Creates requests that can be cancelled.
 */
@Singleton
class CancellableRequestFactory {
  private boolean requestCreated;
  private volatile ThreadSafeRunNotifier currentNotifier;
  private volatile boolean cancelRequested = false;

  /**
   * Creates a request that can be cancelled. Can only be called once.
   *
   * @param delegate request to wrap
   */
  public Request createRequest(Request delegate) {
    Preconditions.checkState(!requestCreated, "a request was already created");
    return new MemoizingRequest(delegate) {
      @Override
      protected Runner createRunner(Request delegate) {
        return new CancellableRunner(delegate.getRunner());
      }
    };
  }

  /**
   * Cancels the request created by this request factory.
   */
  public void cancelRun() {
    cancelRequested = true;
    RunNotifier notifier = currentNotifier;
    if (notifier != null) {
      notifier.pleaseStop();
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
      if (cancelRequested) {
        currentNotifier.pleaseStop();
      }

      try {
        delegate.run(currentNotifier);
      } catch (StoppedByUserException e) {
        if (cancelRequested) {
          throw new RuntimeException("Test run interrupted", e);
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
}
