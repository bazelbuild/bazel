// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.junit4.runner;

import org.junit.runner.Request;
import org.junit.runner.Runner;

/**
 * A {@link Request} that memoizies another {@code Request}.
 * This class is meant to be overridden to modify some behaviors.
 */
@Deprecated
public class MemoizingRequest extends Request {
  private final Request requestDelegate;
  private Runner runnerDelegate;

  public MemoizingRequest(Request delegate) {
    this.requestDelegate = delegate;
  }

  @Override
  public final synchronized Runner getRunner() {
    if (runnerDelegate == null) {
      runnerDelegate = createRunner(requestDelegate);
    }
    return runnerDelegate;
  }

  /**
   * Creates the runner. This method is called at most once.
   * Subclasses can override this method for different behavior.
   * The default implementation returns the runner created by the delegate.
   *
   * @param delegate request to delegate to
   * @return runner
   */
  protected Runner createRunner(Request delegate) {
    return delegate.getRunner();
  }
}
