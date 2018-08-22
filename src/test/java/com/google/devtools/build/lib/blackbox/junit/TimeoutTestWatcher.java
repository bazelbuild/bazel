// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.junit;

import java.util.concurrent.TimeoutException;
import org.junit.rules.TestWatcher;
import org.junit.rules.Timeout;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/**
 * Test watcher, which sets a timeout for the JUnit test and allows to execute some action on
 * timeout. Uses JUnit's org.junit.rules.Timeout rule to set up a timeout; catches timeout exception
 * thrown fromTimeout rule, calls the {@link onTimeout} method, and re-throws the exception.
 *
 * <p>Useful to dump test state information before failing on timeout.
 */
public abstract class TimeoutTestWatcher extends TestWatcher {
  private String name;

  protected abstract long getTimeoutMillis();

  protected abstract boolean onTimeout();

  @Override
  protected void starting(Description description) {
    name = description.getMethodName();
  }

  @Override
  protected void finished(Description description) {
    name = null;
  }

  public String getName() {
    return name;
  }

  @Override
  public Statement apply(Statement base, Description description) {
    // we are using exception wrapping, because unfortunately JUnit's Timeout throws
    // java.util.Exception on timeout, which is hard to distinguish from other cases
    Statement wrapper =
        new Statement() {
          @Override
          public void evaluate() throws Throwable {
            try {
              base.evaluate();
            } catch (Throwable th) {
              throw new ExceptionWrapper(th);
            }
          }
        };

    return new Statement() {
      @Override
      public void evaluate() throws Throwable {
        try {
          new Timeout((int) getTimeoutMillis()).apply(wrapper, description).evaluate();
        } catch (ExceptionWrapper wrapper) {
          // original test exception
          throw wrapper.getCause();
        } catch (Exception e) {
          // timeout exception
          if (!onTimeout()) {
            throw new TimeoutException(e.getMessage());
          }
        }
      }
    };
  }

  /**
   * Exception wrapper wrap-and-caught any exception from the test; this guarantees that we
   * differentiate timeout exception thrown just as java.util.Exception from the test exceptions
   */
  private static class ExceptionWrapper extends Throwable {
    ExceptionWrapper(Throwable cause) {
      super(cause);
    }
  }
}
