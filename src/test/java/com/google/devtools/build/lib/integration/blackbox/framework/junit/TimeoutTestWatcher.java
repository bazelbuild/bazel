/*
 * // Copyright 2018 The Bazel Authors. All rights reserved.
 * //
 * // Licensed under the Apache License, Version 2.0 (the "License");
 * // you may not use this file except in compliance with the License.
 * // You may obtain a copy of the License at
 * //
 * // http://www.apache.org/licenses/LICENSE-2.0
 * //
 * // Unless required by applicable law or agreed to in writing, software
 * // distributed under the License is distributed on an "AS IS" BASIS,
 * // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * // See the License for the specific language governing permissions and
 * // limitations under the License.
 */

package com.google.devtools.build.lib.integration.blackbox.framework.junit;

import java.util.concurrent.TimeoutException;
import org.junit.rules.TestWatcher;
import org.junit.rules.Timeout;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

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
    Statement wrapper = new Statement() {
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

  private static class ExceptionWrapper extends Throwable {
    ExceptionWrapper(Throwable cause) {
      super(cause);
    }
  }
}
