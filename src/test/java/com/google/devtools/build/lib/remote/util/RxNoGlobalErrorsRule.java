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

import io.reactivex.rxjava3.exceptions.CompositeException;
import io.reactivex.rxjava3.plugins.RxJavaPlugins;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.rules.ExternalResource;

/**
 * A JUnit {@link org.junit.Rule} that captures uncaught errors from RxJava streams and rethrows
 * them post-test if left unaddressed.
 *
 * <p>This is to prevent false-positives caused by RxJava's default uncaught error handler, which
 * manually forwards the event to the current Thread's exception handler and bypasses JUnit's
 * failure reporting.
 *
 * <p>Can also be used to assert that no uncaught errors have yet been thrown mid-test. This is
 * useful to ensure tests are in a consistent state before continuing.
 */
public class RxNoGlobalErrorsRule extends ExternalResource {
  private final List<Throwable> errors = new CopyOnWriteArrayList<>();

  @Override
  protected void before() {
    RxJavaPlugins.setErrorHandler(errors::add);
  }

  @Override
  protected void after() {
    assertNoErrors();
  }

  private static final class UncaughtRxErrors extends RuntimeException {
    private UncaughtRxErrors(Throwable cause) {
      super("There were uncaught Rx errors during test execution", cause);
    }
  }

  /**
   * Asserts that no uncaught errors have yet occurred.
   *
   * <p>If an Rx stream has thrown an uncaught error any time before this method is called, an
   * {@link UncaughtRxErrors} is thrown. This is useful for ensuring that tests are in a consistent
   * state before continuing.
   *
   * <p>You may need to advance any test schedulers so that any pending events are flushed.
   */
  private void assertNoErrors() {
    if (errors.size() > 1) {
      Throwable[] errorsArray = errors.toArray(new Throwable[0]);
      throw new UncaughtRxErrors(new CompositeException(errorsArray));
    } else if (errors.size() == 1) {
      throw new UncaughtRxErrors(errors.get(0));
    }
  }
}
