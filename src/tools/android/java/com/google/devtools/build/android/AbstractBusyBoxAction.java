// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Abstract base class containing helper methods and error handling for BusyBox actions. */
public abstract class AbstractBusyBoxAction {
  private final JCommander jc;
  private final String description;
  private final Stopwatch timer = Stopwatch.createUnstarted();

  AbstractBusyBoxAction(JCommander jc, String description) {
    this.jc = jc;
    this.description = description;
  }

  public void invoke(String[] args) throws Exception {
    try {
      invokeWithoutExit(args);
    } catch (UserException | ParameterException e) {
      getLogger().severe(e.getMessage());
      // In Bazel, users tend to assume that a stack trace indicates a bug in underlying Bazel code
      // and ignore the content of the exception. If we know that the exception was actually their
      // fault, we  should just exit immediately rather than print a stack trace.
      System.exit(1);
    } catch (Exception e) {
      getLogger().log(Level.SEVERE, "Unexpected", e);
      throw e;
    }
  }

  /** Invokes the action without calling System.exit or catching exceptions, for use in testing */
  @VisibleForTesting
  void invokeWithoutExit(String[] args) throws Exception {
    timer.start();
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(this.jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(jc.getObjects(), preprocessedArgs);
    this.jc.parse(normalizedArgs);

    try (ScopedTemporaryDirectory scopedTmp = new ScopedTemporaryDirectory(description + "_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      run(scopedTmp.getPath(), executorService);
    }

    logCompletion(description);
  }

  abstract void run(Path tmp, ExecutorServiceCloser executorService) throws Exception;

  abstract Logger getLogger();

  <T> T getOptions(Class<T> clazz) {
    for (Object o : jc.getObjects()) {
      if (o.getClass().equals(clazz)) {
        return clazz.cast(o);
      }
    }

    return null;
  }

  /**
   * Logs that this action or some portion of it completed successfully.
   *
   * @param completedAction the action that was completed, for example "parsing". A timestamp will
   *     be appended to this string.
   */
  void logCompletion(String completedAction) {
    getLogger()
        .fine(String.format("%s finished at %sms", completedAction, timer.elapsed().toMillis()));
  }
}
