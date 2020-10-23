// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.base.Throwables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.ThrowableOrBuilder;
import java.util.Set;
import java.util.stream.Collectors;

/** Factory methods for producing {@link Crash}-type {@link FailureDetail} messages. */
public class CrashFailureDetails {

  /**
   * At most this many {@link FailureDetails.Throwable} messages will be specified by a {@link
   * Crash} submessage.
   */
  private static final int MAX_CAUSE_CHAIN_SIZE = 5;

  /**
   * At most this many stack trace element strings will be specified by a {@link
   * FailureDetails.Throwable} submessage.
   */
  private static final int MAX_STACK_TRACE_SIZE = 1000;

  private CrashFailureDetails() {}

  public static DetailedExitCode detailedExitCodeForThrowable(Throwable throwable) {
    return DetailedExitCode.of(forThrowable(throwable));
  }

  /**
   * Returns a {@link Crash}-type {@link FailureDetail} with {@link Crash.Code#CRASH_UNKNOWN}, with
   * its cause chain filled out.
   */
  public static FailureDetail forThrowable(Throwable throwable) {
    Crash.Builder crashBuilder =
        Crash.newBuilder()
            .setCode(
                (getRootCauseToleratingCycles(throwable) instanceof OutOfMemoryError)
                    ? Crash.Code.CRASH_OOM
                    : Crash.Code.CRASH_UNKNOWN);
    addCause(crashBuilder, throwable, Sets.newIdentityHashSet());
    return FailureDetail.newBuilder()
        .setMessage("Crashed: " + joinSummarizedCauses(crashBuilder))
        .setCrash(crashBuilder)
        .build();
  }

  private static String joinSummarizedCauses(Crash.Builder crashBuilder) {
    return crashBuilder.getCausesOrBuilderList().stream()
        .map(CrashFailureDetails::summarizeCause)
        .collect(Collectors.joining(", "));
  }

  private static String summarizeCause(ThrowableOrBuilder throwableOrBuilder) {
    return String.format(
        "(%s) %s", throwableOrBuilder.getThrowableClass(), throwableOrBuilder.getMessage());
  }

  private static void addCause(
      Crash.Builder crashBuilder, Throwable throwable, Set<Object> addedThrowables) {
    addedThrowables.add(throwable);

    crashBuilder.addCauses(getThrowable(throwable));

    Throwable cause = throwable.getCause();
    if (cause == null
        || addedThrowables.contains(cause)
        || crashBuilder.getCausesOrBuilderList().size() >= MAX_CAUSE_CHAIN_SIZE) {
      return;
    }
    addCause(crashBuilder, cause, addedThrowables);
  }

  private static FailureDetails.Throwable getThrowable(Throwable throwable) {
    String throwableMessage = throwable.getMessage() != null ? throwable.getMessage() : "";
    FailureDetails.Throwable.Builder throwableBuilder =
        FailureDetails.Throwable.newBuilder()
            .setMessage(throwableMessage)
            .setThrowableClass(throwable.getClass().getName());
    StackTraceElement[] stackTrace = throwable.getStackTrace();
    for (StackTraceElement stackTraceElement : stackTrace) {
      if (throwableBuilder.getStackTraceList().size() >= MAX_STACK_TRACE_SIZE) {
        break;
      }
      throwableBuilder.addStackTrace(stackTraceElement.toString());
    }
    return throwableBuilder.build();
  }

  /**
   * Returns the innermost cause of {@code throwable}. The first throwable in a chain provides
   * context from when the error or exception was initially detected. Example usage:
   *
   * <pre>
   * assertEquals("Unable to assign a customer id", Throwables.getRootCause(e).getMessage());
   * </pre>
   *
   * Cloned from {@link Throwables#getRootCause} with a modification to return an arbitrary element
   * of the cycle rather than throw if there is a causal cycle.
   */
  private static Throwable getRootCauseToleratingCycles(Throwable throwable) {
    // Keep a second pointer that slowly walks the causal chain. If the fast pointer ever catches
    // the slower pointer, then there's a loop.
    Throwable slowPointer = throwable;
    boolean advanceSlowPointer = false;

    Throwable cause;
    while ((cause = throwable.getCause()) != null) {
      throwable = cause;

      if (throwable == slowPointer) {
        // There's a cycle: choose an arbitrary element in that cycle.
        return throwable;
      }
      if (advanceSlowPointer) {
        slowPointer = slowPointer.getCause();
      }
      advanceSlowPointer = !advanceSlowPointer; // only advance every other iteration
    }
    return throwable;
  }
}
