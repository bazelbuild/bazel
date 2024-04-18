// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.toCollection;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReporter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Utility to dump stack traces to logs and remotely log on slow interrupt. */
public class ThreadUtils {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final Comparator<
          Map.Entry<StackTraceAndState, ArrayList<Map.Entry<Thread, StackTraceElement[]>>>>
      STACK_SIZE_THEN_THREAD_COUNT_BOTH_DESCENDING_COMPARATOR =
          Map.Entry
              .<StackTraceAndState, ArrayList<Map.Entry<Thread, StackTraceElement[]>>>
                  comparingByKey()
              .thenComparingInt(o -> o.getValue().size())
              .reversed();
  private static final Collector<
          Map.Entry<Thread, StackTraceElement[]>,
          ?,
          Map<StackTraceAndState, ArrayList<Map.Entry<Thread, StackTraceElement[]>>>>
      MAP_WITH_ARRAY_LIST_VALUES_COLLECTOR =
          Collectors.groupingBy(StackTraceAndState::new, toCollection(ArrayList::new));

  private ThreadUtils() {
  }

  /** Write a thread dump to the blaze.INFO log if interrupt took too long. */
  public static synchronized void warnAboutSlowInterrupt(
      @Nullable String slowInterruptMessageSuffix) {
    warnAboutSlowInterrupt(slowInterruptMessageSuffix, BugReporter.defaultInstance());
  }

  @VisibleForTesting
  static synchronized void warnAboutSlowInterrupt(
      @Nullable String slowInterruptMessageSuffix, BugReporter bugReporter) {
    logger.atWarning().log("Interrupt took too long. Dumping thread state.");
    AtomicReference<StackTraceAndState> firstTrace = new AtomicReference<>();
    Thread.getAllStackTraces().entrySet().stream()
        .collect(MAP_WITH_ARRAY_LIST_VALUES_COLLECTOR)
        .entrySet()
        .stream()
        .sorted(STACK_SIZE_THEN_THREAD_COUNT_BOTH_DESCENDING_COMPARATOR)
        .forEach(
            e -> {
              StackTraceAndState stackTraceAndState = e.getKey();
              // Store longest trace but omit "Unsafe#park" and "Object#wait" calls: they are
              // interruptible, so can't be the cause of the slow interrupt. In some cases, we
              // do these calls in a loop on interrupt (see
              // AbstractQueueVisitor#reallyAwaitTermination) but unless there's a bug, there should
              // still be another thread waiting for interrupt somewhere.
              if (firstTrace.get() == null
                  && (!stackTraceAndState.trace[0].getClassName().endsWith("misc.Unsafe")
                      || !stackTraceAndState.trace[0].getMethodName().equals("park"))
                  && (!stackTraceAndState.trace[0].getClassName().endsWith("java.lang.Object")
                      || !stackTraceAndState.trace[0].getMethodName().equals("wait"))) {
                firstTrace.compareAndSet(null, stackTraceAndState);
              }
              logger.atWarning().log(
                  "%s %s%s",
                  stackTraceAndState.state,
                  makeThreadInfoString(e.getValue()),
                  makeString(stackTraceAndState.trace));
            });

    SlowInterruptInnerException inner =
        new SlowInterruptInnerException(
            Joiner.on(' ')
                .skipNulls()
                .join("(Wrapper exception for longest stack trace)", slowInterruptMessageSuffix));
    inner.setStackTrace(firstTrace.get().trace);
    SlowInterruptException ex = new SlowInterruptException(inner);
    bugReporter.sendBugReport(ex);
  }

  private static final class SlowInterruptException extends RuntimeException {
    public SlowInterruptException(SlowInterruptInnerException inner) {
      super("Slow interrupt", inner);
    }
  }

  private static final class StackTraceAndState implements Comparable<StackTraceAndState> {
    private final StackTraceElement[] trace;
    private final Thread.State state;

    StackTraceAndState(Map.Entry<Thread, StackTraceElement[]> threadEntry) {
      this.trace = threadEntry.getValue();
      this.state = threadEntry.getKey().getState();
    }

    @Override
    public int hashCode() {
      return 31 * state.hashCode() + Arrays.hashCode(trace);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof StackTraceAndState that)) {
        return false;
      }
      return Arrays.equals(this.trace, that.trace) && this.state.equals(that.state);
    }

    @Override
    public int compareTo(StackTraceAndState o) {
      return Integer.compare(trace.length, o.trace.length);
    }
  }

  private static String makeString(StackTraceElement[] stackTrace) {
    StringBuilder builder = new StringBuilder();
    for (StackTraceElement elt : stackTrace) {
      builder.append('\n').append('\t').append(elt);
    }
    return builder.toString();
  }

  private static final Comparator<Map.Entry<Thread, StackTraceElement[]>> THREAD_NAME_THEN_ID =
      Comparator.comparing((Map.Entry<Thread, StackTraceElement[]> o) -> o.getKey().getName())
          .thenComparingLong(o -> o.getKey().getId());

  private static String makeThreadInfoString(
      ArrayList<Map.Entry<Thread, StackTraceElement[]>> entries) {
    StringBuilder builder = new StringBuilder();
    if (entries.size() > 10) {
      builder.append(entries.size()).append(" threads, ");
    }
    entries.sort(THREAD_NAME_THEN_ID);
    boolean first = true;
    for (Map.Entry<Thread, StackTraceElement[]> entry : entries) {
      if (first) {
        first = false;
      } else {
        builder.append(", ");
      }
      Thread thread = entry.getKey();
      builder.append('<').append(thread.getName()).append(' ').append(thread.getId()).append('>');
    }
    return builder.toString();
  }

  private static class SlowInterruptInnerException extends Exception {
    SlowInterruptInnerException(String message) {
      super(message);
    }
  }
}
