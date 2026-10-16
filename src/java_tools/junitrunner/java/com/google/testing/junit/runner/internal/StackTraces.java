// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.internal;

import com.sun.management.HotSpotDiagnosticMXBean;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.LockInfo;
import java.lang.management.ManagementFactory;
import java.lang.management.MonitorInfo;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Utilities for stack traces. */
public class StackTraces {

  /**
   * Prints all stack traces to the given stream.
   *
   * @param out Stream to print to
   */
  public static void printAll(PrintStream out) {
    printAll(out, /* emitJsonThreadDump= */ false);
  }

  /**
   * Prints all stack traces to the given stream.
   *
   * @param out Stream to print to
   * @param emitJsonThreadDump Whether to also emit a JSON thread dump to a file
   */
  public static void printAll(PrintStream out, boolean emitJsonThreadDump) {
    out.println("Starting full thread dump ...\n");
    ThreadMXBean mb = ManagementFactory.getThreadMXBean();

    // ThreadInfo has comprehensive information such as locks.
    ThreadInfo[] threadInfos = mb.dumpAllThreads(true, true);

    // But we can know whether a thread is daemon only from Thread
    Set<Thread> threads = Thread.getAllStackTraces().keySet();
    Map<Long, Thread> threadMap = new HashMap<>();
    for (Thread thread : threads) {
      threadMap.put(thread.getId(), thread);
    }

    // Dump non-daemon threads first
    for (ThreadInfo threadInfo : threadInfos) {
      Thread thread = threadMap.get(threadInfo.getThreadId());
      if (thread != null && !thread.isDaemon()) {
        dumpThreadInfo(threadInfo, thread, out);
      }
    }

    // Dump daemon threads
    for (ThreadInfo threadInfo : threadInfos) {
      Thread thread = threadMap.get(threadInfo.getThreadId());
      if (thread != null && thread.isDaemon()) {
        dumpThreadInfo(threadInfo, thread, out);
      }
    }

    long[] deadlockedThreads = mb.findDeadlockedThreads();
    if (deadlockedThreads != null) {
      out.println("Detected deadlocked threads: " + Arrays.toString(deadlockedThreads));
    }
    long[] monitorDeadlockedThreads = mb.findMonitorDeadlockedThreads();
    if (monitorDeadlockedThreads != null) {
      out.println(
          "Detected monitor deadlocked threads: " + Arrays.toString(monitorDeadlockedThreads));
    }
    out.println("\nDone full thread dump.");
    out.flush();

    if (!emitJsonThreadDump) {
      return;
    }
    // The thread dump above does not include virtual threads, so also capture a structured dump
    // that includes virtual threads. Since the dump is potentially large, we write it to a file in
    // the test outputs directory instead of printing it to the console.
    // Find the dumpThreads(String, ThreadDumpFormat) method via reflection to maintain
    // compatibility with JDKs prior to 21.
    Optional<Method> dumpThreadsMethod =
        Arrays.stream(HotSpotDiagnosticMXBean.class.getMethods())
            .filter(
                m ->
                    m.getName().equals("dumpThreads")
                        && m.getParameterCount() == 2
                        && m.getParameterTypes()[0] == String.class
                        && m.getParameterTypes()[1].isEnum())
            .findFirst();
    if (!dumpThreadsMethod.isPresent()) {
      return;
    }
    Optional<?> threadDumpFormatJson =
        Arrays.stream(dumpThreadsMethod.get().getParameterTypes()[1].getEnumConstants())
            .filter(e -> e.toString().equalsIgnoreCase("JSON"))
            .findFirst();
    if (!threadDumpFormatJson.isPresent()) {
      return;
    }
    HotSpotDiagnosticMXBean diagnosticBean =
        ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class);
    String testOutputsDir = System.getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (testOutputsDir == null) {
      return;
    }
    try {
      Path jsonDump = Files.createTempFile(Paths.get(testOutputsDir), "thread_dump", ".json");
      Files.delete(jsonDump);
      out.println("Writing JSON thread dump to " + jsonDump);
      dumpThreadsMethod
          .get()
          .invoke(diagnosticBean, jsonDump.toString(), threadDumpFormatJson.get());
      out.println("Done writing JSON thread dump to " + jsonDump);
    } catch (ReflectiveOperationException | IOException e) {
      out.println("Failed to write JSON thread dump:");
      e.printStackTrace(out);
    } finally {
      out.flush();
    }
  }

  // Adopted from ThreadInfo.toString(), without MAX_FRAMES limit
  private static void dumpThreadInfo(ThreadInfo t, Thread thread, PrintStream out) {
    out.print(
        "\"" + t.getThreadName() + "\"" + " Id=" + t.getThreadId() + " " + t.getThreadState());
    if (t.getLockName() != null) {
      out.print(" on " + t.getLockName());
    }
    if (t.getLockOwnerName() != null) {
      out.print(" owned by \"" + t.getLockOwnerName() + "\" Id=" + t.getLockOwnerId());
    }
    if (t.isSuspended()) {
      out.print(" (suspended)");
    }
    if (t.isInNative()) {
      out.print(" (in native)");
    }
    if (thread.isDaemon()) {
      out.print(" (daemon)");
    }
    out.print('\n');
    StackTraceElement[] stackTrace = t.getStackTrace();
    MonitorInfo[] lockedMonitors = t.getLockedMonitors();
    for (int i = 0; i < stackTrace.length; i++) {
      StackTraceElement ste = stackTrace[i];
      out.print("\tat " + ste.toString());
      out.print('\n');
      if (i == 0 && t.getLockInfo() != null) {
        Thread.State ts = t.getThreadState();
        switch (ts) {
          case BLOCKED:
            out.print("\t-  blocked on " + t.getLockInfo());
            out.print('\n');
            break;
          case WAITING:
            out.print("\t-  waiting on " + t.getLockInfo());
            out.print('\n');
            break;
          case TIMED_WAITING:
            out.print("\t-  waiting on " + t.getLockInfo());
            out.print('\n');
            break;
          default:
        }
      }

      for (MonitorInfo mi : lockedMonitors) {
        if (mi.getLockedStackDepth() == i) {
          out.print("\t-  locked " + mi);
          out.print('\n');
        }
      }
    }

    LockInfo[] locks = t.getLockedSynchronizers();
    if (locks.length > 0) {
      out.print("\n\tNumber of locked synchronizers = " + locks.length);
      out.print('\n');
      for (LockInfo li : locks) {
        out.print("\t- " + li);
        out.print('\n');
      }
    }
    out.print('\n');
  }
}
