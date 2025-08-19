// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.devtools.build.lib.util.ThreadDumper;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import javax.annotation.Nullable;

/** A {@link BlazeModule} that dumps the state of all threads periodically. */
public final class ThreadDumpModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final DateTimeFormatter TIME_FORMAT =
      DateTimeFormatter.ofPattern("yyyyMMddHHmmss");

  @Nullable private Thread dumpThread;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    var commandOptions = env.getOptions().getOptions(CommonCommandOptions.class);
    if (commandOptions == null || !commandOptions.enableThreadDump) {
      return;
    }

    if (commandOptions.threadDumpInterval.isZero()) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_enable_thread_dump is set, but"
                      + " --experimental_thread_dump_interval is 0. No thread dumps will be"
                      + " written."));
      return;
    }

    var runtime = env.getRuntime();
    var clock = runtime.getClock();
    var threadDumpInterval = commandOptions.threadDumpInterval;

    var serverDirectory = runtime.getServerDirectory();
    var dumpDirectory = serverDirectory.getChild("thread_dumps");
    try {
      dumpDirectory.deleteTree();
      dumpDirectory.createDirectoryAndParents();
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
              FailureDetail.newBuilder()
                  .setMessage("Failed to setup thread dump directory")
                  .build()),
          e);
    }

    var pid = ProcessHandle.current().pid();
    checkState(dumpThread == null);
    dumpThread =
        new Thread(
            new ThreadDumpTask(pid, clock, new JavaSleeper(), threadDumpInterval, dumpDirectory),
            "thread-dumper");
    dumpThread.start();
  }

  @Override
  public void afterCommand() {
    if (dumpThread != null) {
      dumpThread.interrupt();
      try (var sc = Profiler.instance().profile("Joining dump thread")) {
        Uninterruptibles.joinUninterruptibly(dumpThread);
      }
      dumpThread = null;
    }
  }

  private static final class ThreadDumpTask implements Runnable {
    private final long pid;
    private final Clock clock;
    private final Sleeper sleeper;
    private final Duration threadDumpInterval;
    private final Path dumpDirectory;

    private ThreadDumpTask(
        long pid, Clock clock, Sleeper sleeper, Duration threadDumpInterval, Path dumpDirectory) {
      this.pid = pid;
      this.clock = clock;
      this.sleeper = sleeper;
      this.threadDumpInterval = threadDumpInterval;
      this.dumpDirectory = dumpDirectory;
    }

    @Override
    public void run() {
      while (true) {
        try {
          sleeper.sleepMillis(threadDumpInterval.toMillis());
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }

        String formattedTime =
            Instant.ofEpochMilli(clock.currentTimeMillis())
                .atZone(ZoneOffset.UTC)
                .format(TIME_FORMAT);
        Path dumpFile =
            dumpDirectory.getChild(String.format("thread_dump.%d.%s.txt", pid, formattedTime));
        try (var sc = Profiler.instance().profile("Dumping threads");
            var out = dumpFile.getOutputStream()) {
          ThreadDumper.dumpThreads(out);
        } catch (IOException e) {
          logger.atWarning().withCause(e).log("Failed to dump threads.");
        }
      }
    }
  }
}
