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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionExecutionInactivityEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory.InvalidPackagePathSymlinkException;
import com.google.devtools.build.lib.runtime.InstrumentationOutputFactory.DestinationRelativeTo;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.ThreadDumpAnalyzer;
import com.google.devtools.build.lib.util.ThreadDumper;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import javax.annotation.Nullable;

/** A {@link BlazeModule} that dumps the state of all threads periodically. */
public final class ThreadDumpModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final DateTimeFormatter TIME_FORMAT =
      DateTimeFormatter.ofPattern("yyyyMMddHHmmss");

  @Nullable private ScheduledExecutorService scheduledExecutor;
  @Nullable private ThreadDumpTask threadDumpTask;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    var commandOptions = env.getOptions().getOptions(CommonCommandOptions.class);
    if (commandOptions == null || !commandOptions.enableThreadDump) {
      return;
    }

    if (commandOptions.threadDumpInterval.isZero()
        && commandOptions.threadDumpActionExecutionInactivityDuration.isZero()) {
      env.getReporter()
          .handle(
              Event.warn(
                  "--experimental_enable_thread_dump is set, but"
                      + " --experimental_thread_dump_interval and"
                      + " --experimental_thread_dump_action_execution_inactivity_duration are 0. No"
                      + " thread dumps will be written."));
      return;
    }

    checkState(threadDumpTask == null);
    checkState(scheduledExecutor == null);

    var bepOptions = env.getOptions().getOptions(BuildEventProtocolOptions.class);
    BuildEventArtifactUploader uploader = null;
    if (bepOptions != null && bepOptions.streamingLogFileUploads) {
      try {
        uploader = newUploader(env, bepOptions);
      } catch (InvalidPackagePathSymlinkException e) {
        throw createAbruptExitException("Failed to create uploader", e);
      }
    }

    var outputBaseRelativeDumpDirectory = prepareDumpDirectory(env);
    threadDumpTask =
        new ThreadDumpTask(
            env,
            ProcessHandle.current().pid(),
            env.getRuntime().getClock(),
            outputBaseRelativeDumpDirectory,
            commandOptions.threadDumpActionExecutionInactivityDuration,
            uploader);

    scheduledExecutor = Executors.newSingleThreadScheduledExecutor();

    var threadDumpInterval = commandOptions.threadDumpInterval;
    if (!threadDumpInterval.isZero()) {
      var unused =
          scheduledExecutor.scheduleAtFixedRate(
              threadDumpTask,
              threadDumpInterval.toMillis(),
              threadDumpInterval.toMillis(),
              MILLISECONDS);
    }

    env.getEventBus().register(this);
  }

  private static BuildEventArtifactUploader newUploader(
      CommandEnvironment env, BuildEventProtocolOptions bepOptions)
      throws InvalidPackagePathSymlinkException {
    return env.getRuntime()
        .getBuildEventArtifactUploaderFactoryMap()
        .select(bepOptions.buildEventUploadStrategy)
        .create(env);
  }

  private PathFragment prepareDumpDirectory(CommandEnvironment env) throws AbruptExitException {
    var runtime = env.getRuntime();
    var serverDirectory = runtime.getServerDirectory();
    var dumpDirectory = serverDirectory.getChild("thread_dumps");
    try {
      dumpDirectory.deleteTree();
      dumpDirectory.createDirectoryAndParents();
    } catch (IOException e) {
      throw createAbruptExitException("Failed to setup thread dump directory", e);
    }
    return dumpDirectory.relativeTo(env.getDirectories().getOutputBase());
  }

  private static AbruptExitException createAbruptExitException(String message, Throwable cause) {
    return new AbruptExitException(
        DetailedExitCode.of(
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
            FailureDetail.newBuilder().setMessage(message).build()),
        cause);
  }

  @Subscribe
  public void onActionExecutionInactivityEvent(ActionExecutionInactivityEvent event) {
    checkState(scheduledExecutor != null);
    checkState(threadDumpTask != null);

    if (threadDumpTask.shouldDumpForActionExecutionInactivity(event)) {
      var unused = scheduledExecutor.schedule(threadDumpTask, 0, MILLISECONDS);
    }
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    checkNotNull(scheduledExecutor);
    checkNotNull(threadDumpTask);

    scheduledExecutor.shutdownNow();
    try (var sc = Profiler.instance().profile("Joining dump thread")) {
      Uninterruptibles.awaitTerminationUninterruptibly(scheduledExecutor);
    }

    var buildToolLogCollection = event.getResult().getBuildToolLogCollection();
    threadDumpTask.shutdown(buildToolLogCollection);

    scheduledExecutor = null;
    threadDumpTask = null;
  }

  private static final class ThreadDumpTask implements Runnable {
    private final CommandEnvironment env;
    private final long pid;
    private final Clock clock;
    private final PathFragment outputBaseRelativeDumpDirectory;
    private final Duration dumpActionExecutionInactivityDuration;
    @Nullable private final BuildEventArtifactUploader uploader;

    private final List<InstrumentationOutput> instrumentationOutputs =
        Collections.synchronizedList(new ArrayList<>());

    private Instant lastDumpAt = Instant.EPOCH;

    private ThreadDumpTask(
        CommandEnvironment env,
        long pid,
        Clock clock,
        PathFragment outputBaseRelativeDumpDirectory,
        Duration dumpActionExecutionInactivityDuration,
        @Nullable BuildEventArtifactUploader uploader) {
      this.env = env;
      this.pid = pid;
      this.clock = clock;
      this.outputBaseRelativeDumpDirectory = outputBaseRelativeDumpDirectory;
      this.dumpActionExecutionInactivityDuration = dumpActionExecutionInactivityDuration;
      this.uploader = uploader;
    }

    @Override
    public void run() {
      var bos = new ByteArrayOutputStream();
      try (var sc = Profiler.instance().profile("Dumping threads")) {
        ThreadDumper.dumpThreads(bos);
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to dump threads.");
      }

      String formattedTime =
          Instant.ofEpochMilli(clock.currentTimeMillis())
              .atZone(ZoneOffset.UTC)
              .format(TIME_FORMAT);
      var dumpOutput =
          createThreadDumpOutput(String.format("thread_dump.%d.%s.txt", pid, formattedTime));
      instrumentationOutputs.add(dumpOutput);
      var analyzer = new ThreadDumpAnalyzer();
      try (var sc = Profiler.instance().profile("Analyzing thread dump");
          var out = dumpOutput.createOutputStream()) {
        analyzer.analyze(new ByteArrayInputStream(bos.toByteArray()), out);
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Failed to analyze threads.");
      }

      lastDumpAt = clock.now();
    }

    boolean shouldDumpForActionExecutionInactivity(ActionExecutionInactivityEvent event) {
      if (dumpActionExecutionInactivityDuration.isZero()) {
        return false;
      }

      var now = clock.now();
      if (now.isBefore(event.lastActionCompletedAt().plus(dumpActionExecutionInactivityDuration))) {
        return false;
      }

      return now.isAfter(lastDumpAt.plus(dumpActionExecutionInactivityDuration));
    }

    void shutdown(BuildToolLogCollection buildToolLogCollection) {
      for (var output : instrumentationOutputs) {
        output.publish(buildToolLogCollection);
      }

      if (uploader != null) {
        uploader.release();
      }
    }

    private InstrumentationOutput createThreadDumpOutput(String name) {
      var outputFactory = env.getRuntime().getInstrumentationOutputFactory();
      if (uploader != null) {
        return outputFactory.createBuildEventArtifactInstrumentationOutput(name, uploader);
      }
      return outputFactory.createInstrumentationOutput(
          /* name= */ name,
          /* destination= */ outputBaseRelativeDumpDirectory.getRelative(name),
          DestinationRelativeTo.OUTPUT_BASE,
          env,
          env.getReporter(),
          /* append= */ null,
          /* internal= */ null,
          /* createParent= */ true);
    }
  }
}
