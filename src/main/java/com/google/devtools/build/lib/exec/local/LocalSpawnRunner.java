// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.local;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.logging.Level.INFO;
import static java.util.logging.Level.SEVERE;

import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * A class that runs local commands. Each request follows state transitions from "parsing" to
 * completion.
 */
@ThreadSafe
public final class LocalSpawnRunner implements SpawnRunner {
  private static final Joiner SPACE_JOINER = Joiner.on(' ');
  private static final String UNHANDLED_EXCEPTION_MSG = "Unhandled exception running a local spawn";
  private static final int LOCAL_EXEC_ERROR = -1;
  private static final int POSIX_TIMEOUT_EXIT_CODE = /*SIGNAL_BASE=*/128 + /*SIGALRM=*/14;

  private static final Logger logger = Logger.getLogger(LocalSpawnRunner.class.getName());

  private final Path execRoot;
  private final ResourceManager resourceManager;

  private final String hostName;

  private final LocalExecutionOptions localExecutionOptions;

  private final boolean useProcessWrapper;
  private final String processWrapper;

  private final String productName;
  private final LocalEnvProvider localEnvProvider;

  private static Path getProcessWrapper(Path execRoot, OS localOs) {
    return execRoot.getRelative("_bin/process-wrapper" + OsUtils.executableExtension(localOs));
  }

  public LocalSpawnRunner(
      Path execRoot,
      LocalExecutionOptions localExecutionOptions,
      ResourceManager resourceManager,
      boolean useProcessWrapper,
      OS localOs,
      String productName,
      LocalEnvProvider localEnvProvider) {
    this.execRoot = execRoot;
    this.processWrapper = getProcessWrapper(execRoot, localOs).getPathString();
    this.localExecutionOptions = Preconditions.checkNotNull(localExecutionOptions);
    this.hostName = NetUtil.findShortHostName();
    this.resourceManager = resourceManager;
    this.useProcessWrapper = useProcessWrapper;
    this.productName = productName;
    this.localEnvProvider = localEnvProvider;
  }

  public LocalSpawnRunner(
      Path execRoot,
      LocalExecutionOptions localExecutionOptions,
      ResourceManager resourceManager,
      String productName,
      LocalEnvProvider localEnvProvider) {
    this(
        execRoot,
        localExecutionOptions,
        resourceManager,
        OS.getCurrent() != OS.WINDOWS && getProcessWrapper(execRoot, OS.getCurrent()).exists(),
        OS.getCurrent(),
        productName,
        localEnvProvider);
  }

  @Override
  public SpawnResult exec(
      Spawn spawn,
      SpawnExecutionPolicy policy) throws IOException, InterruptedException {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    policy.report(ProgressStatus.SCHEDULING, "local");
    try (ResourceHandle handle =
        resourceManager.acquireResources(owner, spawn.getLocalResources())) {
      policy.report(ProgressStatus.EXECUTING, "local");
      policy.lockOutputFiles();
      return new SubprocessHandler(spawn, policy).run();
    }
  }

  private final class SubprocessHandler {
    private final Spawn spawn;
    private final SpawnExecutionPolicy policy;

    private final long creationTime = System.currentTimeMillis();
    private long stateStartTime = creationTime;
    private State currentState = State.INITIALIZING;
    private final Map<State, Long> stateTimes = new EnumMap<>(State.class);

    private final int id;

    public SubprocessHandler(
        Spawn spawn,
        SpawnExecutionPolicy policy) {
      Preconditions.checkArgument(!spawn.getArguments().isEmpty());
      this.spawn = spawn;
      this.policy = policy;
      this.id = policy.getId();
      setState(State.PARSING);
    }

    public SpawnResult run() throws InterruptedException, IOException {
      try {
        return start();
      } catch (Error e) {
        stepLog(SEVERE, UNHANDLED_EXCEPTION_MSG, e);
        throw e;
      } catch (IOException e) {
        stepLog(SEVERE, "Local I/O error", e);
        throw e;
      } catch (RuntimeException e) {
        stepLog(SEVERE, UNHANDLED_EXCEPTION_MSG, e);
        throw new RuntimeException(UNHANDLED_EXCEPTION_MSG, e);
      }
    }

    private void stepLog(Level level, String fmt, Object... args) {
      stepLog(level, fmt, /*cause=*/ null, args);
    }

    @SuppressWarnings("unused")
    private void stepLog(Level level, String fmt, @Nullable Throwable cause, Object... args) {
      String msg = String.format(fmt, args);
      String toLog = String.format("%s (#%d %s)", msg, id, desc());
      logger.log(level, toLog, cause);
    }

    private String desc() {
      String progressMessage = spawn.getResourceOwner().getProgressMessage();
      return progressMessage != null
          ? progressMessage
          : "ActionType=" + spawn.getResourceOwner().getMnemonic();
    }

    private void setState(State newState) {
      long now = System.currentTimeMillis();
      long totalDelta = now - creationTime;
      long stepDelta = now - stateStartTime;
      stateStartTime = now;

      Long stateTimeBoxed = stateTimes.get(currentState);
      long stateTime = (stateTimeBoxed == null) ? 0 : stateTimeBoxed;
      stateTimes.put(currentState, stateTime + stepDelta);

      logger.info(String.format(
          "Step #%d time: %.3f delta: %.3f state: %s --> %s",
          id, totalDelta / 1000f, stepDelta / 1000f, currentState, newState));
      currentState = newState;
    }

    private String debugCmdString() {
      String cmd = SPACE_JOINER.join(spawn.getArguments());
      if (cmd.length() > 500) {
        // Shrink argstr by replacing middle of string with "...".
        return cmd.substring(0, 250) + "..." + cmd.substring(cmd.length() - 250);
      }
      return cmd;
    }

    /** Parse the request and run it locally. */
    private SpawnResult start() throws InterruptedException, IOException {
      logger.info(String.format("starting local subprocess #%d, argv: %s", id, debugCmdString()));

      FileOutErr outErr = policy.getFileOutErr();
      String actionType = spawn.getResourceOwner().getMnemonic();
      if (localExecutionOptions.allowedLocalAction != null
          && !localExecutionOptions.allowedLocalAction.matcher(actionType).matches()) {
        setState(State.PERMANENT_ERROR);
        outErr.getErrorStream().write(
            ("Action type " + actionType + " is not allowed to run locally due to regex filter: "
                + localExecutionOptions.allowedLocalAction + "\n").getBytes(UTF_8));
        return new SpawnResult.Builder()
            .setStatus(Status.LOCAL_ACTION_NOT_ALLOWED)
            .setExitCode(LOCAL_EXEC_ERROR)
            .setExecutorHostname(hostName)
            .build();
      }

      if (Spawns.shouldPrefetchInputsForLocalExecution(spawn)) {
        stepLog(INFO, "prefetching inputs for local execution");
        setState(State.PREFETCHING_LOCAL_INPUTS);
        policy.prefetchInputs(
            Iterables.filter(policy.getInputMapping().values(), Predicates.notNull()));
      }

      stepLog(INFO, "running locally");
      setState(State.LOCAL_ACTION_RUNNING);

      Command cmd;
      OutputStream stdOut = ByteStreams.nullOutputStream();
      OutputStream stdErr = ByteStreams.nullOutputStream();
      if (useProcessWrapper) {
        // If the process wrapper is enabled, we use its timeout feature, which first interrupts the
        // subprocess and only kills it after a grace period so that the subprocess can output a
        // stack trace, test log or similar, which is incredibly helpful for debugging. The process
        // wrapper also supports output file redirection, so we don't need to stream the output
        // through this process.
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(processWrapper);
        cmdLine.add("--timeout=" + policy.getTimeout().getSeconds());
        cmdLine.add("--kill_delay=" + localExecutionOptions.localSigkillGraceSeconds);
        cmdLine.add("--stdout=" + getPathOrDevNull(outErr.getOutputPath()));
        cmdLine.add("--stderr=" + getPathOrDevNull(outErr.getErrorPath()));
        cmdLine.addAll(spawn.getArguments());
        cmd = new Command(
            cmdLine.toArray(new String[0]),
            localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, productName),
            execRoot.getPathFile());
      } else {
        stdOut = outErr.getOutputStream();
        stdErr = outErr.getErrorStream();
        cmd = new Command(
            spawn.getArguments().toArray(new String[0]),
            localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, productName),
            execRoot.getPathFile(),
            policy.getTimeout());
      }

      long startTime = System.currentTimeMillis();
      CommandResult result;
      try {
        result = cmd.execute(stdOut, stdErr);
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException();
        }
      } catch (AbnormalTerminationException e) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException();
        }
        result = e.getResult();
      } catch (CommandException e) {
        // At the time this comment was written, this must be a ExecFailedException encapsulating an
        // IOException from the underlying Subprocess.Factory.
        String msg = e.getMessage() == null ? e.getClass().getName() : e.getMessage();
        setState(State.PERMANENT_ERROR);
        outErr.getErrorStream().write(("Action failed to execute: " + msg + "\n").getBytes(UTF_8));
        outErr.getErrorStream().flush();
        return new SpawnResult.Builder()
            .setStatus(Status.EXECUTION_FAILED)
            .setExitCode(LOCAL_EXEC_ERROR)
            .setExecutorHostname(hostName)
            .build();
      }
      setState(State.SUCCESS);

      long wallTimeMillis = System.currentTimeMillis() - startTime;
      boolean wasTimeout = result.getTerminationStatus().timedout()
          || (useProcessWrapper && wasTimeout(policy.getTimeout(), wallTimeMillis));
      Status status = wasTimeout ? Status.TIMEOUT : Status.SUCCESS;
      int exitCode = status == Status.TIMEOUT
          ? POSIX_TIMEOUT_EXIT_CODE
          : result.getTerminationStatus().getRawExitCode();
      return new SpawnResult.Builder()
          .setStatus(status)
          .setExitCode(exitCode)
          .setExecutorHostname(hostName)
          .setWallTimeMillis(wallTimeMillis)
          .build();
    }

    private String getPathOrDevNull(Path path) {
      return path == null ? "/dev/null" : path.getPathString();
    }

    private boolean wasTimeout(Duration timeout, long wallTimeMillis) {
      return !timeout.isZero() && wallTimeMillis > timeout.toMillis();
    }
  }

  private enum State {
    INITIALIZING,
    PARSING,
    PREFETCHING_LOCAL_INPUTS,
    LOCAL_ACTION_RUNNING,
    PERMANENT_ERROR,
    SUCCESS;
  }
}
