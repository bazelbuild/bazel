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

import com.google.common.collect.Maps;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.exec.ActionInputPrefetcher;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * A class that runs local commands. Each request follows state transitions from "parsing" to
 * completion.
 */
@ThreadSafe
public final class LocalSpawnRunner implements SpawnRunner {
  private static final String UNHANDLED_EXCEPTION_MSG = "Unhandled exception running a local spawn";
  private static final int LOCAL_EXEC_ERROR = -1;

  private final Path execRoot;
  private final ResourceManager resourceManager;

  private final String hostName;

  private final ActionInputPrefetcher actionInputPrefetcher;

  private final LocalExecutionOptions localExecutionOptions;

  private final boolean useProcessWrapper;
  private final String processWrapper;

  public LocalSpawnRunner(
      Path execRoot,
      ActionInputPrefetcher actionInputPrefetcher,
      LocalExecutionOptions localExecutionOptions,
      ResourceManager resourceManager,
      boolean useProcessWrapper) {
    this.execRoot = execRoot;
    this.actionInputPrefetcher = Preconditions.checkNotNull(actionInputPrefetcher);
    this.processWrapper = execRoot.getRelative("_bin/process-wrapper").getPathString();
    this.localExecutionOptions = localExecutionOptions;
    this.hostName = NetUtil.findShortHostName();
    this.resourceManager = resourceManager;
    this.useProcessWrapper = useProcessWrapper;
  }

  public LocalSpawnRunner(
      Path execRoot,
      ActionInputPrefetcher actionInputPrefetcher,
      LocalExecutionOptions localExecutionOptions,
      ResourceManager resourceManager) {
    this(
        execRoot,
        actionInputPrefetcher,
        localExecutionOptions,
        resourceManager,
        // TODO(bazel-team): process-wrapper seems to work on Windows, but requires additional setup
        // as it is an msys2 binary, so it needs msys2 DLLs on %PATH%. Disable it for now to make
        // the setup easier and to avoid further PATH hacks. Ideally we should have a native
        // implementation of process-wrapper for Windows.
        OS.getCurrent() != OS.WINDOWS);
  }

  @Override
  public SpawnResult exec(
      Spawn spawn,
      SpawnExecutionPolicy policy) throws IOException, InterruptedException {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    if (owner.getOwner() != null) {
      policy.report(ProgressStatus.SCHEDULING);
    }
    try (ResourceHandle handle =
        resourceManager.acquireResources(owner, spawn.getLocalResources())) {
      policy.lockOutputFiles();
      if (owner.getOwner() != null) {
        policy.report(ProgressStatus.EXECUTING);
      }
      return new SubprocessHandler(
          spawn,
          policy.shouldPrefetchInputsForLocalExecution(spawn),
          policy.getFileOutErr(),
          policy.getTimeoutMillis() / 1000.0f,
          policy).run();
    }
  }

  private final class SubprocessHandler {
    private final List<String> args;

    private final Map<String, String> env;
    private final FileOutErr outErr;
    private final SortedMap<PathFragment, ActionInput> inputMappings;
    private final boolean localPrefetch;

    private final float timeout;
    private final String actionType;

    private final long creationTime = System.currentTimeMillis();
    private long stateStartTime = creationTime;
    private State currentState = State.INITIALIZING;
    private Map<State, Long> stateTimes = new EnumMap<>(State.class);

    public SubprocessHandler(
        Spawn spawn,
        boolean localPrefetch,
        FileOutErr outErr,
        float timeout,
        SpawnExecutionPolicy policy) throws IOException {
      this.args = spawn.getArguments();
      this.env = Maps.newHashMap(spawn.getEnvironment());
      this.timeout = timeout;

      this.actionType = spawn.getResourceOwner().getMnemonic();

      this.inputMappings = policy.getInputMapping();
      this.localPrefetch = localPrefetch;
      this.outErr = outErr;
      setState(State.PARSING);
    }

    public SpawnResult run() throws InterruptedException, IOException {
      try {
        return start();
      } catch (Error e) {
        stepLog(SEVERE, UNHANDLED_EXCEPTION_MSG, e);
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
      // Do nothing for now.
    }

    private void setState(State newState) {
      long now = System.currentTimeMillis();
      long stepDelta = now - stateStartTime;
      stateStartTime = now;

      Long stateTimeBoxed = stateTimes.get(currentState);
      long stateTime = (stateTimeBoxed == null) ? 0 : stateTimeBoxed;
      stateTimes.put(currentState, stateTime + stepDelta);

      currentState = newState;
    }

    /** Parse the request and run it locally. */
    private SpawnResult start() throws InterruptedException, IOException {
      Preconditions.checkArgument(!args.isEmpty());

      // The old code was dumping action inputs even for local execution.
      // See distributorOptions.dumpActionInputsRegex.

      stepLog(INFO, "prefetching inputs for local execution");
      setState(State.PREFETCHING_LOCAL_INPUTS);

      if (localPrefetch) {
        actionInputPrefetcher.prefetchFiles(inputMappings.values());
      }

      stepLog(INFO, "running locally");
      setState(State.LOCAL_ACTION_RUNNING);

      if (localExecutionOptions.allowedLocalAction != null
          && !localExecutionOptions.allowedLocalAction.matcher(actionType).matches()) {
        return completeWithError(
            "Action type " + actionType + " is not allowed to run locally "
            + "due to regex filter: " + localExecutionOptions.allowedLocalAction,
            LOCAL_EXEC_ERROR,
            Status.LOCAL_ACTION_NOT_ALLOWED);
      }

      Command cmd;
      OutputStream stdOut = ByteStreams.nullOutputStream();
      OutputStream stdErr = ByteStreams.nullOutputStream();
      if (useProcessWrapper) {
        List<String> cmdLine = new ArrayList<>();
        cmdLine.add(processWrapper);
        cmdLine.add(Float.toString(timeout));
        cmdLine.add(Double.toString(localExecutionOptions.localSigkillGraceSeconds));
        cmdLine.add(getPathOrDevNull(outErr.getOutputPath()));
        cmdLine.add(getPathOrDevNull(outErr.getErrorPath()));
        cmdLine.addAll(args);
        cmd = new Command(cmdLine.toArray(new String[]{}), env, execRoot.getPathFile());
      } else {
        stdOut = outErr.getOutputStream();
        stdErr = outErr.getErrorStream();
        cmd = new Command(
            args.toArray(new String[0]),
            env,
            execRoot.getPathFile(),
            (int) timeout);
      }

      long startTime = System.currentTimeMillis();
      CommandResult result;
      try {
        result = cmd.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdOut, stdErr, true);
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
        return completeWithError(
            "Action failed to execute: " + msg,
            LOCAL_EXEC_ERROR,
            Status.EXECUTION_FAILED);
      }
      return complete(
          condense(result.getTerminationStatus()),
          wasTimeout(timeout, startTime) ? Status.TIMEOUT : Status.SUCCESS,
          System.currentTimeMillis() - startTime);
    }

    private String getPathOrDevNull(Path path) {
      return path == null ? "/dev/null" : path.getPathString();
    }

    private boolean wasTimeout(float timeout, long startTime) {
      return timeout > 0 && (System.currentTimeMillis() - startTime) / 1000.0 > timeout;
    }

    private SpawnResult completeWithError(
        String msg, int exitCode, Status status) throws IOException {
      setState(State.PERMANENT_ERROR);
      writeToStdErr(msg);
      SpawnResult.Builder builder = new SpawnResult.Builder();
      builder.setStatus(status);
      builder.setExitCode(exitCode);
      builder.setExecutorHostname(hostName);
      return builder.build();
    }

    private void writeToStdErr(String message) throws IOException {
      outErr.getErrorStream().write((message + "\n").getBytes(UTF_8));
    }

    private SpawnResult complete(int exitCode, Status status, long wallTimeMillis) {
      setState(State.SUCCESS);

      SpawnResult.Builder builder = new SpawnResult.Builder();
      builder.setStatus(status);
      builder.setExitCode(
          status == Status.TIMEOUT ? /*SIGNAL_BASE=*/128 + /*SIGWINCH=*/28 : exitCode);
      builder.setExecutorHostname(hostName);
      builder.setWallTimeMillis(wallTimeMillis);
      return builder.build();
    }

    // TODO(ulfjack): Move this to the TerminationStatus class. It doesn't make sense that we work
    // around APIs that we control ourselves.
    private int condense(TerminationStatus status) {
      // The termination status is already condensed, but it doesn't give easy access to the
      // underlying raw code it wraps.
      return status.exited() ? status.getExitCode() : (128 + status.getTerminatingSignal());
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
