// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.ExecutionStatistics;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;

/** Abstract common ancestor for sandbox spawn runners implementing the common parts. */
abstract class AbstractSandboxSpawnRunner implements SpawnRunner {
  private static final int LOCAL_EXEC_ERROR = -1;
  private static final int POSIX_TIMEOUT_EXIT_CODE = /*SIGNAL_BASE=*/128 + /*SIGALRM=*/14;

  private static final String SANDBOX_DEBUG_SUGGESTION =
      "\n\nUse --sandbox_debug to see verbose messages from the sandbox";

  private final SandboxOptions sandboxOptions;
  private final boolean verboseFailures;
  private final ImmutableSet<Path> inaccessiblePaths;
  protected final BinTools binTools;
  private final Path execRoot;
  private final ResourceManager resourceManager;

  public AbstractSandboxSpawnRunner(CommandEnvironment cmdEnv) {
    this.sandboxOptions = cmdEnv.getOptions().getOptions(SandboxOptions.class);
    this.verboseFailures = cmdEnv.getOptions().getOptions(ExecutionOptions.class).verboseFailures;
    this.inaccessiblePaths =
        sandboxOptions.getInaccessiblePaths(cmdEnv.getRuntime().getFileSystem());
    this.binTools = cmdEnv.getBlazeWorkspace().getBinTools();
    this.execRoot = cmdEnv.getExecRoot();
    this.resourceManager = cmdEnv.getLocalResourceManager();
  }

  @Override
  public final SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    context.report(ProgressStatus.SCHEDULING, getName());
    try (ResourceHandle ignored =
        resourceManager.acquireResources(owner, spawn.getLocalResources())) {
      context.report(ProgressStatus.EXECUTING, getName());
      SandboxedSpawn sandbox = prepareSpawn(spawn, context);
      return runSpawn(spawn, sandbox, context);
    } catch (IOException e) {
      throw new UserExecException("I/O exception during sandboxed execution", e);
    }
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return Spawns.mayBeSandboxed(spawn);
  }

  protected abstract SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ExecException;

  private SpawnResult runSpawn(
      Spawn originalSpawn, SandboxedSpawn sandbox, SpawnExecutionContext context)
      throws IOException, InterruptedException {
    try {
      try (SilentCloseable c = Profiler.instance().profile("sandbox.createFileSystem")) {
        sandbox.createFileSystem();
      }
      FileOutErr outErr = context.getFileOutErr();
      try (SilentCloseable c = Profiler.instance().profile("context.prefetchInputs")) {
        context.prefetchInputs();
      }

      SpawnResult result;
      try (SilentCloseable c = Profiler.instance().profile("subprocess.run")) {
        result = run(originalSpawn, sandbox, context.getTimeout(), outErr);
      }

      context.lockOutputFiles();
      try (SilentCloseable c = Profiler.instance().profile("sandbox.copyOutputs")) {
        // We copy the outputs even when the command failed.
        sandbox.copyOutputs(execRoot);
      } catch (IOException e) {
        throw new IOException("Could not move output artifacts from sandboxed execution", e);
      }
      return result;
    } finally {
      if (!sandboxOptions.sandboxDebug) {
        try (SilentCloseable c = Profiler.instance().profile("sandbox.delete")) {
          sandbox.delete();
        }
      }
    }
  }

  private String makeFailureMessage(Spawn originalSpawn, SandboxedSpawn sandbox) {
    if (sandboxOptions.sandboxDebug) {
      return CommandFailureUtils.describeCommandFailure(
          true,
          sandbox.getArguments(),
          sandbox.getEnvironment(),
          sandbox.getSandboxExecRoot().getPathString(),
          null);
    } else {
      return CommandFailureUtils.describeCommandFailure(
              verboseFailures,
              originalSpawn.getArguments(),
              originalSpawn.getEnvironment(),
              sandbox.getSandboxExecRoot().getPathString(),
              originalSpawn.getExecutionPlatform())
          + SANDBOX_DEBUG_SUGGESTION;
    }
  }

  private final SpawnResult run(
      Spawn originalSpawn, SandboxedSpawn sandbox, Duration timeout, FileOutErr outErr)
      throws IOException, InterruptedException {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setWorkingDirectory(sandbox.getSandboxExecRoot().getPathFile());
    subprocessBuilder.setStdout(outErr.getOutputPath().getPathFile());
    subprocessBuilder.setStderr(outErr.getErrorPath().getPathFile());
    subprocessBuilder.setEnv(sandbox.getEnvironment());
    subprocessBuilder.setArgv(ImmutableList.copyOf(sandbox.getArguments()));
    boolean useSubprocessTimeout = sandbox.useSubprocessTimeout();
    if (useSubprocessTimeout) {
      subprocessBuilder.setTimeoutMillis(timeout.toMillis());
    }
    long startTime = System.currentTimeMillis();
    TerminationStatus terminationStatus;
    try {
      Subprocess subprocess = subprocessBuilder.start();
      subprocess.getOutputStream().close();
      try {
        subprocess.waitFor();
        terminationStatus = new TerminationStatus(subprocess.exitValue(), subprocess.timedout());
      } catch (InterruptedException e) {
        subprocess.destroy();
        throw e;
      }
    } catch (IOException e) {
      String msg = e.getMessage() == null ? e.getClass().getName() : e.getMessage();
      outErr
          .getErrorStream()
          .write(("Action failed to execute: java.io.IOException: " + msg + "\n").getBytes(UTF_8));
      outErr.getErrorStream().flush();
      return new SpawnResult.Builder()
          .setRunnerName(getName())
          .setStatus(Status.EXECUTION_FAILED)
          .setExitCode(LOCAL_EXEC_ERROR)
          .setFailureMessage(makeFailureMessage(originalSpawn, sandbox))
          .build();
    }

    // TODO(b/62588075): Calculate wall time inside Subprocess instead?
    Duration wallTime = Duration.ofMillis(System.currentTimeMillis() - startTime);
    boolean wasTimeout =
        (useSubprocessTimeout && terminationStatus.timedOut())
            || (!useSubprocessTimeout && wasTimeout(timeout, wallTime));
    int exitCode = wasTimeout ? POSIX_TIMEOUT_EXIT_CODE : terminationStatus.getRawExitCode();
    Status status =
        wasTimeout
            ? Status.TIMEOUT
            : (exitCode == 0) ? Status.SUCCESS : Status.NON_ZERO_EXIT;

    SpawnResult.Builder spawnResultBuilder =
        new SpawnResult.Builder()
            .setRunnerName(getName())
            .setStatus(status)
            .setExitCode(exitCode)
            .setWallTime(wallTime)
            .setFailureMessage(
                status != Status.SUCCESS || exitCode != 0
                    ? makeFailureMessage(originalSpawn, sandbox)
                    : "");

    Path statisticsPath = sandbox.getStatisticsPath();
    if (statisticsPath != null) {
      ExecutionStatistics.getResourceUsage(statisticsPath)
          .ifPresent(
              resourceUsage -> {
                spawnResultBuilder.setUserTime(resourceUsage.getUserExecutionTime());
                spawnResultBuilder.setSystemTime(resourceUsage.getSystemExecutionTime());
                spawnResultBuilder.setNumBlockOutputOperations(
                    resourceUsage.getBlockOutputOperations());
                spawnResultBuilder.setNumBlockInputOperations(
                    resourceUsage.getBlockInputOperations());
                spawnResultBuilder.setNumInvoluntaryContextSwitches(
                    resourceUsage.getInvoluntaryContextSwitches());
              });
    }

    return spawnResultBuilder.build();
  }

  private boolean wasTimeout(Duration timeout, Duration wallTime) {
    return !timeout.isZero() && wallTime.compareTo(timeout) > 0;
  }

  /**
   * Gets the list of directories that the spawn will assume to be writable.
   *
   * @throws IOException because we might resolve symlinks, which throws {@link IOException}.
   */
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env)
      throws IOException {
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    ImmutableSet.Builder<Path> writablePaths = ImmutableSet.builder();

    // On Windows, sandboxExecRoot is actually the main execroot. We will specify
    // exactly which output path is writable.
    if (OS.getCurrent() != OS.WINDOWS) {
      writablePaths.add(sandboxExecRoot);
    }

    String testTmpdir = env.get("TEST_TMPDIR");
    if (testTmpdir != null) {
      addWritablePath(
          sandboxExecRoot,
          writablePaths,
          testTmpdir,
          "Cannot resolve symlinks in TEST_TMPDIR because it doesn't exist: \"%s\"");
    }
    // As of 2019-07-08:
    // - every caller of `getWritableDirs` passes a LocalEnvProvider-processed environment as
    //   `env`, therefore `env` surely has an entry for TMPDIR on Unix and TEMP/TMP on Windows.
    if (OS.getCurrent() == OS.WINDOWS) {
      addWritablePath(
          sandboxExecRoot,
          writablePaths,
          Preconditions.checkNotNull(env.get("TEMP")),
          "Cannot resolve symlinks in TEMP because it doesn't exist: \"%s\"");
      addWritablePath(
          sandboxExecRoot,
          writablePaths,
          Preconditions.checkNotNull(env.get("TMP")),
          "Cannot resolve symlinks in TMP because it doesn't exist: \"%s\"");
    } else {
      addWritablePath(
          sandboxExecRoot,
          writablePaths,
          Preconditions.checkNotNull(env.get("TMPDIR")),
          "Cannot resolve symlinks in TMPDIR because it doesn't exist: \"%s\"");
    }

    FileSystem fileSystem = sandboxExecRoot.getFileSystem();
    for (String writablePath : sandboxOptions.sandboxWritablePath) {
      Path path = fileSystem.getPath(writablePath);
      writablePaths.add(path);
      // TODO(laszlocsomor): Remove if guard when path.resolveSymbolicLinks supports non-symlink
      // TODO(laszlocsomor): Figure out why OS.getCurrent() != OS.WINDOWS is required, and remove it
      if (OS.getCurrent() != OS.WINDOWS || path.isSymbolicLink()) {
        writablePaths.add(path.resolveSymbolicLinks());
      }
    }

    return writablePaths.build();
  }

  private void addWritablePath(
      Path sandboxExecRoot,
      ImmutableSet.Builder<Path> writablePaths,
      String pathString,
      String pathDoesNotExistErrorTemplate)
      throws IOException {
    Path path = sandboxExecRoot.getRelative(pathString);
    if (path.startsWith(sandboxExecRoot)) {
      // We add this path even though it is below sandboxExecRoot (and thus already writable as a
      // subpath) to take advantage of the side-effect that SymlinkedExecRoot also creates this
      // needed directory if it doesn't exist yet.
      writablePaths.add(path);
    } else if (path.exists()) {
      // If `path` itself is a symlink, then adding it to `writablePaths` would result in making
      // the symlink itself writable, not what it points to. Therefore we need to resolve symlinks
      // in `path`, however for that we need `path` to exist.
      //
      // TODO(laszlocsomor): Remove if guard when path.resolveSymbolicLinks supports non-symlink
      // TODO(laszlocsomor): Figure out why OS.getCurrent() != OS.WINDOWS is required, and remove it
      if (OS.getCurrent() != OS.WINDOWS || path.isSymbolicLink()) {
        writablePaths.add(path.resolveSymbolicLinks());
      } else {
        writablePaths.add(path);
      }
    } else {
      throw new IOException(String.format(pathDoesNotExistErrorTemplate, path.getPathString()));
    }
  }

  protected ImmutableSet<Path> getInaccessiblePaths() {
    return inaccessiblePaths;
  }

  protected SandboxOptions getSandboxOptions() {
    return sandboxOptions;
  }

  @Override
  public void cleanupSandboxBase(Path sandboxBase, TreeDeleter treeDeleter) throws IOException {
    Path root = sandboxBase.getChild(getName());
    if (root.exists()) {
      for (Path child : root.getDirectoryEntries()) {
        treeDeleter.deleteTree(child);
      }
    }
  }
}
