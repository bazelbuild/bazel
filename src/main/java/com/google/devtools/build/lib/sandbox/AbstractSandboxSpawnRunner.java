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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.ExecutionStatistics;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;

/** Abstract common ancestor for sandbox spawn runners implementing the common parts. */
abstract class AbstractSandboxSpawnRunner implements SpawnRunner {
  private static final int LOCAL_EXEC_ERROR = -1;
  private static final int POSIX_TIMEOUT_EXIT_CODE = /*SIGNAL_BASE=*/128 + /*SIGALRM=*/14;

  private static final String SANDBOX_DEBUG_SUGGESTION =
      "\n\nUse --sandbox_debug to see verbose messages from the sandbox";

  private final Path sandboxBase;
  private final SandboxOptions sandboxOptions;
  private final boolean verboseFailures;
  private final ImmutableSet<Path> inaccessiblePaths;

  public AbstractSandboxSpawnRunner(CommandEnvironment cmdEnv, Path sandboxBase) {
    this.sandboxBase = sandboxBase;
    this.sandboxOptions = cmdEnv.getOptions().getOptions(SandboxOptions.class);
    this.verboseFailures = cmdEnv.getOptions().getOptions(ExecutionOptions.class).verboseFailures;
    this.inaccessiblePaths =
        sandboxOptions.getInaccessiblePaths(cmdEnv.getRuntime().getFileSystem());
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, InterruptedException {
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    policy.report(ProgressStatus.SCHEDULING, getName());
    try (ResourceHandle ignored =
        ResourceManager.instance().acquireResources(owner, spawn.getLocalResources())) {
      policy.report(ProgressStatus.EXECUTING, getName());
      return actuallyExec(spawn, policy);
    } catch (IOException e) {
      throw new UserExecException("I/O exception during sandboxed execution", e);
    }
  }

  // TODO(laszlocsomor): refactor this class to make `actuallyExec`'s contract clearer: the caller
  // of `actuallyExec` should not depend on `actuallyExec` calling `runSpawn` because it's easy to
  // forget to do so in `actuallyExec`'s implementations.
  protected abstract SpawnResult actuallyExec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, InterruptedException, IOException;

  protected SpawnResult runSpawn(
      Spawn originalSpawn,
      SandboxedSpawn sandbox,
      SpawnExecutionPolicy policy,
      Path execRoot,
      Path tmpDir,
      Duration timeout,
      Optional<String> statisticsPath)
      throws IOException, InterruptedException {
    try {
      sandbox.createFileSystem();
      OutErr outErr = policy.getFileOutErr();
      policy.prefetchInputs();

      SpawnResult result =
          run(originalSpawn, sandbox, outErr, timeout, execRoot, tmpDir, statisticsPath);

      policy.lockOutputFiles();
      try {
        // We copy the outputs even when the command failed.
        sandbox.copyOutputs(execRoot);
      } catch (IOException e) {
        throw new IOException("Could not move output artifacts from sandboxed execution", e);
      }
      return result;
    } finally {
      if (!sandboxOptions.sandboxDebug) {
        sandbox.delete();
      }
    }
  }

  private final SpawnResult run(
      Spawn originalSpawn,
      SandboxedSpawn sandbox,
      OutErr outErr,
      Duration timeout,
      Path execRoot,
      Path tmpDir,
      Optional<String> statisticsPath)
      throws IOException, InterruptedException {
    Command cmd = new Command(
        sandbox.getArguments().toArray(new String[0]),
        sandbox.getEnvironment(),
        sandbox.getSandboxExecRoot().getPathFile());
      String failureMessage;
      if (sandboxOptions.sandboxDebug) {
        failureMessage =
            CommandFailureUtils.describeCommandFailure(
                true, sandbox.getArguments(), sandbox.getEnvironment(), execRoot.getPathString());
      } else {
        failureMessage =
            CommandFailureUtils.describeCommandFailure(
                verboseFailures,
                originalSpawn.getArguments(),
                originalSpawn.getEnvironment(),
                execRoot.getPathString()) + SANDBOX_DEBUG_SUGGESTION;
      }

    long startTime = System.currentTimeMillis();
    CommandResult commandResult;
    try {
      if (!tmpDir.exists() && !tmpDir.createDirectory()) {
        throw new IOException(String.format("Could not create temp directory '%s'", tmpDir));
      }
      commandResult = cmd.execute(outErr.getOutputStream(), outErr.getErrorStream());
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException();
      }
    } catch (AbnormalTerminationException e) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException();
      }
      commandResult = e.getResult();
    } catch (CommandException e) {
      // At the time this comment was written, this must be a ExecFailedException encapsulating an
      // IOException from the underlying Subprocess.Factory.
      String msg = e.getMessage() == null ? e.getClass().getName() : e.getMessage();
      outErr.getErrorStream().write(("Action failed to execute: " + msg + "\n").getBytes(UTF_8));
      outErr.getErrorStream().flush();
      return new SpawnResult.Builder()
          .setStatus(Status.EXECUTION_FAILED)
          .setExitCode(LOCAL_EXEC_ERROR)
          .setFailureMessage(failureMessage)
          .build();
    }

    // TODO(b/62588075): Calculate wall time inside commands instead?
    Duration wallTime = Duration.ofMillis(System.currentTimeMillis() - startTime);
    boolean wasTimeout = wasTimeout(timeout, wallTime);
    int exitCode =
        wasTimeout
            ? POSIX_TIMEOUT_EXIT_CODE
            : commandResult.getTerminationStatus().getRawExitCode();
    Status status =
        wasTimeout
            ? Status.TIMEOUT
            : (exitCode == 0) ? Status.SUCCESS : Status.NON_ZERO_EXIT;

    SpawnResult.Builder spawnResultBuilder =
        new SpawnResult.Builder()
            .setStatus(status)
            .setExitCode(exitCode)
            .setWallTime(wallTime)
            .setFailureMessage(status != Status.SUCCESS || exitCode != 0 ? failureMessage : "");

    if (statisticsPath.isPresent()) {
      Optional<ExecutionStatistics.ResourceUsage> resourceUsage =
          ExecutionStatistics.getResourceUsage(statisticsPath.get());
      if (resourceUsage.isPresent()) {
        spawnResultBuilder.setUserTime(resourceUsage.get().getUserExecutionTime());
        spawnResultBuilder.setSystemTime(resourceUsage.get().getSystemExecutionTime());
        spawnResultBuilder.setNumBlockOutputOperations(
            resourceUsage.get().getBlockOutputOperations());
        spawnResultBuilder.setNumBlockInputOperations(
            resourceUsage.get().getBlockInputOperations());
        spawnResultBuilder.setNumInvoluntaryContextSwitches(
            resourceUsage.get().getInvoluntaryContextSwitches());
      }
    }

    return spawnResultBuilder.build();
  }

  private boolean wasTimeout(Duration timeout, Duration wallTime) {
    return !timeout.isZero() && wallTime.compareTo(timeout) > 0;
  }

  /**
   * Returns a temporary directory that should be used as the sandbox directory for a single action.
   */
  protected Path getSandboxRoot() throws IOException {
    return sandboxBase.getRelative(
        java.nio.file.Files.createTempDirectory(
                java.nio.file.Paths.get(sandboxBase.getPathString()), "")
            .getFileName()
            .toString());
  }

  /**
   * Gets the list of directories that the spawn will assume to be writable.
   *
   * @throws IOException because we might resolve symlinks, which throws {@link IOException}.
   */
  protected ImmutableSet<Path> getWritableDirs(
      Path sandboxExecRoot, Map<String, String> env, Path tmpDir) throws IOException {
    FileSystem fileSystem = sandboxExecRoot.getFileSystem();

    // We have to make the TEST_TMPDIR directory writable if it is specified.
    ImmutableSet.Builder<Path> writablePaths = ImmutableSet.builder();
    writablePaths.add(sandboxExecRoot);
    String tmpDirString = env.get("TEST_TMPDIR");
    if (tmpDirString != null) {
      PathFragment testTmpDir = PathFragment.create(tmpDirString);
      if (testTmpDir.isAbsolute()) {
        Path p = fileSystem.getPath(testTmpDir);
        if (!p.exists()) {
          // If `testTmpDir` itself is a symlink, then adding it to `writablePaths` would result in
          // making the symlink itself writable, not what it points to. Therefore we need to resolve
          // symlinks in `testTmpDir`, however for that we need `testTmpDir` to exist.
          throw new IOException(
              String.format(
                  "Cannot resolve symlinks in TEST_TMPDIR, because it is a non-existent, "
                      + "absolute path: \"%s\"",
                  p.getPathString()));
        }
        writablePaths.add(p.resolveSymbolicLinks());
      } else {
        // We add this even though it is below sandboxExecRoot (and thus already writable as a
        // subpath) to take advantage of the side-effect that SymlinkedExecRoot also creates this
        // needed directory if it doesn't exist yet.
        writablePaths.add(sandboxExecRoot.getRelative(testTmpDir));
      }
    }

    writablePaths.add(tmpDir);

    for (String writablePath : sandboxOptions.sandboxWritablePath) {
      Path path = fileSystem.getPath(writablePath);
      writablePaths.add(path);
      writablePaths.add(path.resolveSymbolicLinks());
    }

    return writablePaths.build();
  }

  protected ImmutableSet<Path> getInaccessiblePaths() {
    return inaccessiblePaths;
  }

  protected SandboxOptions getSandboxOptions() {
    return sandboxOptions;
  }

  protected abstract String getName();
}
