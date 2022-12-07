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

package com.google.devtools.build.lib.worker;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.util.Set;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** A {@link SingleplexWorker} that runs inside a sandboxed execution root. */
final class SandboxedWorker extends SingleplexWorker {

  public static final String TMP_DIR_MOUNT_NAME = "_tmp";

  @AutoValue
  public abstract static class WorkerSandboxOptions {
    // Need to have this data class because we can't depend on SandboxOptions in here.
    abstract boolean fakeHostname();

    abstract boolean fakeUsername();

    abstract boolean debugMode();

    abstract ImmutableList<PathFragment> tmpfsPath();

    abstract ImmutableList<String> writablePaths();

    abstract Path sandboxBinary();

    public static WorkerSandboxOptions create(
        Path sandboxBinary,
        boolean fakeHostname,
        boolean fakeUsername,
        boolean debugMode,
        ImmutableList<PathFragment> tmpfsPath,
        ImmutableList<String> writablePaths) {
      return new AutoValue_SandboxedWorker_WorkerSandboxOptions(
          fakeHostname, fakeUsername, debugMode, tmpfsPath, writablePaths, sandboxBinary);
    }
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private final WorkerExecRoot workerExecRoot;
  /** Options specific to hardened sandbox, null if not using that. */
  @Nullable private final WorkerSandboxOptions hardenedSandboxOptions;

  SandboxedWorker(
      WorkerKey workerKey,
      int workerId,
      Path workDir,
      Path logFile,
      @Nullable WorkerSandboxOptions hardenedSandboxOptions) {
    super(workerKey, workerId, workDir, logFile);
    this.workerExecRoot =
        new WorkerExecRoot(
            workDir,
            hardenedSandboxOptions != null
                ? ImmutableList.of(PathFragment.create("../" + TMP_DIR_MOUNT_NAME))
                : ImmutableList.of());
    this.hardenedSandboxOptions = hardenedSandboxOptions;
  }

  @Override
  public boolean isSandboxed() {
    return true;
  }

  private ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot) throws IOException {
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    ImmutableSet.Builder<Path> writableDirs =
        ImmutableSet.<Path>builder().add(sandboxExecRoot).add(sandboxExecRoot.getRelative("/tmp"));

    FileSystem fileSystem = sandboxExecRoot.getFileSystem();
    for (String writablePath : hardenedSandboxOptions.writablePaths()) {
      Path path = fileSystem.getPath(writablePath);
      writableDirs.add(path);
      if (path.isSymbolicLink()) {
        writableDirs.add(path.resolveSymbolicLinks());
      }
    }

    FileSystem fs = sandboxExecRoot.getFileSystem();
    writableDirs.add(fs.getPath("/dev/shm").resolveSymbolicLinks());
    writableDirs.add(fs.getPath("/tmp"));

    return writableDirs.build();
  }

  private SortedMap<Path, Path> getBindMounts(Path sandboxExecRoot, @Nullable Path sandboxTmp) {
    Path tmpPath = sandboxExecRoot.getFileSystem().getPath("/tmp");
    final SortedMap<Path, Path> bindMounts = Maps.newTreeMap();
    // Mount a fresh, empty temporary directory as /tmp for each sandbox rather than reusing the
    // host filesystem's /tmp. Since we're in a worker, we clean this dir between requests.
    bindMounts.put(tmpPath, sandboxTmp);
    // TODO(larsrc): Apply InaccessiblePaths
    // for (Path inaccessiblePath : getInaccessiblePaths()) {
    //   if (inaccessiblePath.isDirectory(Symlinks.NOFOLLOW)) {
    //     bindMounts.put(inaccessiblePath, inaccessibleHelperDir);
    //   } else {
    //     bindMounts.put(inaccessiblePath, inaccessibleHelperFile);
    //   }
    // }
    // validateBindMounts(bindMounts);
    return bindMounts;
  }

  @Override
  protected Subprocess createProcess() throws IOException {
    // TODO(larsrc): Check that execRoot and outputBase are not under /tmp
    // TODO(larsrc): Maybe deduplicate this code copied from super.createProcess()
    if (hardenedSandboxOptions != null) {
      this.shutdownHook =
          new Thread(
              () -> {
                this.shutdownHook = null;
                this.destroy();
              });
      Runtime.getRuntime().addShutdownHook(shutdownHook);

      // TODO(larsrc): Figure out what of the environment rewrite needs doing.
      // ImmutableMap<String, String> environment =
      //     localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

      // TODO(larsrc): Figure out which things can change and make sure workers get restarted
      // ImmutableSet<Path> writableDirs = getWritableDirs(workerExecRoot, environment);

      ImmutableList<String> args = workerKey.getArgs();
      File executable = new File(args.get(0));
      if (!executable.isAbsolute() && executable.getParent() != null) {
        args =
            ImmutableList.<String>builderWithExpectedSize(args.size())
                .add(new File(workDir.getPathFile(), args.get(0)).getAbsolutePath())
                .addAll(args.subList(1, args.size()))
                .build();
      }

      // In hardened mode, we bindmount a temp dir. We put the mount dir in the parent directory to
      // avoid clashes with workspace files.
      Path sandboxTmp = workDir.getParentDirectory().getRelative(TMP_DIR_MOUNT_NAME);
      sandboxTmp.createDirectoryAndParents();

      // TODO(larsrc): Need to make error messages go to stderr.
      LinuxSandboxCommandLineBuilder commandLineBuilder =
          LinuxSandboxCommandLineBuilder.commandLineBuilder(
                  this.hardenedSandboxOptions.sandboxBinary(), args)
              .setWritableFilesAndDirectories(getWritableDirs(workDir))
              // Need all the sandbox options passed in here?
              .setTmpfsDirectories(ImmutableSet.copyOf(this.hardenedSandboxOptions.tmpfsPath()))
              .setPersistentProcess(true)
              .setBindMounts(getBindMounts(workDir, sandboxTmp))
              .setUseFakeHostname(this.hardenedSandboxOptions.fakeHostname())
              // Mostly tests require network, and some blaze run commands, but no workers.
              .setCreateNetworkNamespace(true)
              .setUseDebugMode(hardenedSandboxOptions.debugMode());

      if (this.hardenedSandboxOptions.fakeUsername()) {
        commandLineBuilder.setUseFakeUsername(true);
      }

      SubprocessBuilder processBuilder = new SubprocessBuilder();
      ImmutableList<String> argv = commandLineBuilder.build();
      processBuilder.setArgv(argv);
      processBuilder.setWorkingDirectory(workDir.getPathFile());
      processBuilder.setStderr(logFile.getPathFile());
      processBuilder.setEnv(workerKey.getEnv());

      return processBuilder.start();
    } else {
      return super.createProcess();
    }
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    workerExecRoot.createFileSystem(workerFiles, inputFiles, outputs);

    super.prepareExecution(inputFiles, outputs, workerFiles);
  }

  @Override
  public void finishExecution(Path execRoot, SandboxOutputs outputs) throws IOException {
    super.finishExecution(execRoot, outputs);

    workerExecRoot.copyOutputs(execRoot, outputs);
  }

  @Override
  void destroy() {
    super.destroy();
    try {
      workDir.deleteTree();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Caught IOException while deleting workdir.");
    }
  }
}
