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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auto.value.AutoValue;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.sandbox.CgroupsInfo;
import com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder;
import com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder.BindMount;
import com.google.devtools.build.lib.sandbox.LinuxSandboxUtil;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
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

    abstract int memoryLimit();

    abstract ImmutableSet<Path> inaccessiblePaths();

    public static WorkerSandboxOptions create(
        Path sandboxBinary,
        boolean fakeHostname,
        boolean fakeUsername,
        boolean debugMode,
        ImmutableList<PathFragment> tmpfsPath,
        ImmutableList<String> writablePaths,
        int memoryLimit,
        ImmutableSet<Path> inaccessiblePaths) {
      return new AutoValue_SandboxedWorker_WorkerSandboxOptions(
          fakeHostname,
          fakeUsername,
          debugMode,
          tmpfsPath,
          writablePaths,
          sandboxBinary,
          memoryLimit,
          inaccessiblePaths);
    }
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private final WorkerExecRoot workerExecRoot;
  /** Options specific to hardened sandbox, null if not using that. */
  @Nullable private final WorkerSandboxOptions hardenedSandboxOptions;
  /** If non-null, a directory that allows cgroup control. */
  private String cgroupsDir;

  private Path inaccessibleHelperDir;
  private Path inaccessibleHelperFile;

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
    // writableDirs.add(fs.getPath("/sys/fs/cgroup"));

    return writableDirs.build();
  }

  private ImmutableList<BindMount> getBindMounts(Path sandboxExecRoot, @Nullable Path sandboxTmp)
      throws UserExecException, IOException {
    Path tmpPath = sandboxExecRoot.getFileSystem().getPath("/tmp");
    final SortedMap<Path, Path> bindMounts = Maps.newTreeMap();
    ImmutableList.Builder<BindMount> result = ImmutableList.builder();
    // Mount a fresh, empty temporary directory as /tmp for each sandbox rather than reusing the
    // host filesystem's /tmp. Since we're in a worker, we clean this dir between requests.
    bindMounts.put(tmpPath, sandboxTmp);
    // TODO(larsrc): Add support for sandboxAdditionalMounts
    inaccessibleHelperFile = LinuxSandboxUtil.getInaccessibleHelperFile(sandboxExecRoot);
    inaccessibleHelperDir = LinuxSandboxUtil.getInaccessibleHelperDir(sandboxExecRoot);
    for (Path inaccessiblePath : hardenedSandboxOptions.inaccessiblePaths()) {
      if (inaccessiblePath.isDirectory(Symlinks.NOFOLLOW)) {
        bindMounts.put(inaccessiblePath, inaccessibleHelperDir);
      } else {
        bindMounts.put(inaccessiblePath, inaccessibleHelperFile);
      }
    }
    // TODO(larsrc): Handle hermetic tmp
    for (Map.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      result.add(BindMount.of(bindMount.getKey(), bindMount.getValue()));
    }
    LinuxSandboxUtil.validateBindMounts(bindMounts);
    return result.build();
  }

  @Override
  protected Subprocess createProcess() throws IOException, UserExecException {
    ImmutableList<String> args = makeExecPathAbsolute(workerKey.getArgs());
    // TODO(larsrc): Check that execRoot and outputBase are not under /tmp
    if (hardenedSandboxOptions != null) {
      // In hardened mode, we bindmount a temp dir. We put the mount dir in the parent directory to
      // avoid clashes with workspace files.
      Path sandboxTmp = workDir.getParentDirectory().getRelative(TMP_DIR_MOUNT_NAME);
      sandboxTmp.createDirectoryAndParents();

      // TODO(larsrc): Need to make sure error messages go to stderr.
      LinuxSandboxCommandLineBuilder commandLineBuilder =
          LinuxSandboxCommandLineBuilder.commandLineBuilder(
                  this.hardenedSandboxOptions.sandboxBinary(), args)
              // TODO(larsrc): Figure out which things can change and make sure workers get
              // restarted
              .setWritableFilesAndDirectories(getWritableDirs(workDir))
              .setTmpfsDirectories(ImmutableSet.copyOf(this.hardenedSandboxOptions.tmpfsPath()))
              .setPersistentProcess(true)
              .setBindMounts(getBindMounts(workDir, sandboxTmp))
              .setUseFakeHostname(this.hardenedSandboxOptions.fakeHostname())
              // Mostly tests require network, and some blaze run commands, but no workers.
              .setCreateNetworkNamespace(true)
              .setUseDebugMode(hardenedSandboxOptions.debugMode());

      if (hardenedSandboxOptions.memoryLimit() > 0) {
        CgroupsInfo cgroupsInfo = CgroupsInfo.getInstance();
        // We put the sandbox inside a unique subdirectory using the worker's ID.
        cgroupsDir =
            cgroupsInfo.createMemoryLimitCgroupDir(
                "worker_sandbox_" + workerId, hardenedSandboxOptions.memoryLimit());
        commandLineBuilder.setCgroupsDir(cgroupsDir);
      }

      if (this.hardenedSandboxOptions.fakeUsername()) {
        commandLineBuilder.setUseFakeUsername(true);
      }

      args = commandLineBuilder.build();
    }
    return createProcessBuilder(args).start();
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException, InterruptedException, UserExecException {
    workerExecRoot.createFileSystem(workerFiles, inputFiles, outputs);

    super.prepareExecution(inputFiles, outputs, workerFiles);
  }

  @Override
  void putRequest(WorkRequest request) throws IOException {
    try {
      super.putRequest(request);
    } catch (IOException e) {
      if (cgroupsDir != null && wasCgroupEvent()) {
        throw new IOException("HIT LIMIT", e);
      }
      throw e;
    }
  }

  private boolean wasCgroupEvent() throws IOException {
    // Check if we killed it "ourselves", throw specific exception
    List<String> memoryEvents = Files.readLines(new File(cgroupsDir, "memory.events"), UTF_8);
    for (String ev : memoryEvents) {
      if (ev.startsWith("oom_kill") || ev.startsWith("oom_group_kill")) {
        List<String> pieces = Splitter.on(" ").splitToList(ev);
        int count = Integer.parseInt(pieces.get(1));
        if (count > 0) {
          // BOOM
          return true;
        }
      }
    }
    return false;
  }

  @Override
  WorkResponse getResponse(int requestId) throws IOException, InterruptedException {
    try {
      return super.getResponse(requestId);
    } catch (IOException e) {
      if (cgroupsDir != null && wasCgroupEvent()) {
        throw new IOException("HIT LIMIT", e);
      }
      throw e;
    }
  }

  @Override
  public void finishExecution(Path execRoot, SandboxOutputs outputs) throws IOException {
    super.finishExecution(execRoot, outputs);
    if (cgroupsDir != null) {
      // This is only to not leave too much behind in the cgroups tree, can ignore errors.
      new File(cgroupsDir).delete();
    }
    workerExecRoot.copyOutputs(execRoot, outputs);
  }

  @Override
  void destroy() {
    super.destroy();
    try {
      if (inaccessibleHelperFile != null) {
        inaccessibleHelperFile.delete();
      }
      if (inaccessibleHelperDir != null) {
        inaccessibleHelperDir.delete();
      }
      if (cgroupsDir != null) {
        // This is only to not leave too much behind in the cgroups tree, can ignore errors.
        new File(cgroupsDir).delete();
      }
      workDir.deleteTree();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Caught IOException while deleting workdir.");
    }
  }
}
