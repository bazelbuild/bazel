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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder.BindMount;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.server.FailureDetails.Sandbox.Code;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** Spawn runner that uses linux sandboxing APIs to execute a local subprocess. */
final class LinuxSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {
  private static final PathFragment SLASH_TMP = PathFragment.create("/tmp");
  private static final PathFragment BAZEL_EXECROOT = PathFragment.create("bazel-execroot");
  private static final PathFragment BAZEL_WORKING_DIRECTORY = PathFragment.create("bazel-working-directory");
  private static final PathFragment BAZEL_SOURCE_ROOTS = PathFragment.create("bazel-source-roots");

  // Since checking if sandbox is supported is expensive, we remember what we've checked.
  private static final Map<Path, Boolean> isSupportedMap = new HashMap<>();
  private static final AtomicBoolean warnedAboutNonHermeticTmp = new AtomicBoolean();

  /**
   * Returns whether the linux sandbox is supported on the local machine by running a small command
   * in it.
   */
  public static boolean isSupported(final CommandEnvironment cmdEnv) throws InterruptedException {
    if (OS.getCurrent() != OS.LINUX) {
      return false;
    }
    if (!LinuxSandboxUtil.isSupported(cmdEnv.getBlazeWorkspace())) {
      return false;
    }
    Path linuxSandbox = LinuxSandboxUtil.getLinuxSandbox(cmdEnv.getBlazeWorkspace());
    Boolean isSupported;
    synchronized (isSupportedMap) {
      isSupported = isSupportedMap.get(linuxSandbox);
      if (isSupported != null) {
        return isSupported;
      }
      isSupported = computeIsSupported(cmdEnv, linuxSandbox);
      isSupportedMap.put(linuxSandbox, isSupported);
    }
    return isSupported;
  }

  private static boolean computeIsSupported(CommandEnvironment cmdEnv, Path linuxSandbox)
      throws InterruptedException {
    ImmutableList<String> linuxSandboxArgv =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(
                linuxSandbox, ImmutableList.of("/bin/true"))
            .setTimeout(Duration.ofSeconds(1))
            .build();
    ImmutableMap<String, String> env = ImmutableMap.of();
    Path execRoot = cmdEnv.getExecRoot();
    File cwd = execRoot.getPathFile();

    Command cmd = new Command(linuxSandboxArgv.toArray(new String[0]), env, cwd);
    try (SilentCloseable c = Profiler.instance().profile("LinuxSandboxedSpawnRunner.isSupported")) {
      cmd.execute(ByteStreams.nullOutputStream(), ByteStreams.nullOutputStream());
    } catch (CommandException e) {
      return false;
    }

    return true;
  }

  private final SandboxHelpers helpers;
  private final FileSystem fileSystem;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean allowNetwork;
  private final Path linuxSandbox;
  private final Path sandboxBase;
  private final Path inaccessibleHelperFile;
  private final Path inaccessibleHelperDir;
  private final LocalEnvProvider localEnvProvider;
  private final Duration timeoutKillDelay;
  @Nullable private final SandboxfsProcess sandboxfsProcess;
  private final boolean sandboxfsMapSymlinkTargets;
  private final TreeDeleter treeDeleter;
  private final Reporter reporter;

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param helpers common tools and state across all spawns during sandboxed execution
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param inaccessibleHelperFile path to a file that is (already) inaccessible
   * @param inaccessibleHelperDir path to a directory that is (already) inaccessible
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   * @param sandboxfsProcess instance of the sandboxfs process to use; may be null for none, in
   *     which case the runner uses a symlinked sandbox
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   */
  LinuxSandboxedSpawnRunner(
      SandboxHelpers helpers,
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir,
      Duration timeoutKillDelay,
      @Nullable SandboxfsProcess sandboxfsProcess,
      boolean sandboxfsMapSymlinkTargets,
      TreeDeleter treeDeleter) {
    super(cmdEnv);
    this.helpers = helpers;
    this.fileSystem = cmdEnv.getRuntime().getFileSystem();
    this.blazeDirs = cmdEnv.getDirectories();
    this.execRoot = cmdEnv.getExecRoot();
    this.allowNetwork = helpers.shouldAllowNetwork(cmdEnv.getOptions());
    this.linuxSandbox = LinuxSandboxUtil.getLinuxSandbox(cmdEnv.getBlazeWorkspace());
    this.sandboxBase = sandboxBase;
    this.inaccessibleHelperFile = inaccessibleHelperFile;
    this.inaccessibleHelperDir = inaccessibleHelperDir;
    this.timeoutKillDelay = timeoutKillDelay;
    this.sandboxfsProcess = sandboxfsProcess;
    this.sandboxfsMapSymlinkTargets = sandboxfsMapSymlinkTargets;
    this.localEnvProvider = new PosixLocalEnvProvider(cmdEnv.getClientEnv());
    this.treeDeleter = treeDeleter;
    this.reporter = cmdEnv.getReporter();
  }

  private void createDirectoryWithinSandboxTmp(Path sandboxTmp, Path withinSandboxDirectory) throws IOException {
    PathFragment withinTmp = withinSandboxDirectory.asFragment().relativeTo(SLASH_TMP);
    sandboxTmp.getRelative(withinTmp).createDirectoryAndParents();
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException, ExecException, InterruptedException {
    boolean useHermeticTmp = getSandboxOptions().sandboxHermeticTmp;
    boolean tmpExplicitlyBindMounted = getSandboxOptions().sandboxAdditionalMounts.stream()
        .anyMatch(e -> e.getKey().equals("/tmp"));

    if (useHermeticTmp && tmpExplicitlyBindMounted) {
      if (warnedAboutNonHermeticTmp.compareAndSet(false, true)) {
        reporter.handle(Event.warn(String.format(
            "Falling back to non-hermetic '/tmp' in because a bind mount of '/tmp' is explicitly requested")));
      }
      useHermeticTmp = false;
    }

    if (useHermeticTmp) {
      if (getSandboxOptions().sandboxTmpfsPath.contains(SLASH_TMP)) {
        if (warnedAboutNonHermeticTmp.compareAndSet(false, true)) {
          reporter.handle(Event.warn(String.format(
              "Both hermetic '/tmp' and an explicit tmpfs mount on '/tmp' is requested, using tmpfs")));
        }
        useHermeticTmp = false;
      }
    }

    if (useHermeticTmp) {
      Optional<PathFragment> tmpfsPathUnderTmp =
          getSandboxOptions().sandboxTmpfsPath.stream()
              .filter(path -> path.startsWith(SLASH_TMP))
              .findFirst();
      if (!tmpfsPathUnderTmp.isEmpty()) {
        useHermeticTmp = false;
        if (warnedAboutNonHermeticTmp.compareAndSet(false, true)) {
          reporter.handle(
              Event.warn(
                  String.format(
                      "Falling back to non-hermetic '/tmp' in sandbox due to '%s' being a tmpfs path",
                      tmpfsPathUnderTmp.get())));
        }
      }
    }

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    String workspaceName = execRoot.getBaseName();

    // Each invocation of "exec" gets its own sandbox base.
    // Note that the value returned by context.getId() is only unique inside one given SpawnRunner,
    // so we have to prefix our name to turn it into a globally unique value.
    Path sandboxPath =
        sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));
    Path sandboxExecRootBase = sandboxPath.getRelative("execroot");
    Path sandboxExecRoot = sandboxExecRootBase.getRelative(workspaceName);
    Path sandboxTmp = null;
    Path withinSandboxSourceRoots = null;
    Path withinSandboxWorkingDirectory = null;
    Path withinSandboxExecRoot = execRoot;

    if (useHermeticTmp) {
      sandboxTmp = sandboxPath.getRelative("_tmp");
      withinSandboxSourceRoots = fileSystem.getPath(SLASH_TMP.getRelative(BAZEL_SOURCE_ROOTS));
      withinSandboxWorkingDirectory = fileSystem.getPath(SLASH_TMP.getRelative(BAZEL_WORKING_DIRECTORY))
          .getRelative(workspaceName);
      withinSandboxExecRoot = fileSystem.getPath(SLASH_TMP.getRelative(BAZEL_EXECROOT))
          .getRelative(workspaceName);
    }

    SandboxInputs inputs =
        helpers.processInputFiles(
            context.getInputMapping(PathFragment.EMPTY_FRAGMENT),
            execRoot, withinSandboxExecRoot, withinSandboxSourceRoots);

    sandboxExecRoot.createDirectoryAndParents();

    if (useHermeticTmp) {
      for (Map.Entry<Root, Path> root : inputs.getSourceRootBindMounts().entrySet()) {
        createDirectoryWithinSandboxTmp(sandboxTmp, root.getKey().asPath());
      }

      createDirectoryWithinSandboxTmp(sandboxTmp, withinSandboxExecRoot);
      createDirectoryWithinSandboxTmp(sandboxTmp, withinSandboxWorkingDirectory);
    }

    SandboxOutputs outputs = helpers.getOutputs(spawn);
    ImmutableMap<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");
    ImmutableSet<Path> writableDirs = getWritableDirs(sandboxExecRoot, environment);
    Duration timeout = context.getTimeout();

    LinuxSandboxCommandLineBuilder commandLineBuilder =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandbox, spawn.getArguments())
            .addExecutionInfo(spawn.getExecutionInfo())
            .setWritableFilesAndDirectories(writableDirs)
            .setTmpfsDirectories(ImmutableSet.copyOf(getSandboxOptions().sandboxTmpfsPath))
            .setBindMounts(getBindMounts(blazeDirs, inputs, sandboxExecRootBase, sandboxTmp))
            .setUseFakeHostname(getSandboxOptions().sandboxFakeHostname)
            .setEnablePseudoterminal(getSandboxOptions().sandboxExplicitPseudoterminal)
            .setCreateNetworkNamespace(
                !(allowNetwork
                    || Spawns.requiresNetwork(
                        spawn, getSandboxOptions().defaultSandboxAllowNetwork)))
            .setUseDebugMode(getSandboxOptions().sandboxDebug)
            .setKillDelay(timeoutKillDelay);

    if (useHermeticTmp) {
      commandLineBuilder.setWorkingDirectory(withinSandboxWorkingDirectory);
    }

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }

    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_FAKEROOT)) {
      commandLineBuilder.setUseFakeRoot(true);
    } else if (getSandboxOptions().sandboxFakeUsername) {
      commandLineBuilder.setUseFakeUsername(true);
    }

    Path statisticsPath = null;
    if (getSandboxOptions().collectLocalSandboxExecutionStatistics) {
      statisticsPath = sandboxPath.getRelative("stats.out");
      commandLineBuilder.setStatisticsPath(statisticsPath);
    }

    if (sandboxfsProcess != null) {
      return new SandboxfsSandboxedSpawn(
          sandboxfsProcess,
          sandboxPath,
          workspaceName,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          ImmutableSet.of(),
          sandboxfsMapSymlinkTargets,
          treeDeleter,
          statisticsPath);
    } else if (getSandboxOptions().useHermetic) {
      commandLineBuilder.setHermeticSandboxPath(sandboxPath);
      return new HardlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          writableDirs,
          treeDeleter,
          statisticsPath,
          getSandboxOptions().sandboxDebug);
    } else {
      return new SymlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.build(),
          environment,
          inputs,
          outputs,
          writableDirs,
          treeDeleter,
          statisticsPath,
          getSandboxOptions().reuseSandboxDirectories,
          sandboxBase,
          spawn.getMnemonic());
    }
  }

  @Override
  public String getName() {
    return "linux-sandbox";
  }

  @Override
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env)
      throws IOException {
    ImmutableSet.Builder<Path> writableDirs = ImmutableSet.builder();
    writableDirs.addAll(super.getWritableDirs(sandboxExecRoot, env));

    FileSystem fs = sandboxExecRoot.getFileSystem();
    writableDirs.add(fs.getPath("/dev/shm").resolveSymbolicLinks());
    writableDirs.add(fs.getPath("/tmp"));

    return writableDirs.build();
  }

  private List<BindMount> getBindMounts(
      BlazeDirectories blazeDirs, SandboxInputs inputs,
      Path sandboxExecRootBase, @Nullable Path sandboxTmp)
      throws UserExecException {
    Path tmpPath = fileSystem.getPath("/tmp");
    final SortedMap<Path, Path> bindMounts = Maps.newTreeMap();

    for (ImmutableMap.Entry<String, String> additionalMountPath :
        getSandboxOptions().sandboxAdditionalMounts) {
      try {
        final Path mountTarget = fileSystem.getPath(additionalMountPath.getValue());
        // If source path is relative, treat it as a relative path inside the execution root
        final Path mountSource = sandboxExecRootBase.getRelative(additionalMountPath.getKey());
        // If a target has more than one source path, the latter one will take effect.
        bindMounts.put(mountTarget, mountSource);
      } catch (IllegalArgumentException e) {
        throw new UserExecException(
            createFailureDetail(
                String.format("Error occurred when analyzing bind mount pairs. %s", e.getMessage()),
                Code.BIND_MOUNT_ANALYSIS_FAILURE));
      }
    }

    for (Path inaccessiblePath : getInaccessiblePaths()) {
      if (inaccessiblePath.isDirectory(Symlinks.NOFOLLOW)) {
        bindMounts.put(inaccessiblePath, inaccessibleHelperDir);
      } else {
        bindMounts.put(inaccessiblePath, inaccessibleHelperFile);
      }
    }

    validateBindMounts(bindMounts);
    ImmutableList.Builder<BindMount> result = ImmutableList.builder();

    if (sandboxTmp != null) {
      result.add(BindMount.of(sandboxTmp.getRelative(BAZEL_EXECROOT), blazeDirs.getExecRootBase()));
      result.add(BindMount.of(sandboxTmp.getRelative(BAZEL_WORKING_DIRECTORY), sandboxExecRootBase));

      for (Map.Entry<Root, Path> sourceRoot : inputs.getSourceRootBindMounts().entrySet()) {
        Path realSourceRoot = sourceRoot.getValue();
        Root withinSandboxSourceRoot = sourceRoot.getKey();
        PathFragment sandboxTmpSourceRoot = withinSandboxSourceRoot.asPath().relativeTo(tmpPath);
        result.add(BindMount.of(sandboxTmp.getRelative(sandboxTmpSourceRoot), realSourceRoot));
      }

      result.add(BindMount.of(tmpPath, sandboxTmp));
    }

    for (Map.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      result.add(BindMount.of(bindMount.getKey(), bindMount.getValue()));
    }

    return result.build();
  }

  /**
   * This method does the following things:
   *
   * <ul>
   *   <li>If mount source does not exist on the host system, throw an error message
   *   <li>If mount target exists, check whether the source and target are of the same type
   *   <li>If mount target does not exist on the host system, throw an error message
   * </ul>
   *
   * @param bindMounts the bind mounts map with target as key and source as value
   * @throws UserExecException if any of the mount points are not valid
   */
  private void validateBindMounts(SortedMap<Path, Path> bindMounts) throws UserExecException {
    for (Map.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      final Path source = bindMount.getValue();
      final Path target = bindMount.getKey();
      // Mount source should exist in the file system
      if (!source.exists()) {
        throw new UserExecException(
            createFailureDetail(
                String.format("Mount source '%s' does not exist.", source),
                Code.MOUNT_SOURCE_DOES_NOT_EXIST));
      }
      // If target exists, but is not of the same type as the source, then we cannot mount it.
      if (target.exists()) {
        boolean areBothDirectories = source.isDirectory() && target.isDirectory();
        boolean isSourceFile = source.isFile() || source.isSymbolicLink();
        boolean isTargetFile = target.isFile() || target.isSymbolicLink();
        boolean areBothFiles = isSourceFile && isTargetFile;
        if (!(areBothDirectories || areBothFiles)) {
          // Source and target are not of the same type; we cannot mount it.
          throw new UserExecException(
              createFailureDetail(
                  String.format(
                      "Mount target '%s' is a %s but mount source '%s' is a %s, they must be the"
                          + " same type.",
                      target,
                      (isTargetFile ? "file" : "directory"),
                      source,
                      (isSourceFile ? "file" : "directory")),
                  Code.MOUNT_SOURCE_TARGET_TYPE_MISMATCH));
        }
      } else {
        // Mount target should exist in the file system
        throw new UserExecException(
            createFailureDetail(
                String.format(
                    "Mount target '%s' does not exist. Bazel only supports bind mounting on top of "
                        + "existing files/directories. Please create an empty file or directory at "
                        + "the mount target path according to the type of mount source.",
                    target),
                Code.MOUNT_TARGET_DOES_NOT_EXIST));
      }
    }
  }

  @Override
  public void verifyPostCondition(
      Spawn originalSpawn, SandboxedSpawn sandbox, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException {
    if (getSandboxOptions().useHermetic) {
      checkForConcurrentModifications(context);
    }
  }

  private void checkForConcurrentModifications(SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException {
    for (ActionInput input : context.getInputMapping(PathFragment.EMPTY_FRAGMENT).values()) {
      if (input instanceof VirtualActionInput) {
        continue;
      }

      FileArtifactValue metadata = context.getMetadataProvider().getMetadata(input);
      Path path = execRoot.getRelative(input.getExecPath());

      try {
        if (wasModifiedSinceDigest(metadata.getContentsProxy(), path)) {
          throw new IOException("input dependency " + path + " was modified during execution.");
        }
      } catch (UnsupportedOperationException e) {
        throw new IOException(
            "input dependency "
                + path
                + " could not be checked for modifications during execution.",
            e);
      }
    }
  }

  private boolean wasModifiedSinceDigest(FileContentsProxy proxy, Path path) throws IOException {
    if (proxy == null) {
      return false;
    }
    FileStatus stat = path.statIfFound(Symlinks.FOLLOW);
    return stat == null || !stat.isFile() || proxy.isModified(FileContentsProxy.create(stat));
  }

  @Override
  public void cleanupSandboxBase(Path sandboxBase, TreeDeleter treeDeleter) throws IOException {
    // Delete the inaccessible files synchronously, bypassing the treeDeleter. They are only a
    // couple of files that can be deleted fast, and ensuring they are gone at the end of every
    // build avoids annoying permission denied errors if the user happens to run "rm -rf" on the
    // output base. (We have some tests that do that.)
    if (inaccessibleHelperDir.exists()) {
      inaccessibleHelperDir.chmod(0700);
      inaccessibleHelperDir.deleteTree();
    }
    if (inaccessibleHelperFile.exists()) {
      inaccessibleHelperFile.chmod(0600);
      inaccessibleHelperFile.delete();
    }

    super.cleanupSandboxBase(sandboxBase, treeDeleter);
  }
}
