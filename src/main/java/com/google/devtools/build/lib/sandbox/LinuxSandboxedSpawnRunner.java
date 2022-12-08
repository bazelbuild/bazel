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

import static java.nio.charset.StandardCharsets.UTF_8;

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
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.Nullable;

/** Spawn runner that uses linux sandboxing APIs to execute a local subprocess. */
final class LinuxSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  // Since checking if sandbox is supported is expensive, we remember what we've checked.
  private static final Map<Path, Boolean> isSupportedMap = new HashMap<>();
  private static final AtomicBoolean warnedAboutNonHermeticTmp = new AtomicBoolean();

  private static final AtomicBoolean warnedAboutUnsupportedModificationCheck = new AtomicBoolean();

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
  private final ImmutableList<Root> packageRoots;
  /** If non-null, this is a cgroups directory that sandboxes can put their directories into. */
  @Nullable private static java.nio.file.Path cgroupsBlazeDir;

  private static final ReentrantLock cgroupsBlazeDirLock = new ReentrantLock();

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
    this.packageRoots = cmdEnv.getPackageLocator().getPathEntries();
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException, ExecException, InterruptedException {
    // Each invocation of "exec" gets its own sandbox base.
    // Note that the value returned by context.getId() is only unique inside one given SpawnRunner,
    // so we have to prefix our name to turn it into a globally unique value.
    Path sandboxPath =
        sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));
    sandboxPath.getParentDirectory().createDirectory();
    sandboxPath.createDirectory();

    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    String workspaceName = execRoot.getBaseName();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(workspaceName);
    sandboxExecRoot.getParentDirectory().createDirectory();
    sandboxExecRoot.createDirectory();

    Path sandboxTmp = null;
    SandboxOptions sandboxOptions = getSandboxOptions();
    if (sandboxOptions.sandboxHermeticTmp) {
      PathFragment tmpRoot = PathFragment.create("/tmp");
      // With a tmpfs on /tmp, mounting a disk-based hermetic /tmp isn't necessary.
      if (!sandboxOptions.sandboxTmpfsPath.contains(tmpRoot)) {
        // Mounting a tmpfs strictly below the hermetic /tmp isn't supported. We fall back to
        // non-hermetic /tmp in that case, but print a warning mentioning the problematic mount.
        Optional<PathFragment> tmpfsPathUnderTmp =
            sandboxOptions.sandboxTmpfsPath.stream()
                .filter(path -> path.startsWith(tmpRoot))
                .findFirst();
        if (tmpfsPathUnderTmp.isEmpty()) {
          sandboxTmp = sandboxPath.getRelative("_tmp");
          sandboxTmp.createDirectoryAndParents();
        } else if (warnedAboutNonHermeticTmp.compareAndSet(false, true)) {
          reporter.handle(
              Event.warn(
                  String.format(
                      "Falling back to non-hermetic /tmp in sandbox due to '%s' being a tmpfs path",
                      tmpfsPathUnderTmp.get())));
        }
      }
    }

    ImmutableMap<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    ImmutableSet<Path> writableDirs = getWritableDirs(sandboxExecRoot, environment);

    SandboxInputs inputs =
        helpers.processInputFiles(
            context.getInputMapping(PathFragment.EMPTY_FRAGMENT),
            execRoot,
            execRoot,
            packageRoots,
            null);
    SandboxOutputs outputs = helpers.getOutputs(spawn);

    Duration timeout = context.getTimeout();

    LinuxSandboxCommandLineBuilder commandLineBuilder =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandbox, spawn.getArguments())
            .addExecutionInfo(spawn.getExecutionInfo())
            .setWritableFilesAndDirectories(writableDirs)
            .setTmpfsDirectories(ImmutableSet.copyOf(sandboxOptions.sandboxTmpfsPath))
            .setBindMounts(getBindMounts(blazeDirs, sandboxExecRoot, sandboxTmp))
            .setUseFakeHostname(sandboxOptions.sandboxFakeHostname)
            .setEnablePseudoterminal(sandboxOptions.sandboxExplicitPseudoterminal)
            .setCreateNetworkNamespace(
                !(allowNetwork
                    || Spawns.requiresNetwork(spawn, sandboxOptions.defaultSandboxAllowNetwork)))
            .setUseDebugMode(sandboxOptions.sandboxDebug)
            .setKillDelay(timeoutKillDelay);

    if (sandboxOptions.memoryLimit > 0) {
      String cgroupsDir = createCgroupsDir(context.getId());
      commandLineBuilder.setCgroupsDir(cgroupsDir);
      Files.writeString(
          Paths.get(cgroupsDir).resolve("memory.max"),
          Integer.toString(sandboxOptions.memoryLimit),
          UTF_8);
    }

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }

    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_FAKEROOT)) {
      commandLineBuilder.setUseFakeRoot(true);
    } else if (sandboxOptions.sandboxFakeUsername) {
      commandLineBuilder.setUseFakeUsername(true);
    }

    Path statisticsPath = null;
    if (sandboxOptions.collectLocalSandboxExecutionStatistics) {
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
    } else if (sandboxOptions.useHermetic) {
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
          sandboxOptions.sandboxDebug);
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
          sandboxOptions.reuseSandboxDirectories,
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

  private SortedMap<Path, Path> getBindMounts(
      BlazeDirectories blazeDirs, Path sandboxExecRoot, @Nullable Path sandboxTmp)
      throws UserExecException {
    Path tmpPath = fileSystem.getPath("/tmp");
    final SortedMap<Path, Path> bindMounts = Maps.newTreeMap();
    boolean buildUnderTmp = false;
    if (blazeDirs.getWorkspace().startsWith(tmpPath)) {
      bindMounts.put(blazeDirs.getWorkspace(), blazeDirs.getWorkspace());
      buildUnderTmp = true;
    }
    if (blazeDirs.getOutputBase().startsWith(tmpPath)) {
      bindMounts.put(blazeDirs.getOutputBase(), blazeDirs.getOutputBase());
      buildUnderTmp = true;
    }
    if (sandboxTmp != null) {
      if (buildUnderTmp) {
        if (warnedAboutNonHermeticTmp.compareAndSet(false, true)) {
          reporter.handle(
              Event.warn(
                  "Falling back to non-hermetic /tmp in sandbox since workspace or output base "
                      + "lie under /tmp"));
        }
      } else {
        // Mount a fresh, empty temporary directory as /tmp for each sandbox rather than reusing the
        // host filesystem's /tmp. User-specified bind mounts can override this and use the host's
        // /tmp instead by mounting /tmp to /tmp, if desired.
        bindMounts.put(tmpPath, sandboxTmp);
      }
    }
    for (ImmutableMap.Entry<String, String> additionalMountPath :
        getSandboxOptions().sandboxAdditionalMounts) {
      try {
        final Path mountTarget = fileSystem.getPath(additionalMountPath.getValue());
        // If source path is relative, treat it as a relative path inside the execution root
        final Path mountSource = sandboxExecRoot.getRelative(additionalMountPath.getKey());
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
    return bindMounts;
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

  /** Creates a cgroups directory to control this sandbox. */
  private String createCgroupsDir(int id) throws IOException, InterruptedException {
    java.nio.file.Path blazeCgroupsDir = getCgroupsBlazeDir();
    var cgroupsDir = blazeCgroupsDir.resolve("sandbox_" + id + ".scope");
    Files.createDirectories(cgroupsDir);
    Files.write(
        blazeCgroupsDir.resolve("cgroup.subtree_control"), ImmutableList.of("+memory"), UTF_8);
    Files.write(cgroupsDir.resolve("memory.oom.group"), ImmutableList.of("1"), UTF_8);
    return cgroupsDir.toString();
  }

  /**
   * Gets the common cgroups dir that all sandbox cgroups dirs should live under.
   *
   * @return A Path for a cgroups node we can create more nodes in.
   * @throws IOException If the cgroups setup doesn't allow making our own cgroups.
   */
  private static java.nio.file.Path getCgroupsBlazeDir() throws IOException, InterruptedException {
    if (cgroupsBlazeDir == null) {
      cgroupsBlazeDirLock.lockInterruptibly();
      try {
        if (cgroupsBlazeDir == null) {
          // Using java.nio.Paths in here because we need a truly absolute path.
          var procSelfCgroupContents = Files.readAllLines(Paths.get("/proc/self/cgroup"), UTF_8);
          if (procSelfCgroupContents.isEmpty()
              || !procSelfCgroupContents.get(0).startsWith("0::")) {
            throw new IOException(
                "Cgroups v2 requested, but /proc/self/cgroup does not start with 0::");
          }
          String cgroupsProcPath = procSelfCgroupContents.get(0).trim().substring(3);
          var cgroupsNode = Paths.get("/sys/fs/cgroup" + cgroupsProcPath).getParent();
          cgroupsBlazeDir =
              cgroupsNode.resolve("blaze_" + ProcessHandle.current().pid() + ".slice");
        }
      } finally {
        cgroupsBlazeDirLock.unlock();
      }
    }
    return cgroupsBlazeDir;
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
        // Virtual inputs are not existing in file system and can't be tampered with via sandbox. No
        // need to check them.
        continue;
      }

      FileArtifactValue metadata = context.getMetadataProvider().getMetadata(input);
      if (!metadata.getType().isFile()) {
        // The hermetic sandbox creates hardlinks from files inside sandbox to files outside
        // sandbox. The content of the files outside the sandbox could have been tampered with via
        // the hardlinks. Therefore files are checked for modifications. But directories and
        // unresolved symlinks are not represented as hardlinks in sandbox and don't need to be
        // checked. By continue and not checking them, we avoid UnsupportedOperationException and
        // IllegalStateException.
        continue;
      }

      Path path = execRoot.getRelative(input.getExecPath());
      try {
        if (wasModifiedSinceDigest(metadata.getContentsProxy(), path)) {
          throw new IOException("input dependency " + path + " was modified during execution.");
        }
      } catch (UnsupportedOperationException e) {
        // Some FileArtifactValue implementations are ignored safely and silently already by the
        // isFile check above. The remaining ones should probably be checked, but some are not
        // supporting necessary operations.
        if (warnedAboutUnsupportedModificationCheck.compareAndSet(false, true)) {
          reporter.handle(
              Event.warn(
                  String.format(
                      "Input dependency %s of type %s could not be checked for modifications during"
                          + " execution. Suppressing similar warnings.",
                      path, metadata.getClass().getSimpleName())));
        }
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
