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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder.NetworkNamespace.NETNS_WITH_LOOPBACK;
import static com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder.NetworkNamespace.NO_NETNS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.sandbox.LinuxSandboxCommandLineBuilder.BindMount;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.cgroups.VirtualCGroup;
import com.google.devtools.build.lib.sandbox.cgroups.VirtualCGroupFactory;
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
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** Spawn runner that uses linux sandboxing APIs to execute a local subprocess. */
final class LinuxSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {
  private static final PathFragment SLASH_TMP = PathFragment.create("/tmp");
  private static final PathFragment BAZEL_EXECROOT = PathFragment.create("bazel-execroot");
  private static final PathFragment BAZEL_WORKING_DIRECTORY =
      PathFragment.create("bazel-working-directory");
  private static final PathFragment BAZEL_SOURCE_ROOTS = PathFragment.create("bazel-source-roots");

  // Since checking if sandbox is supported is expensive, we remember what we've checked.
  private static final Map<Path, Boolean> isSupportedMap = new HashMap<>();

  private static final AtomicBoolean warnedAboutUnsupportedModificationCheck = new AtomicBoolean();

  private final VirtualCGroupFactory cgroupFactory;

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
    LocalExecutionOptions options = cmdEnv.getOptions().getOptions(LocalExecutionOptions.class);
    ImmutableList<String> linuxSandboxArgv =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandbox)
            .setTimeout(options.getLocalSigkillGraceSeconds())
            .buildForCommand(ImmutableList.of("/bin/true"));
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
  private final TreeDeleter treeDeleter;
  private final Reporter reporter;
  private final ImmutableList<Root> packageRoots;

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param helpers common tools and state across all spawns during sandboxed execution
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param inaccessibleHelperFile path to a file that is (already) inaccessible
   * @param inaccessibleHelperDir path to a directory that is (already) inaccessible
   * @param timeoutKillDelay an additional grace period before killing timing out commands
   */
  LinuxSandboxedSpawnRunner(
      SandboxHelpers helpers,
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Path inaccessibleHelperFile,
      Path inaccessibleHelperDir,
      Duration timeoutKillDelay,
      TreeDeleter treeDeleter) throws IOException {
    super(cmdEnv);
    this.cgroupFactory =
        new VirtualCGroupFactory(
            "sandbox_",
            VirtualCGroup.getInstance(),
            ImmutableMap.copyOf(getSandboxOptions().limits),
            /* alwaysCreate= */ false);
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
    this.localEnvProvider = new PosixLocalEnvProvider(cmdEnv.getClientEnv());
    this.treeDeleter = treeDeleter;
    this.reporter = cmdEnv.getReporter();
    this.packageRoots = cmdEnv.getPackageLocator().getPathEntries();
  }

  private void createDirectoryWithinSandboxTmp(Path sandboxTmp, Path withinSandboxDirectory)
      throws IOException {
    PathFragment withinTmp = withinSandboxDirectory.asFragment().relativeTo(SLASH_TMP);
    sandboxTmp.getRelative(withinTmp).createDirectoryAndParents();
  }

  private boolean useHermeticTmp() {
    if (!getSandboxOptions().sandboxHermeticTmp) {
      // No hermetic /tmp requested, so let's not do it
      return false;
    }

    if (getSandboxOptions().useHermetic) {
      // The hermetic sandbox is, well, already hermetic. Also, it creates an empty /tmp by default
      // so nothing needs to be done to achieve a /tmp that is also hermetic.
      return false;
    }

    boolean tmpExplicitlyBindMounted =
        getSandboxOptions().sandboxAdditionalMounts.stream()
            .anyMatch(e -> e.getKey().equals("/tmp"));
    if (tmpExplicitlyBindMounted) {
      // An explicit mount on /tmp is likely an explicit way to make it non-hermetic.
      return false;
    }

    if (getSandboxOptions().sandboxTmpfsPath.contains(SLASH_TMP)) {
      // A tmpfs path under /tmp is as hermetic as "hermetic /tmp".
      return false;
    }

    return true;
  }

  @Override
  protected SandboxedSpawn prepareSpawn(Spawn spawn, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException, ExecException, InterruptedException {
    // b/64689608: The execroot of the sandboxed process must end with the workspace name, just like
    // the normal execroot does.
    String workspaceName = execRoot.getBaseName();

    // Each invocation of "exec" gets its own sandbox base.
    // Note that the value returned by context.getId() is only unique inside one given SpawnRunner,
    // so we have to prefix our name to turn it into a globally unique value.
    Path sandboxPath =
        sandboxBase.getRelative(getName()).getRelative(Integer.toString(context.getId()));

    // The exec root base and the exec root of the sandbox from the point of view of the Bazel
    // process (can be different from where the exec root appears within the sandbox due to file
    // system namespace shenanigans).
    Path sandboxExecRootBase = sandboxPath.getRelative("execroot");
    Path sandboxExecRoot = sandboxExecRootBase.getRelative(workspaceName);

    // The directory that will be mounted as the hermetic /tmp, if any (otherwise null)
    Path sandboxTmp = null;

    // These paths are paths that are visible for the processes running inside the sandbox. They
    // can be different from paths from the point of view of the Bazel server because if we use
    // hermetic /tmp and either the output base or a source root are under /tmp, they would be
    // hidden by the newly mounted hermetic /tmp . So in that case, we make the sandboxed processes
    // see the exec root, the source roots and the working directory of the action at constant
    // locations under /tmp .

    // Base directory for source roots; each source root is a sequentially numbered subdirectory.
    Path withinSandboxSourceRoots = null;

    // Working directory of the action; this is where the inputs (and only the inputs) of the action
    // are visible.
    Path withinSandboxWorkingDirectory = null;

    // The exec root. Necessary because the working directory contains symlinks to the execroot.
    Path withinSandboxExecRoot = execRoot;

    boolean useHermeticTmp = useHermeticTmp();

    if (useHermeticTmp) {
      // The directory which will be mounted at /tmp in the sandbox
      sandboxTmp = sandboxPath.getRelative("_hermetic_tmp");
      withinSandboxSourceRoots = fileSystem.getPath(SLASH_TMP.getRelative(BAZEL_SOURCE_ROOTS));
      withinSandboxWorkingDirectory =
          fileSystem
              .getPath(SLASH_TMP.getRelative(BAZEL_WORKING_DIRECTORY))
              .getRelative(workspaceName);
      withinSandboxExecRoot =
          fileSystem.getPath(SLASH_TMP.getRelative(BAZEL_EXECROOT)).getRelative(workspaceName);
    }

    SandboxInputs inputs =
        helpers.processInputFiles(
            context.getInputMapping(PathFragment.EMPTY_FRAGMENT, /* willAccessRepeatedly= */ true),
            context.getInputMetadataProvider(),
            execRoot,
            withinSandboxExecRoot,
            packageRoots,
            withinSandboxSourceRoots);

    sandboxExecRoot.createDirectoryAndParents();

    if (useHermeticTmp) {
      for (Root root : inputs.getSourceRootBindMounts().keySet()) {
        createDirectoryWithinSandboxTmp(sandboxTmp, root.asPath());
      }

      createDirectoryWithinSandboxTmp(sandboxTmp, withinSandboxExecRoot);
      createDirectoryWithinSandboxTmp(sandboxTmp, withinSandboxWorkingDirectory);

      for (PathFragment pathFragment : getSandboxOptions().sandboxTmpfsPath) {
        Path path = fileSystem.getPath(pathFragment);
        if (path.startsWith(SLASH_TMP)) {
          // tmpfs mount points must exist, which is usually the user's responsibility. But if the
          // user requests a tmpfs mount under /tmp, we have to create it under the sandbox tmp
          // directory.
          createDirectoryWithinSandboxTmp(sandboxTmp, path);
        }
      }
    }

    SandboxOutputs outputs = helpers.getOutputs(spawn);
    ImmutableMap<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");
    ImmutableSet<Path> writableDirs =
        getWritableDirs(
            sandboxExecRoot, useHermeticTmp ? withinSandboxExecRoot : sandboxExecRoot, environment);
    Duration timeout = context.getTimeout();
    SandboxOptions sandboxOptions = getSandboxOptions();

    boolean createNetworkNamespace =
        !(allowNetwork || Spawns.requiresNetwork(spawn, sandboxOptions.defaultSandboxAllowNetwork));
    LinuxSandboxCommandLineBuilder commandLineBuilder =
        LinuxSandboxCommandLineBuilder.commandLineBuilder(linuxSandbox)
            .addExecutionInfo(spawn.getExecutionInfo())
            .setWritableFilesAndDirectories(writableDirs)
            .setTmpfsDirectories(ImmutableSet.copyOf(getSandboxOptions().sandboxTmpfsPath))
            .setBindMounts(
                prepareAndGetBindMounts(blazeDirs, inputs, sandboxExecRootBase, sandboxTmp))
            .setUseFakeHostname(getSandboxOptions().sandboxFakeHostname)
            .setEnablePseudoterminal(getSandboxOptions().sandboxExplicitPseudoterminal)
            .setCreateNetworkNamespace(createNetworkNamespace ? NETNS_WITH_LOOPBACK : NO_NETNS)
            .setKillDelay(timeoutKillDelay);

    Path sandboxDebugPath = null;
    if (sandboxOptions.sandboxDebug) {
      sandboxDebugPath = sandboxPath.getRelative("debug.out");
      commandLineBuilder.setSandboxDebugPath(sandboxDebugPath.getPathString());
    }

    if (!spawn.getExecutionInfo().containsKey(ExecutionRequirements.NO_SUPPORTS_CGROUPS)) {
      ImmutableMap<String, Double> spawnResourceLimits = ImmutableMap.of();
      if (sandboxOptions.enforceResources.regexPattern().matcher(spawn.getMnemonic()).matches()) {
        spawnResourceLimits = spawn.getLocalResources().getResources();
      }
      cgroupFactory
        .create(context.getId(), spawnResourceLimits)
        .map(VirtualCGroup::paths)
        .ifPresent(commandLineBuilder::setCgroupsDirs);
    }

    if (useHermeticTmp) {
      commandLineBuilder.setWorkingDirectory(withinSandboxWorkingDirectory);
    }

    if (!timeout.isZero()) {
      commandLineBuilder.setTimeout(timeout);
    }
    if (spawn.getExecutionInfo().containsKey(ExecutionRequirements.REQUIRES_FAKEROOT)) {
      commandLineBuilder.setUseFakeRoot(true);
    } else if (sandboxOptions.sandboxFakeUsername) {
      commandLineBuilder.setUseFakeUsername(true);
    }
    Path statisticsPath = sandboxPath.getRelative("stats.out");
    commandLineBuilder.setStatisticsPath(statisticsPath);
    if (sandboxOptions.useHermetic) {
      commandLineBuilder.setHermeticSandboxPath(sandboxPath);
      return new HardlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.buildForCommand(spawn.getArguments()),
          environment,
          inputs,
          outputs,
          writableDirs,
          treeDeleter,
          sandboxDebugPath,
          statisticsPath,
          sandboxOptions.sandboxDebug,
          spawn.getMnemonic());
    } else {
      return new SymlinkedSandboxedSpawn(
          sandboxPath,
          sandboxExecRoot,
          commandLineBuilder.buildForCommand(spawn.getArguments()),
          environment,
          inputs,
          outputs,
          writableDirs,
          treeDeleter,
          sandboxDebugPath,
          statisticsPath,
          makeInteractiveDebugArguments(commandLineBuilder, sandboxOptions),
          spawn.getMnemonic());
    }
  }

  @Override
  public String getName() {
    return "linux-sandbox";
  }

  @Override
  protected ImmutableSet<Path> getWritableDirs(
      Path sandboxExecRoot, Path withinSandboxExecRoot, Map<String, String> env)
      throws IOException {
    Set<Path> writableDirs = new TreeSet<>();
    writableDirs.addAll(super.getWritableDirs(sandboxExecRoot, withinSandboxExecRoot, env));
    FileSystem fs = sandboxExecRoot.getFileSystem();
    writableDirs.add(fs.getPath("/dev/shm").resolveSymbolicLinks());
    writableDirs.add(fs.getPath("/tmp"));

    if (sandboxExecRoot.equals(withinSandboxExecRoot)) {
      return ImmutableSet.copyOf(writableDirs);
    }

    // If a writable directory is under the sandbox exec root, transform it so that its path will
    // be the one that it will be available at after processing the bind mounts (this is how the
    // sandbox interprets the corresponding arguments)
    //
    // Notably, this is usually the case for $TEST_TMPDIR because its default value is under the
    // execroot.
    return writableDirs.stream()
        .map(
            d ->
                d.startsWith(sandboxExecRoot)
                    ? withinSandboxExecRoot.getRelative(d.relativeTo(sandboxExecRoot))
                    : d)
        .collect(toImmutableSet());
  }

  private ImmutableList<BindMount> prepareAndGetBindMounts(
      BlazeDirectories blazeDirs,
      SandboxInputs inputs,
      Path sandboxExecRootBase,
      @Nullable Path sandboxTmp)
      throws UserExecException, IOException {
    Path tmpPath = fileSystem.getPath(SLASH_TMP);
    final SortedMap<Path, Path> userBindMounts = new TreeMap<>();
    SandboxHelpers.mountAdditionalPaths(
        getSandboxOptions().sandboxAdditionalMounts, sandboxExecRootBase, userBindMounts);

    for (Path inaccessiblePath : getInaccessiblePaths()) {
      if (inaccessiblePath.isDirectory(Symlinks.NOFOLLOW)) {
        userBindMounts.put(inaccessiblePath, inaccessibleHelperDir);
      } else {
        userBindMounts.put(inaccessiblePath, inaccessibleHelperFile);
      }
    }

    LinuxSandboxUtil.validateBindMounts(userBindMounts);

    if (sandboxTmp == null) {
      return userBindMounts.entrySet().stream()
          .map(e -> BindMount.of(e.getKey(), e.getValue()))
          .collect(toImmutableList());
    }

    SortedMap<Path, Path> bindMounts = new TreeMap<>();
    for (var entry : userBindMounts.entrySet()) {
      Path mountPoint = entry.getKey();
      Path content = entry.getValue();
      if (mountPoint.startsWith(tmpPath)) {
        // sandboxTmp should be null if /tmp is an explicit mount point since useHermeticTmp()
        // returns false in that case.
        if (mountPoint.equals(tmpPath)) {
          throw new IOException(
              "Cannot mount /tmp explicitly with hermetic /tmp. Please file a bug at"
                  + " https://github.com/bazelbuild/bazel/issues/new/choose.");
        }
        // We need to rewrite the mount point to be under the sandbox tmp directory, which will be
        // mounted onto /tmp as the final mount.
        mountPoint = sandboxTmp.getRelative(mountPoint.relativeTo(tmpPath));
        mountPoint.createDirectoryAndParents();
      }
      bindMounts.put(mountPoint, content);
    }

    ImmutableList.Builder<BindMount> result = ImmutableList.builder();
    bindMounts.forEach((k, v) -> result.add(BindMount.of(k, v)));

    // First mount the real exec root and the empty directory created as the working dir of the
    // action under $SANDBOX/_tmp
    result.add(BindMount.of(sandboxTmp.getRelative(BAZEL_EXECROOT), blazeDirs.getExecRootBase()));
    result.add(BindMount.of(sandboxTmp.getRelative(BAZEL_WORKING_DIRECTORY), sandboxExecRootBase));

    // Then mount the individual package roots under $SANDBOX/_tmp/bazel-source-roots
    inputs
        .getSourceRootBindMounts()
        .forEach(
            (withinSandbox, real) -> {
              PathFragment sandboxTmpSourceRoot = withinSandbox.asPath().relativeTo(tmpPath);
              result.add(BindMount.of(sandboxTmp.getRelative(sandboxTmpSourceRoot), real));
            });

    // Then mount $SANDBOX/_tmp at /tmp. At this point, even if the output base (and execroot) and
    // individual source roots are under /tmp, they are accessible at /tmp/bazel-*
    result.add(BindMount.of(tmpPath, sandboxTmp));
    return result.build();
  }

  @Override
  public void verifyPostCondition(
      Spawn originalSpawn, SandboxedSpawn sandbox, SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException {
    if (getSandboxOptions().useHermetic) {
      checkForConcurrentModifications(context);
    }
    // We cannot leave the cgroups around and delete them only when we delete the sandboxes
    // because linux has a hard limit of 65535 memory controllers.
    // Ref. https://github.com/torvalds/linux/blob/58d4e450a490d5f02183f6834c12550ba26d3b47/include/linux/memcontrol.h#L69
    cgroupFactory.remove(context.getId()).ifPresent(VirtualCGroup::delete);
  }

  private void checkForConcurrentModifications(SpawnExecutionContext context)
      throws IOException, ForbiddenActionInputException {
    for (ActionInput input :
        context
            .getInputMapping(PathFragment.EMPTY_FRAGMENT, /* willAccessRepeatedly= */ true)
            .values()) {
      if (input instanceof VirtualActionInput) {
        // Virtual inputs are not existing in file system and can't be tampered with via sandbox. No
        // need to check them.
        continue;
      }

      FileArtifactValue metadata = context.getInputMetadataProvider().getInputMetadata(input);
      if (metadata == null) {
        // This can happen if we are executing a spawn in an action that has multiple spawns and
        // the output of one is the input of another. In this case, we assume that no one modifies
        // an output of the first spawn before the action is completed (which requires the
        // the completion of the second spawn, which happens after this point is reached in the
        // code)
        continue;
      }
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
    VirtualCGroup.deleteInstance();
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

  @Nullable
  private ImmutableList<String> makeInteractiveDebugArguments(
      LinuxSandboxCommandLineBuilder commandLineBuilder, SandboxOptions sandboxOptions) {
    if (!sandboxOptions.sandboxDebug) {
      return null;
    }
    return commandLineBuilder.buildForCommand(ImmutableList.of("/bin/sh", "-i"));
  }
}
