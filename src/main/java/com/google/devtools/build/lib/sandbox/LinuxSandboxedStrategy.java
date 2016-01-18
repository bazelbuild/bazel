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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Files;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SearchPath;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Strategy that uses sandboxing to execute a process.
 */
@ExecutionStrategy(name = {"sandboxed"},
                   contextType = SpawnActionContext.class)
public class LinuxSandboxedStrategy implements SpawnActionContext {
  private final ExecutorService backgroundWorkers;

  private final ImmutableMap<String, String> clientEnv;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final boolean sandboxDebug;
  private final StandaloneSpawnStrategy standaloneStrategy;
  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();

  public LinuxSandboxedStrategy(
      Map<String, String> clientEnv,
      BlazeDirectories blazeDirs,
      ExecutorService backgroundWorkers,
      boolean verboseFailures,
      boolean sandboxDebug) {
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.backgroundWorkers = Preconditions.checkNotNull(backgroundWorkers);
    this.verboseFailures = verboseFailures;
    this.sandboxDebug = sandboxDebug;
    this.standaloneStrategy = new StandaloneSpawnStrategy(blazeDirs.getExecRoot(), verboseFailures);
  }

  /**
   * Executes the given {@code spawn}.
   */
  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException {
    Executor executor = actionExecutionContext.getExecutor();

    // Certain actions can't run remotely or in a sandbox - pass them on to the standalone strategy.
    if (!spawn.isRemotable()) {
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(
          Label.print(spawn.getOwner().getLabel()) + " [" + spawn.getResourceOwner().prettyPrint()
              + "]", spawn.asShellCommand(executor.getExecRoot()));
    }

    FileOutErr outErr = actionExecutionContext.getFileOutErr();

    // The execId is a unique ID just for this invocation of "exec".
    String execId = uuid + "-" + execCounter.getAndIncrement();

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath =
        execRoot.getRelative(Constants.PRODUCT_NAME + "-sandbox").getRelative(execId);

    ImmutableMap<Path, Path> mounts;
    try {
      // Gather all necessary mounts for the sandbox.
      mounts = getMounts(spawn, actionExecutionContext);
    } catch (IllegalArgumentException | IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }

    ImmutableSet<Path> createDirs = createImportantDirs(spawn.getEnvironment());

    int timeout = getTimeout(spawn);

    ImmutableSet.Builder<PathFragment> outputFiles = ImmutableSet.<PathFragment>builder();
    for (PathFragment optionalOutput : spawn.getOptionalOutputFiles()) {
      Preconditions.checkArgument(!optionalOutput.isAbsolute());
      outputFiles.add(optionalOutput);
    }
    for (ActionInput output : spawn.getOutputFiles()) {
      outputFiles.add(new PathFragment(output.getExecPathString()));
    }

    try {
      final NamespaceSandboxRunner runner =
          new NamespaceSandboxRunner(
              execRoot, sandboxPath, mounts, createDirs, verboseFailures, sandboxDebug);
      try {
        runner.run(
            spawn.getArguments(),
            spawn.getEnvironment(),
            execRoot.getPathFile(),
            outErr,
            outputFiles.build(),
            timeout,
            !spawn.getExecutionInfo().containsKey("requires-network"));
      } finally {
        // Due to the Linux kernel behavior, if we try to remove the sandbox too quickly after the
        // process has exited, we get "Device busy" errors because some of the mounts have not yet
        // been undone. A second later it usually works. We will just clean the old sandboxes up
        // using a background worker.
        backgroundWorkers.execute(
            new Runnable() {
              @Override
              public void run() {
                try {
                  while (!Thread.currentThread().isInterrupted()) {
                    try {
                      runner.cleanup();
                      return;
                    } catch (IOException e2) {
                      // Sleep & retry.
                      Thread.sleep(250);
                    }
                  }
                } catch (InterruptedException e) {
                  // Exit.
                }
              }
            });
      }
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    }
  }

  private int getTimeout(Spawn spawn) throws ExecException {
    String timeoutStr = spawn.getExecutionInfo().get("timeout");
    if (timeoutStr != null) {
      try {
        return Integer.parseInt(timeoutStr);
      } catch (NumberFormatException e) {
        throw new UserExecException("Could not parse timeout", e);
      }
    }
    return -1;
  }

  /**
   * Most programs expect certain directories to be present, e.g. /tmp. Make sure they are.
   *
   * <p>Note that $HOME is handled by namespace-sandbox.c, because it changes user to nobody and the
   * home directory of that user is not known by us.
   */
  private ImmutableSet<Path> createImportantDirs(Map<String, String> env) {
    ImmutableSet.Builder<Path> dirs = ImmutableSet.builder();
    FileSystem fs = blazeDirs.getFileSystem();
    if (env.containsKey("TEST_TMPDIR")) {
      dirs.add(fs.getPath(env.get("TEST_TMPDIR")));
    }
    dirs.add(fs.getPath("/tmp"));
    return dirs.build();
  }

  private ImmutableMap<Path, Path> getMounts(Spawn spawn, ActionExecutionContext executionContext)
      throws IOException, ExecException {
    MountMap mounts = new MountMap();
    mounts.putAll(mountUsualUnixDirs());
    mounts.putAll(withRecursedDirs(setupBlazeUtils()));
    mounts.putAll(withRecursedDirs(mountRunfilesFromManifests(spawn)));
    mounts.putAll(withRecursedDirs(mountRunfilesFromSuppliers(spawn)));
    mounts.putAll(withRecursedDirs(mountFilesFromFilesetManifests(spawn, executionContext)));
    mounts.putAll(withRecursedDirs(mountInputs(spawn, executionContext)));
    mounts.putAll(withRecursedDirs(mountRunUnderCommand(spawn)));
    return validateMounts(withResolvedSymlinks(mounts));
  }

  /**
   * Validates all mounts against a set of criteria and throws an exception on error.
   *
   * @return an ImmutableMap of all mounts.
   */
  @VisibleForTesting
  static ImmutableMap<Path, Path> validateMounts(Map<Path, Path> mounts) {
    ImmutableMap.Builder<Path, Path> validatedMounts = ImmutableMap.builder();
    for (Entry<Path, Path> mount : mounts.entrySet()) {
      Path target = mount.getKey();
      Path source = mount.getValue();

      // The source must exist.
      Preconditions.checkArgument(source.exists(), "%s does not exist", source.toString());

      validatedMounts.put(target, source);
    }
    return validatedMounts.build();
  }

  /**
   * Checks for each mount if the source refers to a symbolic link and if yes, adds another mount
   * for the target of that symlink to ensure that it keeps working inside the sandbox.
   *
   * @return a new mounts multimap with the added mounts.
   */
  @VisibleForTesting
  static MountMap withResolvedSymlinks(Map<Path, Path> mounts) throws IOException {
    MountMap fixedMounts = new MountMap();
    for (Entry<Path, Path> mount : mounts.entrySet()) {
      Path target = mount.getKey();
      Path source = mount.getValue();
      fixedMounts.put(target, source);

      if (source.isSymbolicLink()) {
        Path symlinkTarget = source.resolveSymbolicLinks();
        fixedMounts.put(symlinkTarget, symlinkTarget);
      }
    }
    return fixedMounts;
  }

  /**
   * Checks for each mount if the source refers to a directory and if yes, replaces that mount with
   * mounts of all files inside that directory.
   *
   * @return a new mounts multimap with the added mounts.
   */
  @VisibleForTesting
  static MountMap withRecursedDirs(Map<Path, Path> mounts) throws IOException {
    MountMap fixedMounts = new MountMap();
    for (Entry<Path, Path> mount : mounts.entrySet()) {
      Path target = mount.getKey();
      Path source = mount.getValue();

      if (source.isDirectory()) {
        for (Path subSource : FileSystemUtils.traverseTree(source, Predicates.alwaysTrue())) {
          Path subTarget = target.getRelative(subSource.relativeTo(source));
          fixedMounts.put(subTarget, subSource);
        }
      } else {
        fixedMounts.put(target, source);
      }
    }
    return fixedMounts;
  }

  /**
   * Mount a certain set of unix directories to make the usual tools and libraries available to the
   * spawn that runs.
   */
  private MountMap mountUsualUnixDirs() throws IOException {
    MountMap mounts = new MountMap();
    FileSystem fs = blazeDirs.getFileSystem();
    mounts.put(fs.getPath("/bin"), fs.getPath("/bin"));
    mounts.put(fs.getPath("/sbin"), fs.getPath("/sbin"));
    mounts.put(fs.getPath("/etc"), fs.getPath("/etc"));
    for (String entry : FilesystemUtils.readdir("/")) {
      if (entry.startsWith("lib")) {
        Path libDir = fs.getRootDirectory().getRelative(entry);
        mounts.put(libDir, libDir);
      }
    }
    for (String entry : FilesystemUtils.readdir("/usr")) {
      if (!entry.equals("local")) {
        Path usrDir = fs.getPath("/usr").getRelative(entry);
        mounts.put(usrDir, usrDir);
      }
    }
    return mounts;
  }

  /**
   * Mount the embedded tools.
   */
  private MountMap setupBlazeUtils() {
    MountMap mounts = new MountMap();
    Path mount = blazeDirs.getEmbeddedBinariesRoot().getRelative("build-runfiles");
    mounts.put(mount, mount);
    return mounts;
  }

  /**
   * Mount all runfiles that the spawn needs as specified in its runfiles manifests.
   */
  private MountMap mountRunfilesFromManifests(Spawn spawn) throws IOException, ExecException {
    MountMap mounts = new MountMap();
    for (Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      Path targetDirectory = execRoot.getRelative(manifest.getKey());

      mounts.putAll(parseManifestFile(targetDirectory, new File(manifestFilePath), false, ""));
    }
    return mounts;
  }

  /**
   * Mount all files that the spawn needs as specified in its fileset manifests.
   */
  private MountMap mountFilesFromFilesetManifests(
      Spawn spawn, ActionExecutionContext executionContext) throws IOException, ExecException {
    final FilesetActionContext filesetContext =
        executionContext.getExecutor().getContext(FilesetActionContext.class);
    MountMap mounts = new MountMap();
    for (Artifact fileset : spawn.getFilesetManifests()) {
      Path manifest =
          execRoot.getRelative(AnalysisUtils.getManifestPathFromFilesetPath(fileset.getExecPath()));
      Path targetDirectory = execRoot.getRelative(fileset.getExecPathString());

      mounts.putAll(
          parseManifestFile(
              targetDirectory, manifest.getPathFile(), true, filesetContext.getWorkspaceName()));
    }
    return mounts;
  }

  static MountMap parseManifestFile(
      Path targetDirectory, File manifestFile, boolean isFilesetManifest, String workspaceName)
      throws IOException, ExecException {
    MountMap mounts = new MountMap();
    int lineNum = 0;
    for (String line : Files.readLines(manifestFile, StandardCharsets.UTF_8)) {
      if (isFilesetManifest && (++lineNum % 2 == 0)) {
        continue;
      }
      if (line.isEmpty()) {
        continue;
      }

      String[] fields = line.trim().split(" ");

      Path targetPath;
      if (isFilesetManifest) {
        PathFragment targetPathFragment = new PathFragment(fields[0]);
        if (!workspaceName.isEmpty()) {
          if (!targetPathFragment.getSegment(0).equals(workspaceName)) {
            throw new EnvironmentalExecException(
                "Fileset manifest line must start with workspace name");
          }
          targetPathFragment = targetPathFragment.subFragment(1, targetPathFragment.segmentCount());
        }
        targetPath = targetDirectory.getRelative(targetPathFragment);
      } else {
        targetPath = targetDirectory.getRelative(fields[0]);
      }

      Path source;
      switch (fields.length) {
        case 1:
          source = targetDirectory.getFileSystem().getPath("/dev/null");
          break;
        case 2:
          source = targetDirectory.getFileSystem().getPath(fields[1]);
          break;
        default:
          throw new IllegalStateException("'" + line + "' splits into more than 2 parts");
      }

      mounts.put(targetPath, source);
    }
    return mounts;
  }

  /**
   * Mount all runfiles that the spawn needs as specified via its runfiles suppliers.
   */
  private MountMap mountRunfilesFromSuppliers(Spawn spawn) throws IOException {
    MountMap mounts = new MountMap();
    FileSystem fs = blazeDirs.getFileSystem();
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      Path root = fs.getRootDirectory().getRelative(rootAndMappings.getKey());
      for (Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        Path source = (sourceArtifact != null) ? sourceArtifact.getPath() : fs.getPath("/dev/null");

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        Path target = root.getRelative(mapping.getKey());
        mounts.put(target, source);
      }
    }
    return mounts;
  }

  /**
   * Mount all inputs of the spawn.
   */
  private MountMap mountInputs(Spawn spawn, ActionExecutionContext actionExecutionContext) {
    MountMap mounts = new MountMap();

    List<ActionInput> inputs =
        ActionInputHelper.expandMiddlemen(
            spawn.getInputFiles(), actionExecutionContext.getMiddlemanExpander());

    if (spawn.getResourceOwner() instanceof CppCompileAction) {
      CppCompileAction action = (CppCompileAction) spawn.getResourceOwner();
      if (action.shouldScanIncludes()) {
        inputs.addAll(action.getAdditionalInputs());
      }
    }

    for (ActionInput input : inputs) {
      if (input.getExecPathString().contains("internal/_middlemen/")) {
        continue;
      }
      Path mount = execRoot.getRelative(input.getExecPathString());
      mounts.put(mount, mount);
    }
    return mounts;
  }

  /**
   * If a --run_under= option is set and refers to a command via its path (as opposed to via its
   * label), we have to mount this. Note that this is best effort and works fine for shell scripts
   * and small binaries, but we can't track any further dependencies of this command.
   *
   * <p>If --run_under= refers to a label, it is automatically provided in the spawn's input files,
   * so mountInputs() will catch that case.
   */
  private MountMap mountRunUnderCommand(Spawn spawn) {
    MountMap mounts = new MountMap();

    if (spawn.getResourceOwner() instanceof TestRunnerAction) {
      TestRunnerAction testRunnerAction = ((TestRunnerAction) spawn.getResourceOwner());
      RunUnder runUnder = testRunnerAction.getExecutionSettings().getRunUnder();
      if (runUnder != null && runUnder.getCommand() != null) {
        PathFragment sourceFragment = new PathFragment(runUnder.getCommand());
        Path mount;
        if (sourceFragment.isAbsolute()) {
          mount = blazeDirs.getFileSystem().getPath(sourceFragment);
        } else if (blazeDirs.getExecRoot().getRelative(sourceFragment).exists()) {
          mount = blazeDirs.getExecRoot().getRelative(sourceFragment);
        } else {
          List<Path> searchPath =
              SearchPath.parse(blazeDirs.getFileSystem(), clientEnv.get("PATH"));
          mount = SearchPath.which(searchPath, runUnder.getCommand());
        }
        if (mount != null) {
          mounts.put(mount, mount);
        }
      }
    }
    return mounts;
  }

  @Override
  public String strategyLocality(String mnemonic, boolean remotable) {
    return "linux-sandboxing";
  }

  @Override
  public boolean isRemotable(String mnemonic, boolean remotable) {
    return false;
  }
}
