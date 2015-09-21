// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.unix.FilesystemUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SearchPath;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
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
    this.backgroundWorkers = backgroundWorkers;
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
      createTestTmpDir(spawn, sandboxPath);
    } catch (IllegalArgumentException | IOException e) {
      throw new UserExecException("Could not prepare mounts for sandbox execution", e);
    }

    int timeout = getTimeout(spawn);

    try {
      final NamespaceSandboxRunner runner =
          new NamespaceSandboxRunner(execRoot, sandboxPath, mounts, verboseFailures, sandboxDebug);
      try {
        runner.run(
            spawn.getArguments(),
            spawn.getEnvironment(),
            blazeDirs.getExecRoot().getPathFile(),
            outErr,
            spawn.getOutputFiles(),
            timeout);
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

  private int getTimeout(Spawn spawn) throws UserExecException {
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
   * Tests are a special case and we have to mount the TEST_SRCDIR where the test expects it to be
   * and also provide a TEST_TMPDIR to the test where it can store temporary files.
   */
  private void createTestTmpDir(Spawn spawn, Path sandboxPath) throws IOException {
    if (spawn.getEnvironment().containsKey("TEST_TMPDIR")) {
      FileSystem fs = blazeDirs.getFileSystem();
      Path source = fs.getPath(spawn.getEnvironment().get("TEST_TMPDIR"));
      Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
      FileSystemUtils.createDirectoryAndParents(target);
    }
  }

  private ImmutableMap<Path, Path> getMounts(
      Spawn spawn, ActionExecutionContext executionContext) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    mounts.putAll(mountUsualUnixDirs());
    mounts.putAll(withRecursedDirs(setupBlazeUtils()));
    mounts.putAll(withRecursedDirs(mountRunfilesFromManifests(spawn)));
    mounts.putAll(withRecursedDirs(mountRunfilesFromSuppliers(spawn)));
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
  static MountMap<Path, Path> withResolvedSymlinks(Map<Path, Path> mounts)
      throws IOException {
    MountMap<Path, Path> fixedMounts = new MountMap<>();
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
  static MountMap<Path, Path> withRecursedDirs(Map<Path, Path> mounts) throws IOException {
    MountMap<Path, Path> fixedMounts = new MountMap<>();
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
  private MountMap<Path, Path> mountUsualUnixDirs() throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    FileSystem fs = blazeDirs.getFileSystem();
    mounts.put(fs.getPath("/bin"), fs.getPath("/bin"));
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
  private MountMap<Path, Path> setupBlazeUtils() throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    Path mount = blazeDirs.getEmbeddedBinariesRoot().getRelative("build-runfiles");
    mounts.put(mount, mount);
    return mounts;
  }

  /**
   * Mount all runfiles that the spawn needs as specified in its runfiles manifests.
   */
  private MountMap<Path, Path> mountRunfilesFromManifests(Spawn spawn)
      throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    for (Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      Path targetDirectory = execRoot.getRelative(manifest.getKey());

      mounts.putAll(parseManifestFile(targetDirectory, new File(manifestFilePath)));
    }
    return mounts;
  }

  static MountMap<Path, Path> parseManifestFile(
      Path targetDirectory, File manifestFile) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    for (String line : Files.readLines(manifestFile, Charset.defaultCharset())) {
      String[] fields = line.trim().split(" ");
      Path source;
      Path targetPath = targetDirectory.getRelative(fields[0]);
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
  private MountMap<Path, Path> mountRunfilesFromSuppliers(Spawn spawn)
      throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
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
  private MountMap<Path, Path> mountInputs(
      Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();

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
  private MountMap<Path, Path> mountRunUnderCommand(Spawn spawn) {
    MountMap<Path, Path> mounts = new MountMap<>();

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
