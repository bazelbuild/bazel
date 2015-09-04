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
import com.google.common.base.Throwables;
import com.google.common.collect.ForwardingMap;
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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.syntax.Label;
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
import java.util.LinkedHashMap;
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

  private final BlazeRuntime blazeRuntime;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final StandaloneSpawnStrategy standaloneStrategy;
  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();

  /**
   * A map that throws an exception when trying to replace a key (i.e. once a key gets a value,
   * any additional attempt of putting a value on the same key will throw an exception).
   */
  static class MountMap<K, V> extends ForwardingMap<K, V> {
    final LinkedHashMap<K, V> delegate = new LinkedHashMap<>();

    @Override
    protected Map<K, V> delegate() {
      return delegate;
    }

    @Override
    public V put(K key, V value) {
      V previousValue = get(key);
      if (previousValue == null) {
        return super.put(key, value);
      } else if (previousValue.equals(value)) {
        return value;
      } else {
        throw new IllegalArgumentException(
            String.format("Cannot mount both '%s' and '%s' onto '%s'", previousValue, value, key));
      }
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> map) {
      for (Entry<? extends K, ? extends V> entry : map.entrySet()) {
        put(entry.getKey(), entry.getValue());
      }
    }
  }

  public LinuxSandboxedStrategy(
      BlazeRuntime blazeRuntime, boolean verboseFailures, ExecutorService backgroundWorkers) {
    this.blazeRuntime = blazeRuntime;
    this.blazeDirs = blazeRuntime.getDirectories();
    this.execRoot = blazeDirs.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.backgroundWorkers = backgroundWorkers;
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
      mounts = getMounts(spawn, sandboxPath, actionExecutionContext);
      createTestTmpDir(spawn, sandboxPath);
    } catch (IllegalArgumentException | IOException e) {
      throw new UserExecException("Could not prepare mounts for sandbox execution", e);
    }

    int timeout = getTimeout(spawn);

    try {
      final NamespaceSandboxRunner runner =
          new NamespaceSandboxRunner(execRoot, sandboxPath, mounts, verboseFailures);
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
    } catch (CommandException e) {
      EventHandler handler = actionExecutionContext.getExecutor().getEventHandler();
      handler.handle(
          Event.error("Sandboxed execution failed: " + spawn.getOwner().getLabel() + "."));
      throw new UserExecException("Error during execution of spawn", e);
    } catch (IOException e) {
      EventHandler handler = actionExecutionContext.getExecutor().getEventHandler();
      handler.handle(
          Event.error(
              "I/O error during sandboxed execution:\n" + Throwables.getStackTraceAsString(e)));
      throw new UserExecException("Could not execute spawn", e);
    }
  }

  private int getTimeout(Spawn spawn) throws UserExecException {
    String timeoutStr = spawn.getExecutionInfo().get("timeout");
    if (timeoutStr != null) {
      try {
        return Integer.parseInt(timeoutStr);
      } catch (NumberFormatException e) {
        throw new UserExecException("could not parse timeout: " + e);
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
      Spawn spawn, Path sandboxPath, ActionExecutionContext executionContext) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    mounts.putAll(mountUsualUnixDirs(sandboxPath));
    mounts.putAll(withRecursedDirs(setupBlazeUtils(sandboxPath)));
    mounts.putAll(withRecursedDirs(mountRunfilesFromManifests(spawn, sandboxPath)));
    mounts.putAll(withRecursedDirs(mountRunfilesFromSuppliers(spawn, sandboxPath)));
    mounts.putAll(withRecursedDirs(mountInputs(spawn, sandboxPath, executionContext)));
    mounts.putAll(withRecursedDirs(mountRunUnderCommand(spawn, sandboxPath)));
    return validateMounts(sandboxPath, withResolvedSymlinks(sandboxPath, mounts));
  }

  /**
   * Validates all mounts against a set of criteria and throws an exception on error.
   *
   * @return an ImmutableMap of all mounts.
   */
  @VisibleForTesting
  static ImmutableMap<Path, Path> validateMounts(Path sandboxPath, Map<Path, Path> mounts) {
    ImmutableMap.Builder<Path, Path> validatedMounts = ImmutableMap.builder();
    for (Entry<Path, Path> mount : mounts.entrySet()) {
      Path target = mount.getKey();
      Path source = mount.getValue();

      // The source must exist.
      Preconditions.checkArgument(source.exists(), "%s does not exist", source.toString());

      // Mounts must always mount into the sandbox, otherwise they might corrupt the host system.
      Preconditions.checkArgument(
          target.startsWith(sandboxPath),
          String.format("(%s -> %s) does not mount into sandbox", source, target));

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
  static MountMap<Path, Path> withResolvedSymlinks(Path sandboxPath, Map<Path, Path> mounts)
      throws IOException {
    MountMap<Path, Path> fixedMounts = new MountMap<>();
    for (Entry<Path, Path> mount : mounts.entrySet()) {
      Path target = mount.getKey();
      Path source = mount.getValue();
      fixedMounts.put(target, source);

      if (source.isSymbolicLink()) {
        source = source.resolveSymbolicLinks();
        target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
        fixedMounts.put(target, source);
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
  private MountMap<Path, Path> mountUsualUnixDirs(Path sandboxPath) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    FileSystem fs = blazeDirs.getFileSystem();
    mounts.put(sandboxPath.getRelative("bin"), fs.getPath("/bin"));
    mounts.put(sandboxPath.getRelative("etc"), fs.getPath("/etc"));
    for (String entry : FilesystemUtils.readdir("/")) {
      if (entry.startsWith("lib")) {
        mounts.put(sandboxPath.getRelative(entry), fs.getRootDirectory().getRelative(entry));
      }
    }
    for (String entry : FilesystemUtils.readdir("/usr")) {
      if (!entry.equals("local")) {
        mounts.put(
            sandboxPath.getRelative("usr").getRelative(entry),
            fs.getPath("/usr").getRelative(entry));
      }
    }
    return mounts;
  }

  /**
   * Mount the embedded tools.
   */
  private MountMap<Path, Path> setupBlazeUtils(Path sandboxPath) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    Path source = blazeDirs.getEmbeddedBinariesRoot().getRelative("build-runfiles");
    Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
    mounts.put(target, source);
    return mounts;
  }

  /**
   * Mount all runfiles that the spawn needs as specified in its runfiles manifests.
   */
  private MountMap<Path, Path> mountRunfilesFromManifests(Spawn spawn, Path sandboxPath)
      throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    for (Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      Path targetDirectory = execRoot.getRelative(manifest.getKey());

      mounts.putAll(parseManifestFile(sandboxPath, targetDirectory, new File(manifestFilePath)));
    }
    return mounts;
  }

  static MountMap<Path, Path> parseManifestFile(
      Path sandboxPath, Path targetDirectory, File manifestFile) throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    for (String line : Files.readLines(manifestFile, Charset.defaultCharset())) {
      String[] fields = line.trim().split(" ");
      Path source;
      Path targetPath = targetDirectory.getRelative(fields[0]);
      Path targetInSandbox = sandboxPath.getRelative(targetPath.asFragment().relativeTo("/"));
      switch (fields.length) {
        case 1:
          source = sandboxPath.getFileSystem().getPath("/dev/null");
          break;
        case 2:
          source = sandboxPath.getFileSystem().getPath(fields[1]);
          break;
        default:
          throw new IllegalStateException("'" + line + "' splits into more than 2 parts");
      }
      mounts.put(targetInSandbox, source);
    }
    return mounts;
  }

  /**
   * Mount all runfiles that the spawn needs as specified via its runfiles suppliers.
   */
  private MountMap<Path, Path> mountRunfilesFromSuppliers(Spawn spawn, Path sandboxPath)
      throws IOException {
    MountMap<Path, Path> mounts = new MountMap<>();
    FileSystem fs = blazeDirs.getFileSystem();
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      if (root.isAbsolute()) {
        root = root.relativeTo("/");
      }
      for (Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        Path source = (sourceArtifact != null) ? sourceArtifact.getPath() : fs.getPath("/dev/null");

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        Path target = sandboxPath.getRelative(root.getRelative(mapping.getKey()));
        mounts.put(target, source);
      }
    }
    return mounts;
  }

  /**
   * Mount all inputs of the spawn.
   */
  private MountMap<Path, Path> mountInputs(
      Spawn spawn, Path sandboxPath, ActionExecutionContext actionExecutionContext)
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
      Path source = execRoot.getRelative(input.getExecPathString());
      Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
      mounts.put(target, source);
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
  private MountMap<Path, Path> mountRunUnderCommand(Spawn spawn, Path sandboxPath) {
    MountMap<Path, Path> mounts = new MountMap<>();

    if (spawn.getResourceOwner() instanceof TestRunnerAction) {
      TestRunnerAction testRunnerAction = ((TestRunnerAction) spawn.getResourceOwner());
      RunUnder runUnder = testRunnerAction.getExecutionSettings().getRunUnder();
      if (runUnder != null && runUnder.getCommand() != null) {
        PathFragment sourceFragment = new PathFragment(runUnder.getCommand());
        Path source;
        if (sourceFragment.isAbsolute()) {
          source = blazeDirs.getFileSystem().getPath(sourceFragment);
        } else if (blazeDirs.getExecRoot().getRelative(sourceFragment).exists()) {
          source = blazeDirs.getExecRoot().getRelative(sourceFragment);
        } else {
          List<Path> searchPath =
              SearchPath.parse(blazeDirs.getFileSystem(), blazeRuntime.getClientEnv().get("PATH"));
          source = SearchPath.which(searchPath, runUnder.getCommand());
        }
        if (source != null) {
          Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
          mounts.put(target, source);
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
