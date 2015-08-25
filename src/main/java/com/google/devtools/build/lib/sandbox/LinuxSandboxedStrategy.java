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

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
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
    String execId = uuid + "-" + Integer.toString(execCounter.getAndIncrement());

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath =
        execRoot.getRelative(Constants.PRODUCT_NAME + "-sandbox").getRelative(execId);

    ImmutableMultimap<Path, Path> mounts;
    try {
      // Gather all necessary mounts for the sandbox.
      mounts = getMounts(spawn, sandboxPath, actionExecutionContext);
    } catch (IOException e) {
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

  private ImmutableMultimap<Path, Path> getMounts(
      Spawn spawn, Path sandboxPath, ActionExecutionContext actionExecutionContext)
      throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
    mounts.putAll(mountUsualUnixDirs(sandboxPath));
    mounts.putAll(setupBlazeUtils(sandboxPath));
    mounts.putAll(mountRunfilesFromManifests(spawn, sandboxPath));
    mounts.putAll(mountRunfilesFromSuppliers(spawn, sandboxPath));
    mounts.putAll(mountRunfilesForTests(spawn, sandboxPath));
    mounts.putAll(mountInputs(spawn, sandboxPath, actionExecutionContext));
    mounts.putAll(mountRunUnderCommand(spawn, sandboxPath));

    SetMultimap<Path, Path> fixedMounts = LinkedHashMultimap.create();
    for (Entry<Path, Path> mount : mounts.build().entries()) {
      Path source = mount.getKey();
      Path target = mount.getValue();
      validateAndAddMount(sandboxPath, fixedMounts, source, target);

      // Iteratively resolve symlinks and mount the whole chain. Take care not to run into a cyclic
      // symlink - when we already processed the source once, we can exit the loop. Skyframe will
      // catch cyclic symlinks for declared inputs, but this won't help if there is one in the parts
      // of the host system that we mount.
      Set<Path> seenSources = new HashSet<>();
      while (source.isSymbolicLink() && seenSources.add(source)) {
        source = source.getParentDirectory().getRelative(source.readSymbolicLink());
        target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));

        validateAndAddMount(sandboxPath, fixedMounts, source, target);
      }
    }
    return ImmutableMultimap.copyOf(fixedMounts);
  }

  /**
   * Adds the new mount ("source" -> "target") to "mounts" after doing some validations on it.
   *
   * @return true if the mount was added to the multimap, or false if the multimap already contained
   *         the mount.
   */
  private static boolean validateAndAddMount(
      Path sandboxPath, SetMultimap<Path, Path> mounts, Path source, Path target) {
    // The source must exist.
    Preconditions.checkArgument(source.exists(), source.toString() + " does not exist");

    // We cannot mount two different things onto the same target.
    if (!mounts.containsEntry(source, target) && mounts.containsValue(target)) {
      // There is a conflicting entry, find it and error out.
      for (Entry<Path, Path> mount : mounts.entries()) {
        if (mount.getValue().equals(target)) {
          throw new IllegalStateException(
              String.format(
                  "Cannot mount both '%s' and '%s' onto '%s'", mount.getKey(), source, target));
        }
      }
    }

    // Mounts must always mount into the sandbox, otherwise they might corrupt the host system.
    Preconditions.checkArgument(
        target.startsWith(sandboxPath),
        String.format("(%s -> %s) does not mount into sandbox", source, target));

    return mounts.put(source, target);
  }

  /**
   * Mount a certain set of unix directories to make the usual tools and libraries available to the
   * spawn that runs.
   */
  private ImmutableMultimap<Path, Path> mountUsualUnixDirs(Path sandboxPath) throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
    FileSystem fs = blazeDirs.getFileSystem();
    mounts.put(fs.getPath("/bin"), sandboxPath.getRelative("bin"));
    mounts.put(fs.getPath("/etc"), sandboxPath.getRelative("etc"));
    for (String entry : FilesystemUtils.readdir("/")) {
      if (entry.startsWith("lib")) {
        mounts.put(fs.getRootDirectory().getRelative(entry), sandboxPath.getRelative(entry));
      }
    }
    for (String entry : FilesystemUtils.readdir("/usr")) {
      if (!entry.equals("local")) {
        mounts.put(
            fs.getPath("/usr").getRelative(entry),
            sandboxPath.getRelative("usr").getRelative(entry));
      }
    }
    return mounts.build();
  }

  /**
   * Mount the embedded tools.
   */
  private ImmutableMultimap<Path, Path> setupBlazeUtils(Path sandboxPath) throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
    Path source = blazeDirs.getEmbeddedBinariesRoot().getRelative("build-runfiles");
    Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
    mounts.put(source, target);
    return mounts.build();
  }

  /**
   * Mount all runfiles that the spawn needs as specified in its runfiles manifests.
   */
  private ImmutableMultimap<Path, Path> mountRunfilesFromManifests(Spawn spawn, Path sandboxPath)
      throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
    FileSystem fs = blazeDirs.getFileSystem();
    for (Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      Path targetDirectory = execRoot.getRelative(manifest.getKey());
      for (String line : Files.readLines(new File(manifestFilePath), Charset.defaultCharset())) {
        String[] fields = line.split(" ");
        Preconditions.checkState(
            fields.length == 2, "'" + line + "' does not split into exactly 2 parts");
        Path source = fs.getPath(fields[1]);
        Path targetPath = targetDirectory.getRelative(fields[0]);
        Path targetInSandbox = sandboxPath.getRelative(targetPath.asFragment().relativeTo("/"));
        mounts.put(source, targetInSandbox);
      }
    }
    return mounts.build();
  }

  /**
   * Mount all runfiles that the spawn needs as specified via its runfiles suppliers.
   */
  private ImmutableMultimap<Path, Path> mountRunfilesFromSuppliers(Spawn spawn, Path sandboxPath)
      throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
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
        mounts.put(source, target);
      }
    }
    return mounts.build();
  }

  /**
   * Tests are a special case and we have to mount the TEST_SRCDIR where the test expects it to be
   * and also provide a TEST_TMPDIR to the test where it can store temporary files.
   */
  private ImmutableMultimap<Path, Path> mountRunfilesForTests(Spawn spawn, Path sandboxPath)
      throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();
    FileSystem fs = blazeDirs.getFileSystem();
    if (spawn.getEnvironment().containsKey("TEST_TMPDIR")) {
      Path source = fs.getPath(spawn.getEnvironment().get("TEST_TMPDIR"));
      Path target = sandboxPath.getRelative(source.asFragment().relativeTo("/"));
      FileSystemUtils.createDirectoryAndParents(target);
    }
    return mounts.build();
  }

  /**
   * Mount all inputs of the spawn.
   */
  private ImmutableMultimap<Path, Path> mountInputs(
      Spawn spawn, Path sandboxPath, ActionExecutionContext actionExecutionContext)
      throws IOException {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();

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
      mounts.put(source, target);
    }
    return mounts.build();
  }

  /**
   * If a --run_under= option is set and refers to a command via its path (as opposed to via its
   * label), we have to mount this. Note that this is best effort and works fine for shell scripts
   * and small binaries, but we can't track any further dependencies of this command.
   *
   * <p>If --run_under= refers to a label, it is automatically provided in the spawn's input files,
   * so mountInputs() will catch that case.
   */
  private ImmutableMultimap<Path, Path> mountRunUnderCommand(Spawn spawn, Path sandboxPath) {
    ImmutableMultimap.Builder<Path, Path> mounts = ImmutableMultimap.builder();

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
          mounts.put(source, target);
        }
      }
    }
    return mounts.build();
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
