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
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
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
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Strategy that uses sandboxing to execute a process.
 */
@ExecutionStrategy(
  name = {"sandboxed"},
  contextType = SpawnActionContext.class
)
public class LinuxSandboxedStrategy implements SpawnActionContext {
  private static Boolean sandboxingSupported = null;

  public static boolean isSupported(CommandEnvironment env) {
    if (sandboxingSupported == null) {
      sandboxingSupported = LinuxSandboxRunner.isSupported(env);
    }
    return sandboxingSupported.booleanValue();
  }

  private final ExecutorService backgroundWorkers;

  private final BuildRequest buildRequest;
  private final SandboxOptions sandboxOptions;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();
  private final String productName;

  LinuxSandboxedStrategy(
      BuildRequest buildRequest,
      BlazeDirectories blazeDirs,
      ExecutorService backgroundWorkers,
      boolean verboseFailures,
      String productName) {
    this.buildRequest = buildRequest;
    this.sandboxOptions = buildRequest.getOptions(SandboxOptions.class);
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.backgroundWorkers = Preconditions.checkNotNull(backgroundWorkers);
    this.verboseFailures = verboseFailures;
    this.productName = productName;
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
      StandaloneSpawnStrategy standaloneStrategy =
          Preconditions.checkNotNull(executor.getContext(StandaloneSpawnStrategy.class));
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(
          Label.print(spawn.getOwner().getLabel()) + " [" + spawn.getResourceOwner().prettyPrint()
              + "]", spawn.asShellCommand(executor.getExecRoot()));
    }

    executor
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "sandbox"));

    FileOutErr outErr = actionExecutionContext.getFileOutErr();

    // The execId is a unique ID just for this invocation of "exec".
    String execId = uuid + "-" + execCounter.getAndIncrement();

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxExecRoot =
        blazeDirs.getOutputBase().getRelative(productName + "-sandbox").getRelative(execId);

    // Gather all necessary mounts for the sandbox.
    Map<PathFragment, Path> mounts;
    try {
      mounts = getMounts(spawn, actionExecutionContext);
    } catch (IllegalArgumentException | IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }

    Map<String, String> env = new HashMap<>(spawn.getEnvironment());

    ImmutableSet<Path> writablePaths = getWritablePaths(sandboxExecRoot, env);
    ImmutableList<Path> inaccessiblePaths = getInaccessiblePaths();

    int timeout = getTimeout(spawn);

    ImmutableSet.Builder<PathFragment> outputFiles = ImmutableSet.builder();
    for (PathFragment optionalOutput : spawn.getOptionalOutputFiles()) {
      Preconditions.checkArgument(!optionalOutput.isAbsolute());
      outputFiles.add(optionalOutput);
    }
    for (ActionInput output : spawn.getOutputFiles()) {
      outputFiles.add(new PathFragment(output.getExecPathString()));
    }

    try {
      final LinuxSandboxRunner runner =
          new LinuxSandboxRunner(
              execRoot,
              sandboxExecRoot,
              writablePaths,
              inaccessiblePaths,
              verboseFailures,
              sandboxOptions.sandboxDebug);
      try {
        runner.run(
            spawn.getArguments(),
            env,
            outErr,
            mounts,
            outputFiles.build(),
            timeout,
            SandboxHelpers.shouldAllowNetwork(buildRequest, spawn));
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

  /** Gets the list of directories that the spawn will assume to be writable. */
  private ImmutableSet<Path> getWritablePaths(Path sandboxExecRoot, Map<String, String> env) {
    ImmutableSet.Builder<Path> writablePaths = ImmutableSet.builder();
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    if (env.containsKey("TEST_TMPDIR")) {
      Path testTmpDir = sandboxExecRoot.getRelative(env.get("TEST_TMPDIR"));
      writablePaths.add(testTmpDir);
      env.put("TEST_TMPDIR", testTmpDir.getPathString());
    }
    return writablePaths.build();
  }

  private ImmutableList<Path> getInaccessiblePaths() {
    ImmutableList.Builder<Path> inaccessiblePaths = ImmutableList.builder();
    for (String path : sandboxOptions.sandboxBlockPath) {
      inaccessiblePaths.add(blazeDirs.getFileSystem().getPath(path));
    }
    return inaccessiblePaths.build();
  }

  private Map<PathFragment, Path> getMounts(Spawn spawn, ActionExecutionContext executionContext)
      throws IOException, ExecException {
    Map<PathFragment, Path> mounts = new HashMap<>();
    mountRunfilesFromManifests(mounts, spawn);
    mountRunfilesFromSuppliers(mounts, spawn);
    mountFilesFromFilesetManifests(mounts, spawn, executionContext);
    mountInputs(mounts, spawn, executionContext);
    return mounts;
  }

  /** Mount all runfiles that the spawn needs as specified in its runfiles manifests. */
  private void mountRunfilesFromManifests(Map<PathFragment, Path> mounts, Spawn spawn)
      throws IOException, ExecException {
    for (Map.Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      PathFragment targetDirectory = manifest.getKey();

      parseManifestFile(
          blazeDirs.getFileSystem(),
          mounts,
          targetDirectory,
          new File(manifestFilePath),
          false,
          "");
    }
  }

  /** Mount all files that the spawn needs as specified in its fileset manifests. */
  private void mountFilesFromFilesetManifests(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext executionContext)
      throws IOException, ExecException {
    final FilesetActionContext filesetContext =
        executionContext.getExecutor().getContext(FilesetActionContext.class);
    for (Artifact fileset : spawn.getFilesetManifests()) {
      File manifestFile =
          new File(
              execRoot.getPathString(),
              AnalysisUtils.getManifestPathFromFilesetPath(fileset.getExecPath()).getPathString());
      PathFragment targetDirectory = fileset.getExecPath();

      parseManifestFile(
          blazeDirs.getFileSystem(),
          mounts,
          targetDirectory,
          manifestFile,
          true,
          filesetContext.getWorkspaceName());
    }
  }

  /** A parser for the MANIFEST files used by Filesets and runfiles. */
  static void parseManifestFile(
      FileSystem fs,
      Map<PathFragment, Path> mounts,
      PathFragment targetDirectory,
      File manifestFile,
      boolean isFilesetManifest,
      String workspaceName)
      throws IOException, ExecException {
    int lineNum = 0;
    for (String line : Files.readLines(manifestFile, StandardCharsets.UTF_8)) {
      if (isFilesetManifest && (++lineNum % 2 == 0)) {
        continue;
      }
      if (line.isEmpty()) {
        continue;
      }

      String[] fields = line.trim().split(" ");

      // The "target" field is always a relative path that is to be interpreted in this way:
      // (1) If this is a fileset manifest and our workspace name is not empty, the first segment
      // of each "target" path must be the workspace name, which is then stripped before further
      // processing.
      // (2) The "target" path is then appended to the "targetDirectory", which is a path relative
      // to the execRoot. Together, this results in the full path in the execRoot in which place a
      // symlink referring to "source" has to be created (see below).
      PathFragment targetPath;
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

      // The "source" field, if it exists, is always an absolute path and may point to any file in
      // the filesystem (it is not limited to files in the workspace or execroot).
      Path source;
      switch (fields.length) {
        case 1:
          source = fs.getPath("/dev/null");
          break;
        case 2:
          source = fs.getPath(fields[1]);
          break;
        default:
          throw new IllegalStateException("'" + line + "' splits into more than 2 parts");
      }

      mounts.put(targetPath, source);
    }
  }

  /** Mount all runfiles that the spawn needs as specified via its runfiles suppliers. */
  private void mountRunfilesFromSuppliers(Map<PathFragment, Path> mounts, Spawn spawn)
      throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      if (root.isAbsolute()) {
        root = root.relativeTo(execRoot.asFragment());
      }
      for (Map.Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        PathFragment source =
            (sourceArtifact != null) ? sourceArtifact.getExecPath() : new PathFragment("/dev/null");

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        PathFragment target = root.getRelative(mapping.getKey());
        mounts.put(target, execRoot.getRelative(source));
      }
    }
  }

  /** Mount all inputs of the spawn. */
  private void mountInputs(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext actionExecutionContext) {
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());

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
      PathFragment mount = new PathFragment(input.getExecPathString());
      mounts.put(mount, execRoot.getRelative(mount));
    }
  }

  @Override
  public boolean willExecuteRemotely(boolean remotable) {
    return false;
  }

  @Override
  public String toString() {
    return "sandboxed";
  }

  @Override
  public boolean shouldPropagateExecException() {
    return verboseFailures && sandboxOptions.sandboxDebug;
  }
}
