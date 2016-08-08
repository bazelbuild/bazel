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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SearchPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/** Strategy that uses sandboxing to execute a process, for Darwin */
@ExecutionStrategy(
  name = {"sandboxed"},
  contextType = SpawnActionContext.class
)
public class DarwinSandboxedStrategy implements SpawnActionContext {

  private final ExecutorService backgroundWorkers;
  private final ImmutableMap<String, String> clientEnv;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final SandboxOptions sandboxOptions;
  private final boolean verboseFailures;
  private final boolean unblockNetwork;
  private final String productName;
  private final ImmutableList<Path> confPaths;

  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();

  private DarwinSandboxedStrategy(
      SandboxOptions options,
      Map<String, String> clientEnv,
      BlazeDirectories blazeDirs,
      ExecutorService backgroundWorkers,
      boolean verboseFailures,
      boolean unblockNetwork,
      String productName,
      ImmutableList<Path> confPaths) {
    this.sandboxOptions = options;
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.backgroundWorkers = Preconditions.checkNotNull(backgroundWorkers);
    this.verboseFailures = verboseFailures;
    this.unblockNetwork = unblockNetwork;
    this.productName = productName;
    this.confPaths = confPaths;
  }

  public static DarwinSandboxedStrategy create(
      SandboxOptions options,
      Map<String, String> clientEnv,
      BlazeDirectories blazeDirs,
      ExecutorService backgroundWorkers,
      boolean verboseFailures,
      boolean unblockNetwork,
      String productName)
      throws IOException {
    // On OS X, in addition to what is specified in $TMPDIR, two other temporary directories may be
    // written to by processes. We have to get their location by calling "getconf".
    List<String> confVars = ImmutableList.of("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR");
    ImmutableList.Builder<Path> writablePaths = ImmutableList.builder();
    for (String confVar : confVars) {
      Path path = blazeDirs.getFileSystem().getPath(getConfStr(confVar));
      if (path.exists()) {
        writablePaths.add(path);
      }
    }

    return new DarwinSandboxedStrategy(
        options,
        clientEnv,
        blazeDirs,
        backgroundWorkers,
        verboseFailures,
        unblockNetwork,
        productName,
        writablePaths.build());
  }

  /**
   * Returns the value of a POSIX or X/Open system configuration variable.
   */
  private static String getConfStr(String confVar) throws IOException {
    String[] commandArr = new String[2];
    commandArr[0] = "getconf";
    commandArr[1] = confVar;
    Command cmd = new Command(commandArr);
    CommandResult res;
    try {
      res = cmd.execute();
    } catch (CommandException e) {
      throw new IOException("getconf failed", e);
    }
    return new String(res.getStdout(), UTF_8).trim();
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
          Label.print(spawn.getOwner().getLabel())
              + " ["
              + spawn.getResourceOwner().prettyPrint()
              + "]",
          spawn.asShellCommand(executor.getExecRoot()));
    }

    executor
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "sandbox"));

    FileOutErr outErr = actionExecutionContext.getFileOutErr();

    // The execId is a unique ID just for this invocation of "exec".
    String execId = uuid + "-" + execCounter.getAndIncrement();

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath =
        blazeDirs.getOutputBase().getRelative(productName + "-sandbox").getRelative(execId);

    ImmutableSet<PathFragment> createDirs =
        createImportantDirs(spawn.getEnvironment(), sandboxPath);

    int timeout = getTimeout(spawn);

    ImmutableSet.Builder<PathFragment> outputFiles = ImmutableSet.<PathFragment>builder();
    final DarwinSandboxRunner runner =
        getRunnerForExec(spawn, actionExecutionContext, sandboxPath, createDirs, outputFiles);

    try {
      runner.run(
          spawn.getArguments(), spawn.getEnvironment(), outErr, outputFiles.build(), timeout);
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    } finally {
      // By deleting the sandbox directory in the background, we avoid having to wait for it to
      // complete before returning from the action, which improves performance.
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
  }

  private DarwinSandboxRunner getRunnerForExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      Path sandboxPath,
      ImmutableSet<PathFragment> createDirs,
      ImmutableSet.Builder<PathFragment> outputFiles)
      throws ExecException {
    ImmutableMap<PathFragment, Path> linkPaths;
    try {
      // Gather all necessary linkPaths for the sandbox.
      linkPaths = getMounts(spawn, actionExecutionContext);
    } catch (IllegalArgumentException | IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }

    for (PathFragment optionalOutput : spawn.getOptionalOutputFiles()) {
      Preconditions.checkArgument(!optionalOutput.isAbsolute());
      outputFiles.add(optionalOutput);
    }
    for (ActionInput output : spawn.getOutputFiles()) {
      outputFiles.add(new PathFragment(output.getExecPathString()));
    }

    DarwinSandboxRunner runner;
    try {
      Path sandboxConfigPath =
          generateScriptFile(
              sandboxPath,
              unblockNetwork || spawn.getExecutionInfo().containsKey("requires-network"));
      runner =
          new DarwinSandboxRunner(
              execRoot,
              sandboxPath,
              sandboxPath.getRelative("execroot"),
              sandboxConfigPath,
              linkPaths,
              createDirs,
              verboseFailures,
              sandboxOptions.sandboxDebug);
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    }

    return runner;
  }

  private ImmutableSet<PathFragment> createImportantDirs(
      Map<String, String> env, Path sandboxPath) {
    ImmutableSet.Builder<PathFragment> dirs = ImmutableSet.builder();
    if (env.containsKey("TEST_TMPDIR")) {
      PathFragment testTmpDir = new PathFragment(env.get("TEST_TMPDIR"));
      if (!testTmpDir.isAbsolute()) {
        testTmpDir = sandboxPath.asFragment().getRelative("execroot").getRelative(testTmpDir);
      }
      dirs.add(testTmpDir);
    }
    return dirs.build();
  }

  private Path generateScriptFile(Path sandboxPath, boolean network) throws IOException {
    FileSystemUtils.createDirectoryAndParents(sandboxPath);
    Path sandboxConfigPath =
        sandboxPath.getParentDirectory().getRelative(sandboxPath.getBaseName() + ".sb");
    try (PrintWriter out = new PrintWriter(sandboxConfigPath.getOutputStream())) {
      out.println("(version 1)");
      out.println("(debug deny)");
      out.println("(deny default)");
      out.println("(allow signal)");
      out.println("(allow system*)");
      out.println("(allow process*)");
      out.println("(allow sysctl*)");
      out.println("(allow mach-lookup)");
      out.println("(allow ipc*)");
      out.println("(allow network* (local ip \"localhost:*\"))");
      out.println("(allow network* (remote ip \"localhost:*\"))");
      out.println("(allow network* (remote unix-socket (subpath \"/\")))");
      out.println("(allow network* (local unix-socket (subpath \"/\")))");
      // check network
      if (network) {
        out.println("(allow network*)");
      }

      // except workspace
      out.println("(deny file-read-data (subpath \"" + blazeDirs.getWorkspace() + "\"))");
      // except exec_root
      out.println("(deny file-read-data (subpath \"" + execRoot + "\"))");
      // almost everything is readable
      out.println("(allow file-read* (subpath \"/\"))");

      allowWriteSubpath(out, blazeDirs.getFileSystem().getPath("/dev"));

      // Write access to sandbox
      allowWriteSubpath(out, sandboxPath);

      // Write access to the system TMPDIR
      Path sysTmpDir = blazeDirs.getFileSystem().getPath(System.getenv("TMPDIR"));
      allowWriteSubpath(out, sysTmpDir);
      allowWriteSubpath(out, blazeDirs.getFileSystem().getPath("/tmp"));

      // Other tmpdir from getconf
      for (Path path : confPaths) {
        if (path.exists()) {
          allowWriteSubpath(out, path);
        }
      }
    }

    return sandboxConfigPath;
  }

  private void allowWriteSubpath(PrintWriter out, Path path) throws IOException {
    out.println("(allow file* (subpath \"" + path.getPathString() + "\"))");
    Path resolvedPath = path.resolveSymbolicLinks();
    if (!resolvedPath.equals(path)) {
      out.println("(allow file* (subpath \"" + resolvedPath.getPathString() + "\"))");
    }
  }

  private ImmutableMap<PathFragment, Path> getMounts(
      Spawn spawn, ActionExecutionContext executionContext) throws IOException, ExecException {
    ImmutableMap.Builder<PathFragment, Path> result = new ImmutableMap.Builder<>();
    result.putAll(mountInputs(spawn, executionContext));

    Map<PathFragment, Path> unfinalized = new HashMap<>();
    unfinalized.putAll(mountRunfilesFromManifests(spawn));
    unfinalized.putAll(mountRunfilesFromSuppliers(spawn));
    unfinalized.putAll(mountFilesFromFilesetManifests(spawn, executionContext));
    unfinalized.putAll(mountRunUnderCommand(spawn));
    result.putAll(finalizeLinks(unfinalized));

    return result.build();
  }

  private ImmutableMap<PathFragment, Path> finalizeLinks(Map<PathFragment, Path> unfinalized)
      throws IOException {
    ImmutableMap.Builder<PathFragment, Path> finalizedLinks = new ImmutableMap.Builder<>();
    for (Map.Entry<PathFragment, Path> mount : unfinalized.entrySet()) {
      PathFragment target = mount.getKey();
      Path source = mount.getValue();

      FileStatus stat = source.statNullable(Symlinks.NOFOLLOW);

      if (stat != null && stat.isDirectory()) {
        for (Path subSource : FileSystemUtils.traverseTree(source, Predicates.alwaysTrue())) {
          PathFragment subTarget = target.getRelative(subSource.relativeTo(source));
          finalizeLinksPath(
              finalizedLinks, subTarget, subSource, subSource.statNullable(Symlinks.NOFOLLOW));
        }
      } else {
        finalizeLinksPath(finalizedLinks, target, source, stat);
      }
    }
    return finalizedLinks.build();
  }

  private void finalizeLinksPath(
      ImmutableMap.Builder<PathFragment, Path> finalizedMounts,
      PathFragment target,
      Path source,
      FileStatus stat)
      throws IOException {
    // The source must exist.
    Preconditions.checkArgument(stat != null, "%s does not exist", source.toString());
    finalizedMounts.put(target, source);
  }

  /** Mount all inputs of the spawn. */
  private Map<PathFragment, Path> mountInputs(
      Spawn spawn, ActionExecutionContext actionExecutionContext) {
    Map<PathFragment, Path> mounts = new HashMap<>();

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
      Path mount = execRoot.getRelative(input.getExecPathString());
      mounts.put(new PathFragment(input.getExecPathString()), mount);
    }
    return mounts;
  }

  /** Mount all runfiles that the spawn needs as specified in its runfiles manifests. */
  private Map<PathFragment, Path> mountRunfilesFromManifests(Spawn spawn)
      throws IOException, ExecException {
    Map<PathFragment, Path> mounts = new HashMap<>();
    for (Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());

      mounts.putAll(parseManifestFile(manifest.getKey(), new File(manifestFilePath), false, ""));
    }
    return mounts;
  }

  /** Mount all files that the spawn needs as specified in its fileset manifests. */
  private Map<PathFragment, Path> mountFilesFromFilesetManifests(
      Spawn spawn, ActionExecutionContext executionContext) throws IOException, ExecException {
    final FilesetActionContext filesetContext =
        executionContext.getExecutor().getContext(FilesetActionContext.class);
    Map<PathFragment, Path> mounts = new HashMap<>();
    for (Artifact fileset : spawn.getFilesetManifests()) {
      Path manifest =
          execRoot.getRelative(AnalysisUtils.getManifestPathFromFilesetPath(fileset.getExecPath()));

      mounts.putAll(
          parseManifestFile(
              fileset.getExecPath(),
              manifest.getPathFile(),
              true,
              filesetContext.getWorkspaceName()));
    }
    return mounts;
  }

  /** Mount all runfiles that the spawn needs as specified via its runfiles suppliers. */
  private Map<PathFragment, Path> mountRunfilesFromSuppliers(Spawn spawn) throws IOException {
    Map<PathFragment, Path> mounts = new HashMap<>();
    FileSystem fs = blazeDirs.getFileSystem();
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root =
          fs.getRootDirectory().getRelative(rootAndMappings.getKey()).relativeTo(execRoot);
      for (Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        Path source = (sourceArtifact != null) ? sourceArtifact.getPath() : fs.getPath("/dev/null");

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        PathFragment target = root.getRelative(mapping.getKey());
        mounts.put(target, source);
      }
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
  private Map<PathFragment, Path> mountRunUnderCommand(Spawn spawn) {
    Map<PathFragment, Path> mounts = new HashMap<>();

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
        // only need to hardlink when under workspace
        Path workspace = blazeDirs.getWorkspace();
        if (mount != null && mount.startsWith(workspace)) {
          mounts.put(mount.relativeTo(workspace), mount);
        }
      }
    }
    return mounts;
  }

  private Map<PathFragment, Path> parseManifestFile(
      PathFragment targetDirectory,
      File manifestFile,
      boolean isFilesetManifest,
      String workspaceName)
      throws IOException, ExecException {
    Map<PathFragment, Path> mounts = new HashMap<>();
    int lineNum = 0;
    for (String line : Files.readLines(manifestFile, StandardCharsets.UTF_8)) {
      if (isFilesetManifest && (++lineNum % 2 == 0)) {
        continue;
      }
      if (line.isEmpty()) {
        continue;
      }

      String[] fields = line.trim().split(" ");

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

      Path source;
      switch (fields.length) {
        case 1:
          source = blazeDirs.getFileSystem().getPath("/dev/null");
          break;
        case 2:
          source = blazeDirs.getFileSystem().getPath(fields[1]);
          break;
        default:
          throw new IllegalStateException("'" + line + "' splits into more than 2 parts");
      }

      mounts.put(targetPath, source);
    }
    return mounts;
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
