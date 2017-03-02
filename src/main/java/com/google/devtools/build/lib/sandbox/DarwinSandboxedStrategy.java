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
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SearchPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Strategy that uses sandboxing to execute a process, for Darwin */
@ExecutionStrategy(
  name = {"sandboxed"},
  contextType = SpawnActionContext.class
)
public class DarwinSandboxedStrategy extends SandboxStrategy {

  private final ImmutableMap<String, String> clientEnv;
  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean sandboxDebug;
  private final boolean verboseFailures;
  private final String productName;
  private final ImmutableList<Path> confPaths;
  private final SpawnHelpers spawnHelpers;

  private final UUID uuid = UUID.randomUUID();
  private final AtomicInteger execCounter = new AtomicInteger();

  private DarwinSandboxedStrategy(
      BuildRequest buildRequest,
      Map<String, String> clientEnv,
      BlazeDirectories blazeDirs,
      boolean verboseFailures,
      String productName,
      ImmutableList<Path> confPaths,
      SpawnHelpers spawnHelpers) {
    super(
        buildRequest,
        blazeDirs,
        verboseFailures,
        buildRequest.getOptions(SandboxOptions.class));
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.sandboxDebug = buildRequest.getOptions(SandboxOptions.class).sandboxDebug;
    this.verboseFailures = verboseFailures;
    this.productName = productName;
    this.confPaths = confPaths;
    this.spawnHelpers = spawnHelpers;
  }

  public static DarwinSandboxedStrategy create(
      BuildRequest buildRequest,
      Map<String, String> clientEnv,
      BlazeDirectories blazeDirs,
      boolean verboseFailures,
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
        buildRequest,
        clientEnv,
        blazeDirs,
        verboseFailures,
        productName,
        writablePaths.build(),
        new SpawnHelpers(blazeDirs.getExecRoot()));
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

  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    exec(spawn, actionExecutionContext, null);
  }

  @Override
  public void exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    // Certain actions can't run remotely or in a sandbox - pass them on to the standalone strategy.
    if (!spawn.isRemotable() || spawn.hasNoSandbox()) {
      SandboxHelpers.fallbackToNonSandboxedExecution(spawn, actionExecutionContext, executor);
      return;
    }

    EventBus eventBus = actionExecutionContext.getExecutor().getEventBus();
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    eventBus.post(ActionStatusMessage.schedulingStrategy(owner));
    try (ResourceHandle handle =
        ResourceManager.instance().acquireResources(owner, spawn.getLocalResources())) {
      SandboxHelpers.postActionStatusMessage(eventBus, spawn);
      actuallyExec(spawn, actionExecutionContext, writeOutputFiles);
    }
  }

  private void actuallyExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    SandboxHelpers.reportSubcommand(executor, spawn);

    PrintWriter errWriter =
        sandboxDebug
            ? new PrintWriter(actionExecutionContext.getFileOutErr().getErrorStream())
            : null;

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = SandboxHelpers.getSandboxRoot(blazeDirs, productName, uuid, execCounter);
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    if (errWriter != null) {
      errWriter.printf("sandbox root is %s\n", sandboxPath.toString());
      errWriter.printf("working dir is %s\n", sandboxExecRoot.toString());
    }

    ImmutableMap<String, String> spawnEnvironment =
        StandaloneSpawnStrategy.locallyDeterminedEnv(execRoot, productName, spawn.getEnvironment());

    Set<Path> writableDirs = getWritableDirs(sandboxExecRoot, spawn.getEnvironment());

    Path runUnderPath = getRunUnderPath(spawn);

    HardlinkedExecRoot hardlinkedExecRoot =
        new HardlinkedExecRoot(execRoot, sandboxPath, sandboxExecRoot, errWriter);
    ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);
    try {
      hardlinkedExecRoot.createFileSystem(
          getMounts(spawn, actionExecutionContext), outputs, writableDirs);
    } catch (IOException e) {
      throw new UserExecException("Could not prepare sandbox directory", e);
    }

    // Flush our logs before executing the spawn, otherwise they might get overwritten.
    if (errWriter != null) {
      errWriter.flush();
    }

    DarwinSandboxRunner runner =
        new DarwinSandboxRunner(
            sandboxPath,
            sandboxExecRoot,
            getWritableDirs(sandboxExecRoot, spawnEnvironment),
            getInaccessiblePaths(),
            runUnderPath,
            verboseFailures);
    try {
      runSpawn(
          spawn,
          actionExecutionContext,
          spawnEnvironment,
          hardlinkedExecRoot,
          outputs,
          runner,
          writeOutputFiles);
    } finally {
      if (!sandboxDebug) {
        try {
          FileSystemUtils.deleteTree(sandboxPath);
        } catch (IOException e) {
          executor
              .getEventHandler()
              .handle(
                  Event.warn(
                      String.format(
                          "Cannot delete sandbox directory after action execution: %s (%s)",
                          sandboxPath.getPathString(), e)));
        }
      }
    }
  }

  @Override
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env) {
    FileSystem fs = sandboxExecRoot.getFileSystem();
    ImmutableSet.Builder<Path> writableDirs = ImmutableSet.builder();

    writableDirs.addAll(super.getWritableDirs(sandboxExecRoot, env));
    writableDirs.add(fs.getPath("/dev"));

    String sysTmpDir = System.getenv("TMPDIR");
    if (sysTmpDir != null) {
      writableDirs.add(fs.getPath(sysTmpDir));
    }

    writableDirs.add(fs.getPath("/tmp"));

    // ~/Library/Cache and ~/Library/Logs need to be writable (cf. issue #2231).
    Path homeDir = fs.getPath(System.getProperty("user.home"));
    writableDirs.add(homeDir.getRelative("Library/Cache"));
    writableDirs.add(homeDir.getRelative("Library/Logs"));

    // Other temporary directories from getconf.
    for (Path path : confPaths) {
      if (path.exists()) {
        writableDirs.add(path);
      }
    }

    return writableDirs.build();
  }

  @Override
  protected ImmutableSet<Path> getInaccessiblePaths() {
    ImmutableSet.Builder<Path> inaccessiblePaths = ImmutableSet.builder();
    inaccessiblePaths.addAll(super.getInaccessiblePaths());
    inaccessiblePaths.add(blazeDirs.getWorkspace());
    inaccessiblePaths.add(execRoot);
    return inaccessiblePaths.build();
  }

  @Override
  public Map<PathFragment, Path> getMounts(Spawn spawn, ActionExecutionContext executionContext)
      throws ExecException {
    try {
      Map<PathFragment, Path> mounts = new HashMap<>();
      spawnHelpers.mountInputs(mounts, spawn, executionContext);

      Map<PathFragment, Path> unfinalized = new HashMap<>();
      spawnHelpers.mountRunfilesFromSuppliers(unfinalized, spawn);
      spawnHelpers.mountFilesFromFilesetManifests(unfinalized, spawn, executionContext);
      mounts.putAll(finalizeLinks(unfinalized));

      return mounts;
    } catch (IllegalArgumentException | IOException e) {
      throw new EnvironmentalExecException("Could not prepare mounts for sandbox execution", e);
    }
  }

  private Map<PathFragment, Path> finalizeLinks(Map<PathFragment, Path> unfinalized)
      throws IOException {
    HashMap<PathFragment, Path> finalizedLinks = new HashMap<>();
    for (Map.Entry<PathFragment, Path> mount : unfinalized.entrySet()) {
      PathFragment target = mount.getKey();
      Path source = mount.getValue();

      // If the source is null, the target is supposed to be an empty file. In this case we don't
      // have to deal with finalizing the link.
      if (source == null) {
        finalizedLinks.put(target, source);
        continue;
      }

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
    return finalizedLinks;
  }

  private void finalizeLinksPath(
      Map<PathFragment, Path> finalizedMounts, PathFragment target, Path source, FileStatus stat)
      throws IOException {
    // The source must exist.
    Preconditions.checkArgument(stat != null, "%s does not exist", source.toString());
    finalizedMounts.put(target, source);
  }

  /**
   * If a --run_under= option is set and refers to a command via its path (as opposed to via its
   * label), we have to mount this. Note that this is best effort and works fine for shell scripts
   * and small binaries, but we can't track any further dependencies of this command.
   *
   * <p>If --run_under= refers to a label, it is automatically provided in the spawn's input files,
   * so mountInputs() will catch that case.
   */
  private Path getRunUnderPath(Spawn spawn) {
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
          return mount;
        }
      }
    }
    return null;
  }
}
