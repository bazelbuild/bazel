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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/** Strategy that uses sandboxing to execute a process, for Darwin */
@ExecutionStrategy(
  name = {"sandboxed", "darwin-sandbox"},
  contextType = SpawnActionContext.class
)
public class DarwinSandboxedStrategy extends SandboxStrategy {

  private final Path execRoot;
  private final boolean sandboxDebug;
  private final boolean verboseFailures;
  private final String productName;
  private final SpawnInputExpander spawnInputExpander;

  /**
   * The set of directories that always should be writable, independent of the Spawn itself.
   *
   * <p>We cache this, because creating it involves executing {@code getconf}, which is expensive.
   */
  private final ImmutableSet<Path> alwaysWritableDirs;
  private final LocalEnvProvider localEnvProvider;

  private DarwinSandboxedStrategy(
      CommandEnvironment cmdEnv,
      BuildRequest buildRequest,
      Path sandboxBase,
      boolean verboseFailures,
      String productName,
      ImmutableSet<Path> alwaysWritableDirs) {
    super(
        cmdEnv,
        buildRequest,
        sandboxBase,
        verboseFailures,
        buildRequest.getOptions(SandboxOptions.class));
    this.execRoot = cmdEnv.getExecRoot();
    this.sandboxDebug = buildRequest.getOptions(SandboxOptions.class).sandboxDebug;
    this.verboseFailures = verboseFailures;
    this.productName = productName;
    this.alwaysWritableDirs = alwaysWritableDirs;
    this.spawnInputExpander = new SpawnInputExpander(false);
    this.localEnvProvider = new XCodeLocalEnvProvider();
  }

  public static DarwinSandboxedStrategy create(
      CommandEnvironment cmdEnv,
      BuildRequest buildRequest,
      Path sandboxBase,
      boolean verboseFailures,
      String productName)
      throws IOException {
    return new DarwinSandboxedStrategy(
        cmdEnv,
        buildRequest,
        sandboxBase,
        verboseFailures,
        productName,
        getAlwaysWritableDirs(cmdEnv.getDirectories().getFileSystem()));
  }

  private static void addPathToSetIfExists(FileSystem fs, Set<Path> paths, String path)
      throws IOException {
    if (path != null) {
      addPathToSetIfExists(paths, fs.getPath(path));
    }
  }

  private static void addPathToSetIfExists(Set<Path> paths, Path path) throws IOException {
    if (path.exists()) {
      paths.add(path.resolveSymbolicLinks());
    }
  }

  private static ImmutableSet<Path> getAlwaysWritableDirs(FileSystem fs) throws IOException {
    HashSet<Path> writableDirs = new HashSet<>();

    addPathToSetIfExists(fs, writableDirs, "/dev");
    addPathToSetIfExists(fs, writableDirs, System.getenv("TMPDIR"));
    addPathToSetIfExists(fs, writableDirs, "/tmp");
    addPathToSetIfExists(fs, writableDirs, "/private/tmp");
    addPathToSetIfExists(fs, writableDirs, "/private/var/tmp");

    // On macOS, in addition to what is specified in $TMPDIR, two other temporary directories may be
    // written to by processes. We have to get their location by calling "getconf".
    addPathToSetIfExists(fs, writableDirs, getConfStr("DARWIN_USER_TEMP_DIR"));
    addPathToSetIfExists(fs, writableDirs, getConfStr("DARWIN_USER_CACHE_DIR"));

    // ~/Library/Cache and ~/Library/Logs need to be writable (cf. issue #2231).
    Path homeDir = fs.getPath(System.getProperty("user.home"));
    addPathToSetIfExists(writableDirs, homeDir.getRelative("Library/Cache"));
    addPathToSetIfExists(writableDirs, homeDir.getRelative("Library/Logs"));

    // Certain Xcode tools expect to be able to write to this path.
    addPathToSetIfExists(writableDirs, homeDir.getRelative("Library/Developer"));

    return ImmutableSet.copyOf(writableDirs);
  }

  /**
   * Returns the value of a POSIX or X/Open system configuration variable.
   */
  private static String getConfStr(String confVar) throws IOException {
    String[] commandArr = new String[2];
    commandArr[0] = "/usr/bin/getconf";
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
  protected void actuallyExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException, IOException {
    actionExecutionContext
        .getEventBus()
        .post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "darwin-sandbox"));
    SandboxHelpers.reportSubcommand(actionExecutionContext, spawn);

    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = getSandboxRoot();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    Map<String, String> spawnEnvironment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, productName);

    HashSet<Path> writableDirs = new HashSet<>(alwaysWritableDirs);
    ImmutableSet<Path> extraWritableDirs = getWritableDirs(sandboxExecRoot, spawnEnvironment);
    writableDirs.addAll(extraWritableDirs);

    SymlinkedExecRoot symlinkedExecRoot = new SymlinkedExecRoot(sandboxExecRoot);
    ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);
    symlinkedExecRoot.createFileSystem(
        SandboxHelpers.getInputFiles(
            spawnInputExpander, this.execRoot, spawn, actionExecutionContext),
        outputs,
        writableDirs);

    // This will add the resolved versions of the spawn-dependant writable paths (e.g. its execroot
    // or TEST_TMPDIR) to the set, now that they have been created by the SymlinkedExecRoot.
    for (Path extraWritableDir : extraWritableDirs) {
      addPathToSetIfExists(writableDirs, extraWritableDir);
    }

    DarwinSandboxRunner runner =
        new DarwinSandboxRunner(
            sandboxPath, sandboxExecRoot, writableDirs, getInaccessiblePaths(), verboseFailures);
    try {
      runSpawn(
          spawn,
          actionExecutionContext,
          spawnEnvironment,
          symlinkedExecRoot,
          outputs,
          runner,
          writeOutputFiles);
    } finally {
      if (!sandboxDebug) {
        try {
          FileSystemUtils.deleteTree(sandboxPath);
        } catch (IOException e) {
          // This usually means that the Spawn itself exited, but still has children running that
          // we couldn't wait for, which now block deletion of the sandbox directory. On Linux this
          // should never happen, as we use PID namespaces and where they are not available the
          // subreaper feature to make sure all children have been reliably killed before returning,
          // but on other OS this might not always work. The SandboxModule will try to delete them
          // again when the build is all done, at which point it hopefully works, so let's just go
          // on here.
        }
      }
    }
  }
}
