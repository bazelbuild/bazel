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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ProcessWrapperUtil;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Spawn runner that uses Darwin (macOS) sandboxing to execute a process. */
@ExecutionStrategy(
  name = {"sandboxed", "darwin-sandbox"},
  contextType = SpawnActionContext.class
)
final class DarwinSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {
  private static final String SANDBOX_EXEC = "/usr/bin/sandbox-exec";

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    if (OS.getCurrent() != OS.DARWIN) {
      return false;
    }
    if (!ProcessWrapperUtil.isSupported(cmdEnv)) {
      return false;
    }

    List<String> args = new ArrayList<>();
    args.add(SANDBOX_EXEC);
    args.add("-p");
    args.add("(version 1) (allow default)");
    args.add("/usr/bin/true");

    ImmutableMap<String, String> env = ImmutableMap.of();
    File cwd = new File("/usr/bin");

    Command cmd = new Command(args.toArray(new String[0]), env, cwd);
    try {
      cmd.execute(ByteStreams.nullOutputStream(), ByteStreams.nullOutputStream());
    } catch (CommandException e) {
      return false;
    }

    return true;
  }

  private final Path execRoot;
  private final boolean allowNetwork;
  private final String productName;
  private final Path processWrapper;
  private final int timeoutGraceSeconds;

  /**
   * The set of directories that always should be writable, independent of the Spawn itself.
   *
   * <p>We cache this, because creating it involves executing {@code getconf}, which is expensive.
   */
  private final ImmutableSet<Path> alwaysWritableDirs;
  private final LocalEnvProvider localEnvProvider;

  DarwinSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      int timeoutGraceSeconds)
      throws IOException {
    super(cmdEnv, sandboxBase);
    this.execRoot = cmdEnv.getExecRoot();
    this.allowNetwork = SandboxHelpers.shouldAllowNetwork(cmdEnv.getOptions());
    this.productName = productName;
    this.alwaysWritableDirs = getAlwaysWritableDirs(cmdEnv.getRuntime().getFileSystem());
    this.processWrapper = ProcessWrapperUtil.getProcessWrapper(cmdEnv);
    this.localEnvProvider = new XCodeLocalEnvProvider();
    this.timeoutGraceSeconds = timeoutGraceSeconds;
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
    addPathToSetIfExists(fs, writableDirs, "/tmp");
    addPathToSetIfExists(fs, writableDirs, "/private/tmp");
    addPathToSetIfExists(fs, writableDirs, "/private/var/tmp");

    // On macOS, processes may write to not only $TMPDIR but also to two other temporary
    // directories. We have to get their location by calling "getconf".
    addPathToSetIfExists(fs, writableDirs, getConfStr("DARWIN_USER_TEMP_DIR"));
    addPathToSetIfExists(fs, writableDirs, getConfStr("DARWIN_USER_CACHE_DIR"));
    // We don't add any value for $TMPDIR here, instead we compute its value later in
    // {@link #actuallyExec} and add it as a writable directory in
    // {@link AbstractSandboxSpawnRunner#getWritableDirs}.

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
  protected SpawnResult actuallyExec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, IOException, InterruptedException {
    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = getSandboxRoot();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    // Each sandboxed action runs in its own execroot, so we don't need to make the temp directory's
    // name unique (like we have to with standalone execution strategy).
    Path tmpDir = sandboxExecRoot.getRelative("tmp");

    Map<String, String> spawnEnvironment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, tmpDir, productName);

    final HashSet<Path> writableDirs = new HashSet<>(alwaysWritableDirs);
    ImmutableSet<Path> extraWritableDirs =
        getWritableDirs(sandboxExecRoot, spawnEnvironment, tmpDir);
    writableDirs.addAll(extraWritableDirs);

    ImmutableSet<PathFragment> outputs = SandboxHelpers.getOutputFiles(spawn);

    final Path sandboxConfigPath = sandboxPath.getRelative("sandbox.sb");
    Duration timeout = policy.getTimeout();
    List<String> arguments =
        computeCommandLine(spawn, timeout, sandboxConfigPath, timeoutGraceSeconds);
    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, tmpDir, productName);

    boolean allowNetworkForThisSpawn = allowNetwork || Spawns.requiresNetwork(spawn);
    SandboxedSpawn sandbox = new SymlinkedSandboxedSpawn(
        sandboxPath,
        sandboxExecRoot,
        arguments,
        environment,
        SandboxHelpers.getInputFiles(spawn, policy, execRoot),
        outputs,
        writableDirs) {
      @Override
      public void createFileSystem() throws IOException {
        super.createFileSystem();
        writeConfig(
            sandboxConfigPath, writableDirs, getInaccessiblePaths(), allowNetworkForThisSpawn);
      }
    };
    return runSpawn(spawn, sandbox, policy, execRoot, tmpDir, timeout);
  }

  private List<String> computeCommandLine(
      Spawn spawn, Duration timeout, Path sandboxConfigPath, int timeoutGraceSeconds) {
    List<String> commandLineArgs = new ArrayList<>();
    commandLineArgs.add(SANDBOX_EXEC);
    commandLineArgs.add("-f");
    commandLineArgs.add(sandboxConfigPath.getPathString());
    commandLineArgs.addAll(
        ProcessWrapperUtil.getCommandLine(
            processWrapper, spawn.getArguments(), timeout, timeoutGraceSeconds));
    return commandLineArgs;
  }

  private void writeConfig(
      Path sandboxConfigPath,
      Set<Path> writableDirs,
      Set<Path> inaccessiblePaths,
      boolean allowNetwork) throws IOException {
    try (PrintWriter out =
        new PrintWriter(
            new BufferedWriter(
                new OutputStreamWriter(sandboxConfigPath.getOutputStream(), UTF_8)))) {
      // Note: In Apple's sandbox configuration language, the *last* matching rule wins.
      out.println("(version 1)");
      out.println("(debug deny)");
      out.println("(allow default)");

      if (!allowNetwork) {
        out.println("(deny network*)");
        out.println("(allow network* (local ip \"localhost:*\"))");
        out.println("(allow network* (remote ip \"localhost:*\"))");
        out.println("(allow network* (remote unix-socket))");
      }

      // By default, everything is read-only.
      out.println("(deny file-write*)");

      out.println("(allow file-write*");
      for (Path path : writableDirs) {
        out.println("    (subpath \"" + path.getPathString() + "\")");
      }
      out.println(")");

      if (!inaccessiblePaths.isEmpty()) {
        out.println("(deny file-read*");
        // The sandbox configuration file is not part of a cache key and sandbox-exec doesn't care
        // about ordering of paths in expressions, so it's fine if the iteration order is random.
        for (Path inaccessiblePath : inaccessiblePaths) {
          out.println("    (subpath \"" + inaccessiblePath + "\")");
        }
        out.println(")");
      }
    }
  }

  @Override
  protected String getName() {
    return "darwin-sandbox";
  }
}
