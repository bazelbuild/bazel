// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ProcessWrapperUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/** Strategy that uses sandboxing to execute a process. */
final class ProcessWrapperSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    return OS.isPosixCompatible() && ProcessWrapperUtil.isSupported(cmdEnv);
  }

  private final Path execRoot;
  private final String productName;
  private final Path processWrapper;
  private final LocalEnvProvider localEnvProvider;
  private final int timeoutGraceSeconds;

  ProcessWrapperSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      String productName,
      int timeoutGraceSeconds) {
    super(cmdEnv, sandboxBase);
    this.execRoot = cmdEnv.getExecRoot();
    this.productName = productName;
    this.timeoutGraceSeconds = timeoutGraceSeconds;
    this.processWrapper = ProcessWrapperUtil.getProcessWrapper(cmdEnv);
    this.localEnvProvider =
        OS.getCurrent() == OS.DARWIN
            ? new XCodeLocalEnvProvider()
            : LocalEnvProvider.ADD_TEMP_POSIX;
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

    Duration timeout = policy.getTimeout();
    List<String> arguments =
        ProcessWrapperUtil.getCommandLine(
            processWrapper, spawn.getArguments(), timeout, timeoutGraceSeconds);
    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, tmpDir, productName);

    SandboxedSpawn sandbox =
        new SymlinkedSandboxedSpawn(
            sandboxPath,
            sandboxExecRoot,
            arguments,
            environment,
            SandboxHelpers.getInputFiles(spawn, policy, execRoot),
            SandboxHelpers.getOutputFiles(spawn),
            getWritableDirs(sandboxExecRoot, spawn.getEnvironment(), tmpDir));

    return runSpawn(spawn, sandbox, policy, execRoot, tmpDir, timeout);
  }

  @Override
  protected String getName() {
    return "processwrapper-sandbox";
  }
}
