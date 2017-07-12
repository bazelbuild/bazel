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
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.apple.XCodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Strategy that uses sandboxing to execute a process. */
final class ProcessWrapperSandboxedSpawnRunner extends AbstractSandboxSpawnRunner {

  public static boolean isSupported(CommandEnvironment cmdEnv) {
    return OS.isPosixCompatible() && ProcessWrapperRunner.isSupported(cmdEnv);
  }

  private final Path execRoot;
  private final String productName;
  private final Path processWrapper;
  private final LocalEnvProvider localEnvProvider;

  ProcessWrapperSandboxedSpawnRunner(
      CommandEnvironment cmdEnv,
      BuildRequest buildRequest,
      Path sandboxBase,
      String productName) {
    super(
        cmdEnv,
        sandboxBase,
        buildRequest.getOptions(SandboxOptions.class));
    this.execRoot = cmdEnv.getExecRoot();
    this.productName = productName;
    this.processWrapper = ProcessWrapperRunner.getProcessWrapper(cmdEnv);
    this.localEnvProvider = OS.getCurrent() == OS.DARWIN
        ? new XCodeLocalEnvProvider()
        : LocalEnvProvider.UNMODIFIED;
  }

  @Override
  protected SpawnResult actuallyExec(Spawn spawn, SpawnExecutionPolicy policy)
      throws ExecException, IOException, InterruptedException {
    // Each invocation of "exec" gets its own sandbox.
    Path sandboxPath = getSandboxRoot();
    Path sandboxExecRoot = sandboxPath.getRelative("execroot").getRelative(execRoot.getBaseName());

    int timeoutSeconds = (int) TimeUnit.MILLISECONDS.toSeconds(policy.getTimeoutMillis());
    List<String> arguments =
        ProcessWrapperRunner.getCommandLine(processWrapper, spawn.getArguments(), timeoutSeconds);
    Map<String, String> environment =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), execRoot, productName);

    SandboxedSpawn sandbox = new SymlinkedSandboxedSpawn(
        sandboxPath,
        sandboxExecRoot,
        arguments,
        environment,
        SandboxHelpers.getInputFiles(spawn, policy, execRoot),
        SandboxHelpers.getOutputFiles(spawn),
        getWritableDirs(sandboxExecRoot, spawn.getEnvironment()));
    return runSpawn(spawn, sandbox, policy, execRoot, timeoutSeconds);
  }

  @Override
  protected String getName() {
    return "processwrapper-sandbox";
  }
}
