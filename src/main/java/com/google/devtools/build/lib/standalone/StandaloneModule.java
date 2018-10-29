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
package com.google.devtools.build.lib.standalone;

import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.actions.LocalTemplateExpansionStrategy;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.FileWriteStrategy;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.StandaloneTestStrategy;
import com.google.devtools.build.lib.exec.TestStrategy;
import com.google.devtools.build.lib.exec.apple.XcodeLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.exec.local.LocalExecutionOptions;
import com.google.devtools.build.lib.exec.local.LocalSpawnRunner;
import com.google.devtools.build.lib.exec.local.PosixLocalEnvProvider;
import com.google.devtools.build.lib.exec.local.WindowsLocalEnvProvider;
import com.google.devtools.build.lib.rules.cpp.SpawnGccStrategy;
import com.google.devtools.build.lib.rules.test.ExclusiveTestStrategy;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;

/**
 * StandaloneModule provides pluggable functionality for blaze.
 */
public class StandaloneModule extends BlazeModule {
  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    // TODO(ulfjack): Move this to another module.
    builder.addActionContextProvider(new DummyIncludeScanningContextProvider(env));

    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    Path testTmpRoot =
        TestStrategy.getTmpRoot(env.getWorkspace(), env.getExecRoot(), executionOptions);
    TestActionContext testStrategy =
        new StandaloneTestStrategy(
            executionOptions,
            env.getBlazeWorkspace().getBinTools(),
            testTmpRoot);

    // Order of strategies passed to builder is significant - when there are many strategies that
    // could potentially be used and a spawnActionContext doesn't specify which one it wants, the
    // last one from strategies list will be used
    builder.addActionContext(
        new StandaloneSpawnStrategy(env.getExecRoot(), createLocalRunner(env)));
    builder.addActionContext(new SpawnGccStrategy());
    builder.addActionContext(testStrategy);
    builder.addActionContext(new ExclusiveTestStrategy(testStrategy));
    builder.addActionContext(new FileWriteStrategy());
    builder.addActionContext(new LocalTemplateExpansionStrategy());

    // This makes the "sandboxed" strategy the default Spawn strategy, unless it is overridden by a
    // later BlazeModule.
    builder.addStrategyByMnemonic("", "standalone");

    // This makes the "standalone" strategy available via --spawn_strategy=standalone, but it is not
    // necessarily the default.
    builder.addStrategyByContext(SpawnActionContext.class, "standalone");
  }

  private static SpawnRunner createLocalRunner(CommandEnvironment env) {
    LocalExecutionOptions localExecutionOptions =
        env.getOptions().getOptions(LocalExecutionOptions.class);
    LocalEnvProvider localEnvProvider =
        OS.getCurrent() == OS.DARWIN
            ? new XcodeLocalEnvProvider(env.getClientEnv())
            : (OS.getCurrent() == OS.WINDOWS
                ? new WindowsLocalEnvProvider(env.getClientEnv())
                : new PosixLocalEnvProvider(env.getClientEnv()));
    return
        new LocalSpawnRunner(
            env.getExecRoot(),
            localExecutionOptions,
            ResourceManager.instance(),
            localEnvProvider,
            env.getBlazeWorkspace().getBinTools());
  }
}
