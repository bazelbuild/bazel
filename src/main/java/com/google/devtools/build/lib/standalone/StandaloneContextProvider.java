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
package com.google.devtools.build.lib.standalone;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.FileWriteStrategy;
import com.google.devtools.build.lib.rules.cpp.IncludeScanningContext;
import com.google.devtools.build.lib.rules.cpp.LocalGccStrategy;
import com.google.devtools.build.lib.rules.cpp.LocalLinkStrategy;
import com.google.devtools.build.lib.rules.test.ExclusiveTestStrategy;
import com.google.devtools.build.lib.rules.test.StandaloneTestStrategy;
import com.google.devtools.build.lib.rules.test.TestActionContext;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;

import java.io.IOException;

/**
 * Provide a standalone, local execution context.
 */
public class StandaloneContextProvider implements ActionContextProvider {

  /**
   * a IncludeScanningContext that does nothing. Since local execution does not need to
   * discover inclusion in advance, we do not need include scanning.
   */
  @ExecutionStrategy(contextType = IncludeScanningContext.class)
  class DummyIncludeScanningContext implements IncludeScanningContext {
    @Override
    public void extractIncludes(ActionExecutionContext actionExecutionContext,
        ActionMetadata resourceOwner, Artifact primaryInput, Artifact primaryOutput)
        throws IOException, InterruptedException {
      FileSystemUtils.writeContent(primaryOutput.getPath(), new byte[]{});
    }

    @Override
    public ArtifactResolver getArtifactResolver() {
      return runtime.getView().getArtifactFactory();
    }
  }

  @SuppressWarnings("unchecked")
  private final ActionContext standaloneSpawnStrategy;
  private final ImmutableList<ActionContext> strategies;
  private final BlazeRuntime runtime;

  public StandaloneContextProvider(BlazeRuntime runtime, BuildRequest buildRequest) {
    boolean verboseFailures = buildRequest.getOptions(ExecutionOptions.class).verboseFailures;

    standaloneSpawnStrategy = new StandaloneSpawnStrategy(runtime.getExecRoot(), verboseFailures);
    this.runtime = runtime;

    TestActionContext testStrategy = new StandaloneTestStrategy(buildRequest,
        runtime.getStartupOptionsProvider(), runtime.getBinTools(), runtime.getClientEnv(),
        runtime.getWorkspace());
    Builder<ActionContext> strategiesBuilder = ImmutableList.builder();
    // order of strategies passed to builder is significant - when there are many strategies that
    // could potentially be used and a spawnActionContext doesn't specify which one it wants, the
    // last one from strategies list will be used

    // put sandboxed strategy first, as we don't want it by default
    if (OS.getCurrent() == OS.LINUX) {
      LinuxSandboxedStrategy sandboxedLinuxStrategy =
          new LinuxSandboxedStrategy(runtime.getDirectories(), verboseFailures);
      strategiesBuilder.add(sandboxedLinuxStrategy);
    }
    strategiesBuilder.add(
        standaloneSpawnStrategy,
        new DummyIncludeScanningContext(),
        new LocalLinkStrategy(),
        testStrategy,
        new ExclusiveTestStrategy(testStrategy),
        new LocalGccStrategy(buildRequest),
        new FileWriteStrategy());


    this.strategies = strategiesBuilder.build();
  }

  @Override
  public Iterable<ActionContext> getActionContexts() {
    return strategies;
  }

  @Override
  public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {
  }

  @Override
  public void executionPhaseStarting(
      ActionInputFileCache actionInputFileCache,
      ActionGraph actionGraph,
      Iterable<Artifact> topLevelArtifacts) throws ExecutorInitException {
  }

  @Override
  public void executionPhaseEnding()  {}
}


