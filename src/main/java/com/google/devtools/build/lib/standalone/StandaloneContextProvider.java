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
import com.google.devtools.build.lib.actions.ActionContextProvider;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactMTimeCache;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;

/**
 * Provide a standalone, local execution context.
 */
public class StandaloneContextProvider implements ActionContextProvider {

  @SuppressWarnings("unchecked")
  private final ActionContext localSpawnStrategy;
  public StandaloneContextProvider() {
    localSpawnStrategy = new LocalSpawnStrategy();
  }

  @Override
  public Iterable<ActionContext> getActionContexts() {
    return ImmutableList.<ActionContext>of(localSpawnStrategy);
  }

  @Override
  public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {
  }

  @Override
  public void executionPhaseStarting(
      ActionInputFileCache actionInputFileCache,
      ArtifactMTimeCache artifactMTimeCache,
      ActionGraph actionGraph,
      Iterable<Artifact> topLevelArtifacts) throws ExecutorInitException {
  }

  @Override
  public void executionPhaseEnding()  {}
}
