// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.SortedMap;

/** Execution context for tests */
public class FakeSpawnExecutionContext implements SpawnExecutionContext {

  private boolean lockOutputFilesCalled;

  private void artifactExpander(Artifact artifact, Collection<? super Artifact> output) {
    output.add(artifact);
  }

  private final Spawn spawn;
  private final MetadataProvider metadataProvider;
  private final Path execRoot;
  private final FileOutErr outErr;
  private final ClassToInstanceMap<ActionContext> actionContextRegistry;

  public FakeSpawnExecutionContext(
      Spawn spawn, MetadataProvider metadataProvider, Path execRoot, FileOutErr outErr) {
    this(spawn, metadataProvider, execRoot, outErr, ImmutableClassToInstanceMap.of());
  }

  public FakeSpawnExecutionContext(
      Spawn spawn,
      MetadataProvider metadataProvider,
      Path execRoot,
      FileOutErr outErr,
      ClassToInstanceMap<ActionContext> actionContextRegistry) {
    this.spawn = spawn;
    this.metadataProvider = metadataProvider;
    this.execRoot = execRoot;
    this.outErr = outErr;
    this.actionContextRegistry = actionContextRegistry;
  }

  public boolean isLockOutputFilesCalled() {
    return lockOutputFilesCalled;
  }

  @Override
  public int getId() {
    return 0;
  }

  @Override
  public void prefetchInputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void lockOutputFiles() {
    lockOutputFilesCalled = true;
  }

  @Override
  public boolean speculating() {
    return false;
  }

  @Override
  public MetadataProvider getMetadataProvider() {
    return metadataProvider;
  }

  @Override
  public ArtifactExpander getArtifactExpander() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Duration getTimeout() {
    return Duration.ZERO;
  }

  @Override
  public FileOutErr getFileOutErr() {
    return outErr;
  }

  @Override
  public SortedMap<PathFragment, ActionInput> getInputMapping(boolean expandTreeArtifactsInRunfiles)
      throws IOException {
    return new SpawnInputExpander(execRoot, /*strict*/ false)
        .getInputMapping(spawn, this::artifactExpander, metadataProvider, true);
  }

  @Override
  public void report(ProgressStatus state, String name) {
    // Intentionally left empty.
  }

  @Override
  public MetadataInjector getMetadataInjector() {
    return ActionsTestUtil.THROWING_METADATA_HANDLER;
  }

  @Override
  public <T extends ActionContext> T getContext(Class<T> identifyingType) {
    return actionContextRegistry.getInstance(identifyingType);
  }

  @Override
  public boolean isRewindingEnabled() {
    return false;
  }

  @Override
  public void checkForLostInputs() {}
}
