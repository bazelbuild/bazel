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
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.RemoteActionFileSystem;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** Execution context for tests */
public class FakeSpawnExecutionContext implements SpawnExecutionContext {

  private boolean lockOutputFilesCalled;

  private void artifactExpander(Artifact artifact, Collection<? super Artifact> output) {
    output.add(artifact);
  }

  private final Spawn spawn;
  private final InputMetadataProvider inputMetadataProvider;
  private final Path execRoot;
  private final FileOutErr outErr;
  private final ClassToInstanceMap<ActionContext> actionContextRegistry;
  @Nullable private final RemoteActionFileSystem actionFileSystem;

  public FakeSpawnExecutionContext(
      Spawn spawn, InputMetadataProvider inputMetadataProvider, Path execRoot, FileOutErr outErr) {
    this(spawn, inputMetadataProvider, execRoot, outErr, ImmutableClassToInstanceMap.of(), null);
  }

  public FakeSpawnExecutionContext(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      Path execRoot,
      FileOutErr outErr,
      ClassToInstanceMap<ActionContext> actionContextRegistry) {
    this(spawn, inputMetadataProvider, execRoot, outErr, actionContextRegistry, null);
  }

  public FakeSpawnExecutionContext(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      Path execRoot,
      FileOutErr outErr,
      ClassToInstanceMap<ActionContext> actionContextRegistry,
      @Nullable RemoteActionFileSystem actionFileSystem) {
    this.spawn = spawn;
    this.inputMetadataProvider = inputMetadataProvider;
    this.execRoot = execRoot;
    this.outErr = outErr;
    this.actionContextRegistry = actionContextRegistry;
    this.actionFileSystem = actionFileSystem;
  }

  public boolean isLockOutputFilesCalled() {
    return lockOutputFilesCalled;
  }

  @Override
  public int getId() {
    return 0;
  }

  @Override
  public ListenableFuture<Void> prefetchInputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void lockOutputFiles(int exitCode, String errorMessage, FileOutErr outErr) {
    lockOutputFilesCalled = true;
  }

  @Override
  public boolean speculating() {
    return false;
  }

  @Override
  public InputMetadataProvider getInputMetadataProvider() {
    return inputMetadataProvider;
  }

  @Override
  public ArtifactExpander getArtifactExpander() {
    return this::artifactExpander;
  }

  @Override
  public SpawnInputExpander getSpawnInputExpander() {
    return new SpawnInputExpander(execRoot, /* strict= */ false);
  }

  @Override
  public ArtifactPathResolver getPathResolver() {
    return ArtifactPathResolver.forExecRoot(execRoot);
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
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      PathFragment baseDirectory, boolean willAccessRepeatedly)
      throws IOException, ForbiddenActionInputException {
    return getSpawnInputExpander()
        .getInputMapping(spawn, this::artifactExpander, baseDirectory, inputMetadataProvider);
  }

  @Override
  public void report(ProgressStatus progress) {
    // Intentionally left empty.
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

  @Nullable
  @Override
  public RemoteActionFileSystem getActionFileSystem() {
    return actionFileSystem;
  }
}
