// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.Set;
import java.util.SortedMap;

class FakeSpawnExecutionContext implements SpawnExecutionContext {
  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER =
      new ArtifactExpander() {
        @Override
        public void expand(Artifact artifact, Collection<? super Artifact> output) {
          output.add(artifact);
        }
      };

  private final Spawn spawn;
  private final MetadataProvider fileCache;
  private final FileOutErr outErr;
  private final Path execRoot;

  FakeSpawnExecutionContext(
      Spawn spawn,
      MetadataProvider fileCache,
      FileOutErr outErr,
      Path execRoot) {
    this.spawn = spawn;
    this.fileCache = fileCache;
    this.outErr = outErr;
    this.execRoot = execRoot;
  }

  @Override
  public int getId() {
    return 0;
  }

  @Override
  public void prefetchInputs() throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void lockOutputFiles() throws InterruptedException {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean speculating() {
    return false;
  }

  @Override
  public MetadataProvider getMetadataProvider() {
    return fileCache;
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
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      boolean expandTreeArtifactsInRunfiles) throws IOException {
    return new SpawnInputExpander(execRoot, /*strict*/ false)
        .getInputMapping(spawn, SIMPLE_ARTIFACT_EXPANDER, ArtifactPathResolver.IDENTITY,
            fileCache, true);
  }

  @Override
  public void report(ProgressStatus state, String name) {
  }


  private Set<String> getRequiredOutputs() {
    return spawn.getOutputFiles()
        .stream()
        .map((inp) -> inp.getExecPath().getPathString())
        .collect(ImmutableSet.toImmutableSet());
  }

  @Override
  public boolean areOutputsValid(Path root) {
    return Iterables.all(
        getRequiredOutputs(),
        (path) -> root.getRelative(path).isFile());
  }
}
