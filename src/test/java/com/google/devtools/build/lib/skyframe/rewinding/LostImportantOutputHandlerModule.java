// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.stream;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactExpander.MissingExpansionException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Installs an {@link ImportantOutputHandler} that allows customizing lost outputs for rewinding
 * tests.
 */
final class LostImportantOutputHandlerModule extends BlazeModule {

  private final Set<String> pathsToConsiderLost = Sets.newConcurrentHashSet();
  private final Function<byte[], String> digestFn;
  private PathFragment execRoot;

  LostImportantOutputHandlerModule(Function<byte[], String> digestFn) {
    this.digestFn = checkNotNull(digestFn);
  }

  void addLostOutput(String execPath) {
    pathsToConsiderLost.add(execPath);
  }

  void verifyAllLostOutputsConsumed() {
    assertThat(pathsToConsiderLost).isEmpty();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    env.getEventBus().register(this);
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    execRoot = env.getExecRoot().asFragment();
    registryBuilder.register(ImportantOutputHandler.class, this::getLostOutputs);
  }

  private ImmutableMap<String, ActionInput> getLostOutputs(
      Iterable<Artifact> outputs,
      ArtifactExpander expander,
      InputMetadataProvider metadataProvider) {
    ImmutableMap.Builder<String, ActionInput> lost = ImmutableMap.builder();
    for (ActionInput output : expand(outputs, expander)) {
      PathFragment execPath = output.getExecPath();
      if (execPath.isAbsolute()) {
        execPath = execPath.relativeTo(execRoot);
      }
      if (!pathsToConsiderLost.remove(execPath.getPathString())) {
        continue;
      }
      FileArtifactValue metadata;
      try {
        metadata = metadataProvider.getInputMetadata(output);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      lost.put(digestFn.apply(metadata.getDigest()), output);
    }
    return lost.buildKeepingLast();
  }

  private ImmutableList<ActionInput> expand(Iterable<Artifact> outputs, ArtifactExpander expander) {
    return stream(outputs)
        .flatMap(artifact -> expand(artifact, expander))
        .collect(toImmutableList());
  }

  private Stream<? extends ActionInput> expand(Artifact output, ArtifactExpander expander) {
    if (output.isTreeArtifact()) {
      var children = expander.expandTreeArtifact(output).stream();
      var archivedTreeArtifact = expander.getArchivedTreeArtifact(output);
      return archivedTreeArtifact == null
          ? children
          : Stream.concat(children, Stream.of(archivedTreeArtifact));
    }
    if (output.isFileset()) {
      ImmutableList<FilesetOutputSymlink> links;
      try {
        links = expander.expandFileset(output);
      } catch (MissingExpansionException e) {
        throw new IllegalStateException(e);
      }
      return links.stream()
          .filter(FilesetOutputSymlink::isRelativeToExecRoot)
          .map(link -> ActionInputHelper.fromPath(link.reconstituteTargetPath(execRoot)));
    }
    return Stream.of(output);
  }
}
