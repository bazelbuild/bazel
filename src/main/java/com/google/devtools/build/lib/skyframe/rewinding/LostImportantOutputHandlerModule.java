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

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ImportantOutputHandler;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.ForOverride;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Installs an {@link ImportantOutputHandler} that allows customizing lost outputs for rewinding
 * tests.
 */
public class LostImportantOutputHandlerModule extends BlazeModule {

  // This is a multiset so an output can be marked lost more than once. This is necessary to test
  // scenarios where there might be a restart in rewinding.
  private final ConcurrentHashMultiset<String> pathsToConsiderLost =
      ConcurrentHashMultiset.create();
  private final BiFunction<byte[], Long, String> digestFn;
  private boolean outputHandlerEnabled = true;

  public LostImportantOutputHandlerModule(BiFunction<byte[], Long, String> digestFn) {
    this.digestFn = checkNotNull(digestFn);
  }

  /** Controls whether an {@link ImportantOutputHandler} will be installed. */
  public final void setOutputHandlerEnabled(boolean enabled) {
    outputHandlerEnabled = enabled;
  }

  public final void addLostOutput(String execPath) {
    pathsToConsiderLost.add(execPath, 1);
  }

  public final void verifyAllLostOutputsConsumed() {
    assertThat(pathsToConsiderLost).isEmpty();
  }

  @Override
  public final void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    if (outputHandlerEnabled) {
      registryBuilder.register(ImportantOutputHandler.class, createOutputHandler(env));
    }
  }

  @ForOverride
  protected ImportantOutputHandler createOutputHandler(CommandEnvironment env) {
    return new MockImportantOutputHandler();
  }

  /**
   * Returns whether the given output should be treated as lost.
   *
   * <p>If {@code true} is returned, the given output is removed from the set of lost outputs so
   * that a subsequent call to this method with the same output will return {@code false}.
   */
  protected final boolean outputIsLost(PathFragment execPath) {
    return pathsToConsiderLost.removeExactly(execPath.getPathString(), 1);
  }

  private final class MockImportantOutputHandler implements ImportantOutputHandler {

    @Override
    public LostArtifacts processOutputsAndGetLostArtifacts(
        Iterable<Artifact> importantOutputs,
        InputMetadataProvider importantMetadataProvider,
        InputMetadataProvider fullMetadataProvider) {
      return getLostOutputs(importantOutputs, importantMetadataProvider);
    }

    @Override
    public LostArtifacts processRunfilesAndGetLostArtifacts(
        PathFragment runfilesDir,
        Map<PathFragment, Artifact> runfiles,
        InputMetadataProvider metadataProvider,
        String inputManifestExtension) {
      return getLostOutputs(runfiles.values(), metadataProvider);
    }

    @Override
    public void processTestOutputs(Collection<Path> testOutputs) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void processWorkspaceStatusOutputs(Path stableOutput, Path volatileOutput) {
      throw new UnsupportedOperationException();
    }

    private LostArtifacts getLostOutputs(
        Iterable<Artifact> outputs, InputMetadataProvider metadataProvider) {
      ImmutableMap.Builder<String, ActionInput> lost = ImmutableMap.builder();
      LostInputOwners owners = new LostInputOwners();
      for (OutputAndOwner outputAndOwner : expand(outputs, metadataProvider)) {
        ActionInput output = outputAndOwner.output;
        Artifact owner = outputAndOwner.owner;
        if (!outputIsLost(output.getExecPath())) {
          continue;
        }
        FileArtifactValue metadata;
        try {
          metadata = metadataProvider.getInputMetadata(output);
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
        lost.put(digestFn.apply(metadata.getDigest(), metadata.getSize()), output);
        if (owner != null) {
          owners.addOwner(output, owner);
        }
      }
      return new LostArtifacts(lost.buildKeepingLast(), Optional.of(owners));
    }

    private static ImmutableList<OutputAndOwner> expand(
        Iterable<Artifact> outputs, InputMetadataProvider inputMetadataProvider) {
      return stream(outputs)
          .flatMap(artifact -> expand(artifact, inputMetadataProvider))
          .collect(toImmutableList());
    }

    private static Stream<OutputAndOwner> expand(
        Artifact output, InputMetadataProvider inputMetadataProvider) {
      if (output.isTreeArtifact()) {
        TreeArtifactValue treeArtifactValue = inputMetadataProvider.getTreeMetadata(output);
        var archivedTreeArtifact = treeArtifactValue.getArchivedArtifact();
        var children = treeArtifactValue.getChildren().stream();
        var expansion =
            archivedTreeArtifact == null
                ? children
                : Stream.concat(children, Stream.of(archivedTreeArtifact));
        return expansion.map(child -> new OutputAndOwner(child, output));
      }
      if (output.isFileset()) {
        ImmutableList<FilesetOutputSymlink> links =
            inputMetadataProvider.getFileset(output).symlinks();
        return links.stream().map(link -> new OutputAndOwner(link.target(), output));
      }
      return Stream.of(new OutputAndOwner(output, null));
    }

    private record OutputAndOwner(ActionInput output, @Nullable Artifact owner) {}
  }
}
