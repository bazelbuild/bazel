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
package com.google.devtools.build.lib.actions;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpanderImpl;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * {@link CompletionContext} contains an {@link ArtifactExpander} and {@link ArtifactPathResolver}
 * used to resolve output files during a {@link
 * com.google.devtools.build.lib.skyframe.CompletionFunction} evaluation.
 */
@AutoValue
public abstract class CompletionContext {

  public static final CompletionContext FAILED_COMPLETION_CTX = createNull();

  public abstract ArtifactExpander expander();

  public abstract ArtifactPathResolver pathResolver();

  public static CompletionContext create(
      Map<Artifact, Collection<Artifact>> expandedArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      ActionInputMap inputMap,
      PathResolverFactory pathResolverFactory,
      String workspaceName) {
    ArtifactExpander expander = new ArtifactExpanderImpl(expandedArtifacts, expandedFilesets);
    ArtifactPathResolver pathResolver =
        pathResolverFactory.shouldCreatePathResolverForArtifactValues()
            ? pathResolverFactory.createPathResolverForArtifactValues(
                inputMap, expandedArtifacts, expandedFilesets.keySet(), workspaceName)
            : ArtifactPathResolver.IDENTITY;
    return new AutoValue_CompletionContext(expander, pathResolver);
  }

  private static CompletionContext createNull() {
    return new AutoValue_CompletionContext((artifact, output) -> {}, ArtifactPathResolver.IDENTITY);
  }

  public void visitArtifacts(Iterable<Artifact> artifacts, ArtifactReceiver receiver) {
    for (Artifact artifact : artifacts) {
      if (artifact.isMiddlemanArtifact() || artifact.isFileset()) {
        // We never want to report middleman artifacts. They are for internal use only.
        // Filesets are not currently supported, but should be in the future.
        continue;
      } else if (artifact.isTreeArtifact()) {
        List<Artifact> expandedArtifacts = new ArrayList<>();
        expander().expand(artifact, expandedArtifacts);
        for (Artifact expandedArtifact : expandedArtifacts) {
          receiver.accept(expandedArtifact);
        }
      } else {
        receiver.accept(artifact);
      }
    }
  }

  /** A function that accepts an {@link Artifact}. */
  @FunctionalInterface
  public interface ArtifactReceiver {
    void accept(Artifact a);
  }

  /** A factory for {@link ArtifactPathResolver}. */
  public interface PathResolverFactory {
    ArtifactPathResolver createPathResolverForArtifactValues(
        ActionInputMap actionInputMap,
        Map<Artifact, Collection<Artifact>> expandedArtifacts,
        Iterable<Artifact> filesets,
        String workspaceName);

    boolean shouldCreatePathResolverForArtifactValues();
  }
}
