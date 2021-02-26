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

import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.RESOLVE;
import static com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior.RESOLVE_FULLY;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpanderImpl;
import com.google.devtools.build.lib.actions.Artifact.MissingExpansionException;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehavior;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Container for the data one needs to resolve aggregate artifacts from events signaling the
 * completion of a target or an aspect ({@code TargetCompleteEvent} and {@code
 * AspectCompleteEvent}).
 *
 * <p>This is needed because some artifacts (tree artifacts and Filesets) are in fact aggregations
 * of multiple files.
 */
public class CompletionContext {
  public static final CompletionContext FAILED_COMPLETION_CTX =
      new CompletionContext(
          null, (artifact, output) -> {}, ArtifactPathResolver.IDENTITY, false, false);

  private final Path execRoot;
  private final ArtifactExpander expander;
  private final ArtifactPathResolver pathResolver;

  private final boolean expandFilesets;
  private final boolean fullyResolveFilesetLinks;

  private CompletionContext(
      Path execRoot,
      ArtifactExpander expander,
      ArtifactPathResolver pathResolver,
      boolean expandFilesets,
      boolean fullyResolveFilesetLinks) {
    this.execRoot = execRoot;
    this.expander = expander;
    this.pathResolver = pathResolver;
    this.expandFilesets = expandFilesets;
    this.fullyResolveFilesetLinks = fullyResolveFilesetLinks;
  }

  public static CompletionContext create(
      Map<Artifact, ImmutableCollection<Artifact>> expandedArtifacts,
      Map<SpecialArtifact, ArchivedTreeArtifact> archivedTreeArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> expandedFilesets,
      boolean expandFilesets,
      boolean fullyResolveFilesetSymlinks,
      ActionInputMap inputMap,
      PathResolverFactory pathResolverFactory,
      Path execRoot,
      String workspaceName)
      throws IOException {
    ArtifactExpander expander =
        new ArtifactExpanderImpl(expandedArtifacts, archivedTreeArtifacts, expandedFilesets);
    ArtifactPathResolver pathResolver =
        pathResolverFactory.shouldCreatePathResolverForArtifactValues()
            ? pathResolverFactory.createPathResolverForArtifactValues(
                inputMap, expandedArtifacts, expandedFilesets, workspaceName)
            : ArtifactPathResolver.IDENTITY;
    return new CompletionContext(
        execRoot, expander, pathResolver, expandFilesets, fullyResolveFilesetSymlinks);
  }

  public ArtifactPathResolver pathResolver() {
    return pathResolver;
  }

  public void visitArtifacts(Iterable<Artifact> artifacts, ArtifactReceiver receiver) {
    for (Artifact artifact : artifacts) {
      if (artifact.isMiddlemanArtifact()) {
        continue;
      } else if (artifact.isFileset()) {
        if (expandFilesets) {
          visitFileset(artifact, receiver, fullyResolveFilesetLinks ? RESOLVE_FULLY : RESOLVE);
        }
      } else if (artifact.isTreeArtifact()) {
        List<Artifact> expandedArtifacts = new ArrayList<>();
        expander.expand(artifact, expandedArtifacts);
        for (Artifact expandedArtifact : expandedArtifacts) {
          receiver.accept(expandedArtifact);
        }
      } else {
        receiver.accept(artifact);
      }
    }
  }

  private void visitFileset(
      Artifact filesetArtifact,
      ArtifactReceiver receiver,
      RelativeSymlinkBehavior relativeSymlinkBehavior) {
    ImmutableList<FilesetOutputSymlink> links;
    try {
      links = expander.getFileset(filesetArtifact);
    } catch (MissingExpansionException e) {
      throw new IllegalStateException(e);
    }
    FilesetManifest filesetManifest;
    try {
      filesetManifest =
          FilesetManifest.constructFilesetManifest(
              links, PathFragment.EMPTY_FRAGMENT, relativeSymlinkBehavior);
    } catch (IOException e) {
      // Unexpected: RelativeSymlinkBehavior.RESOLVE should not throw.
      throw new IllegalStateException(e);
    }

    for (Map.Entry<PathFragment, String> mapping : filesetManifest.getEntries().entrySet()) {
      String targetFile = mapping.getValue();
      PathFragment locationInFileset = mapping.getKey();
      receiver.acceptFilesetMapping(
          filesetArtifact, locationInFileset, execRoot.getRelative(targetFile));
    }
  }

  /** A function that accepts an {@link Artifact}. */
  public interface ArtifactReceiver {
    void accept(Artifact artifact);
    void acceptFilesetMapping(Artifact fileset, PathFragment relName, Path targetFile);
  }

  /** A factory for {@link ArtifactPathResolver}. */
  public interface PathResolverFactory {
    ArtifactPathResolver createPathResolverForArtifactValues(
        ActionInputMap actionInputMap,
        Map<Artifact, ImmutableCollection<Artifact>> expandedArtifacts,
        Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets,
        String workspaceName)
        throws IOException;

    boolean shouldCreatePathResolverForArtifactValues();
  }
}
