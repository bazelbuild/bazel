// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import javax.annotation.Nullable;

/**
 * Implementation of {@link ArtifactExpander} that reads the expansions from an {@link
 * InputMetadataProvider}.
 */
public record InputMetadataProviderArtifactExpander(InputMetadataProvider inputMetadataProvider)
    implements ArtifactExpander {
  @Override
  public ImmutableSortedSet<TreeFileArtifact> expandTreeArtifact(Artifact treeArtifact)
      throws MissingExpansionException {
    checkArgument(treeArtifact.isTreeArtifact(), treeArtifact);
    TreeArtifactValue tree = inputMetadataProvider.getTreeMetadata(treeArtifact);
    if (tree == null) {
      throw new MissingExpansionException("Missing expansion for tree artifact: " + treeArtifact);
    }
    return tree.getChildren();
  }

  @Override
  public FilesetOutputTree expandFileset(Artifact fileset) throws MissingExpansionException {
    checkArgument(fileset.isFileset(), fileset);
    FilesetOutputTree filesetOutput = inputMetadataProvider.getFileset(fileset);
    if (filesetOutput == null) {
      throw new MissingExpansionException("Missing expansion for fileset: " + fileset);
    }
    return filesetOutput;
  }

  @Override
  @Nullable
  public ArchivedTreeArtifact getArchivedTreeArtifact(Artifact treeArtifact) {
    checkArgument(treeArtifact.isTreeArtifact(), treeArtifact);
    TreeArtifactValue tree = inputMetadataProvider.getTreeMetadata(treeArtifact);
    return tree == null ? null : tree.getArchivedArtifact();
  }

  /**
   * Wraps an {@link InputMetadataProvider} into an {@link ArtifactExpander}.
   *
   * <p>Returns null if the argument is null.
   */
  @Nullable
  public static ArtifactExpander maybeFrom(@Nullable InputMetadataProvider inputMetadataProvider) {
    return inputMetadataProvider == null
        ? null
        : new InputMetadataProviderArtifactExpander(inputMetadataProvider);
  }
}
