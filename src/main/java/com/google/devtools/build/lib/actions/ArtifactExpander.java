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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import javax.annotation.Nullable;

/** Expands tree artifacts and filesets. */
public interface ArtifactExpander {

  /**
   * Returns the expansion of the given {@linkplain SpecialArtifactType#TREE tree artifact}.
   *
   * @throws MissingExpansionException if this expander does not have data for the given tree
   *     artifact
   */
  ImmutableSortedSet<TreeFileArtifact> expandTreeArtifact(Artifact treeArtifact)
      throws MissingExpansionException;

  /**
   * Returns the expansion of the given {@linkplain SpecialArtifactType#TREE tree artifact}.
   *
   * <p>If this expander does not have data for the given tree artifact, returns an empty set.
   */
  default ImmutableSortedSet<TreeFileArtifact> tryExpandTreeArtifact(Artifact treeArtifact) {
    try {
      return expandTreeArtifact(treeArtifact);
    } catch (MissingExpansionException e) {
      return ImmutableSortedSet.of();
    }
  }

  /**
   * Returns the expansion of the given {@linkplain SpecialArtifactType#FILESET fileset artifact}.
   *
   * @throws MissingExpansionException if the expander is missing data needed to expand provided
   *     fileset.
   */
  default FilesetOutputTree expandFileset(Artifact fileset) throws MissingExpansionException {
    throw new MissingExpansionException("Cannot expand fileset " + fileset);
  }

  /**
   * Returns an {@link ArchivedTreeArtifact} for a provided {@linkplain SpecialArtifactType#TREE
   * tree artifact} if one is available.
   *
   * <p>The {@linkplain ArchivedTreeArtifact archived tree artifact} can be used instead of the tree
   * artifact expansion.
   */
  @Nullable
  default ArchivedTreeArtifact getArchivedTreeArtifact(Artifact treeArtifact) {
    return null;
  }

  /**
   * Exception thrown when attempting to {@linkplain ArtifactExpander expand} an artifact for which
   * we do not have the necessary data.
   */
  final class MissingExpansionException extends Exception {
    public MissingExpansionException(String message) {
      super(message);
    }
  }
}
