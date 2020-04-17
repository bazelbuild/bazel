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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Morally a {@link SetMultimap} from artifacts to the labels that own them, which may not be just
 * {@link Artifact#getOwnerLabel} depending on the build. Optimizes for artifacts that are owned by
 * their {@link Artifact#getOwnerLabel} by storing them separately.
 */
public class ArtifactsToOwnerLabels {
  private final SetMultimap<Artifact, Label> artifactToMultipleOrDifferentOwnerLabels;
  private final Set<Artifact> artifactsOwnedOnlyByTheirLabels;

  private ArtifactsToOwnerLabels(
      SetMultimap<Artifact, Label> artifactToMultipleOrDifferentOwnerLabels,
      Set<Artifact> artifactsOwnedOnlyByTheirLabels) {
    this.artifactToMultipleOrDifferentOwnerLabels = artifactToMultipleOrDifferentOwnerLabels;
    this.artifactsOwnedOnlyByTheirLabels = artifactsOwnedOnlyByTheirLabels;
  }

  public Set<Label> getOwners(Artifact artifact) {
    if (artifactsOwnedOnlyByTheirLabels.contains(artifact)) {
      Preconditions.checkState(
          !artifactToMultipleOrDifferentOwnerLabels.containsKey(artifact),
          "Artifact %s incorrectly in multiple categories",
          artifact);
      Label ownerLabel = artifact.getOwnerLabel();
      if (ownerLabel == null) {
        return ImmutableSet.of();
      }
      return ImmutableSet.of(ownerLabel);
    }
    return Preconditions.checkNotNull(
        artifactToMultipleOrDifferentOwnerLabels.get(artifact), artifact);
  }

  public Set<Artifact> getArtifacts() {
    return Sets.union(
        artifactsOwnedOnlyByTheirLabels, artifactToMultipleOrDifferentOwnerLabels.keySet());
  }

  public Builder toBuilder() {
    return new Builder(
        HashMultimap.create(artifactToMultipleOrDifferentOwnerLabels),
        new HashSet<>(artifactsOwnedOnlyByTheirLabels));
  }

  /** Builder for {@link ArtifactsToOwnerLabels}. */
  public static class Builder {
    private final SetMultimap<Artifact, Label> artifactToMultipleOrDifferentOwnerLabels;
    private final Set<Artifact> artifactsOwnedOnlyByTheirLabels;

    public Builder() {
      this(HashMultimap.create(), new HashSet<>());
    }

    private Builder(
        SetMultimap<Artifact, Label> artifactToMultipleOrDifferentOwnerLabels,
        Set<Artifact> artifactsOwnedOnlyByTheirLabels) {
      this.artifactToMultipleOrDifferentOwnerLabels = artifactToMultipleOrDifferentOwnerLabels;
      this.artifactsOwnedOnlyByTheirLabels = artifactsOwnedOnlyByTheirLabels;
    }

    public Builder addArtifact(Artifact artifact) {
      if (artifactToMultipleOrDifferentOwnerLabels.containsKey(artifact)) {
        Label ownerLabel = artifact.getOwnerLabel();
        if (ownerLabel != null) {
          artifactToMultipleOrDifferentOwnerLabels.put(artifact, artifact.getOwnerLabel());
        }
      } else {
        artifactsOwnedOnlyByTheirLabels.add(artifact);
      }
      return this;
    }

    public Builder addArtifact(Artifact artifact, Label label) {
      Preconditions.checkNotNull(label, artifact);
      if (label.equals(artifact.getOwnerLabel())) {
        addArtifact(artifact);
      } else {
        artifactToMultipleOrDifferentOwnerLabels.put(artifact, label);
        if (artifactsOwnedOnlyByTheirLabels.remove(artifact)) {
          // Redoing this call now that we have a mismatched label will force addition into the
          // multimap.
          addArtifact(artifact);
        }
      }
      return this;
    }

    public Builder filterArtifacts(Predicate<Artifact> artifactsToKeep) {
      Predicate<Artifact> artifactsToRemove = artifactsToKeep.negate();
      artifactToMultipleOrDifferentOwnerLabels.keySet().removeIf(artifactsToRemove);
      artifactsOwnedOnlyByTheirLabels.removeIf(artifactsToRemove);
      return this;
    }

    ArtifactsToOwnerLabels build() {
      return new ArtifactsToOwnerLabels(
          artifactToMultipleOrDifferentOwnerLabels, artifactsOwnedOnlyByTheirLabels);
    }
  }
}
