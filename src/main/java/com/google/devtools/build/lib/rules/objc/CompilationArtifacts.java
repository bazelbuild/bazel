// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;

/**
 * Artifacts related to compilation. Any rule containing compilable sources will create an instance
 * of this class.
 */
final class CompilationArtifacts {
  static class Builder {
    private Iterable<Artifact> srcs = ImmutableList.of();
    private Iterable<Artifact> nonArcSrcs = ImmutableList.of();
    private Optional<Artifact> pchFile;
    private IntermediateArtifacts intermediateArtifacts;

    Builder addSrcs(Iterable<Artifact> srcs) {
      this.srcs = Iterables.concat(this.srcs, srcs);
      return this;
    }

    Builder addNonArcSrcs(Iterable<Artifact> nonArcSrcs) {
      this.nonArcSrcs = Iterables.concat(this.nonArcSrcs, nonArcSrcs);
      return this;
    }

    Builder setPchFile(Optional<Artifact> pchFile) {
      Preconditions.checkState(this.pchFile == null,
          "pchFile is already set to: %s", this.pchFile);
      this.pchFile = Preconditions.checkNotNull(pchFile);
      return this;
    }

    Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      Preconditions.checkState(this.intermediateArtifacts == null,
          "intermediateArtifacts is already set to: %s", this.intermediateArtifacts);
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    CompilationArtifacts build() {
      Optional<Artifact> archive = Optional.absent();
      if (!Iterables.isEmpty(srcs) || !Iterables.isEmpty(nonArcSrcs)) {
        archive = Optional.of(intermediateArtifacts.archive());
      }
      return new CompilationArtifacts(srcs, nonArcSrcs, archive, pchFile);
    }
  }

  private final Iterable<Artifact> srcs;
  private final Iterable<Artifact> nonArcSrcs;
  private final Optional<Artifact> archive;
  private final Optional<Artifact> pchFile;

  private CompilationArtifacts(
      Iterable<Artifact> srcs,
      Iterable<Artifact> nonArcSrcs,
      Optional<Artifact> archive,
      Optional<Artifact> pchFile) {
    this.srcs = Preconditions.checkNotNull(srcs);
    this.nonArcSrcs = Preconditions.checkNotNull(nonArcSrcs);
    this.archive = Preconditions.checkNotNull(archive);
    this.pchFile = Preconditions.checkNotNull(pchFile);
  }

  public Iterable<Artifact> getSrcs() {
    return srcs;
  }

  public Iterable<Artifact> getNonArcSrcs() {
    return nonArcSrcs;
  }

  public Optional<Artifact> getArchive() {
    return archive;
  }

  public Optional<Artifact> getPchFile() {
    return pchFile;
  }
}
