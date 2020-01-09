// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * Artifacts related to compilation. Any rule containing compilable sources will create an instance
 * of this class.
 */
final class CompilationArtifacts {
  static class Builder {
    // TODO(bazel-team): Should these be sets instead of just iterables?
    private Iterable<Artifact> srcs = ImmutableList.of();
    private Iterable<Artifact> nonArcSrcs = ImmutableList.of();
    private Iterable<Artifact> additionalHdrs = ImmutableList.of();
    private Iterable<Artifact> privateHdrs = ImmutableList.of();
    private Iterable<Artifact> precompiledSrcs = ImmutableList.of();
    private IntermediateArtifacts intermediateArtifacts;

    Builder addSrcs(Iterable<Artifact> srcs) {
      this.srcs = Iterables.concat(this.srcs, srcs);
      return this;
    }

    Builder addNonArcSrcs(Iterable<Artifact> nonArcSrcs) {
      this.nonArcSrcs = Iterables.concat(this.nonArcSrcs, nonArcSrcs);
      return this;
    }

    /**
     * Adds header artifacts that should be directly accessible to dependers, but aren't specified
     * in the hdrs attribute. {@code additionalHdrs} should not be a {@link NestedSet}, as it will
     * be flattened when added.
     */
    Builder addAdditionalHdrs(Iterable<Artifact> additionalHdrs) {
      this.additionalHdrs = Iterables.concat(this.additionalHdrs, additionalHdrs);
      return this;
    }

    /**
     * Adds header artifacts that should not be directly accessible to dependers.
     * {@code privateHdrs} should not be a {@link NestedSet}, as it will be flattened when added.
     */
    Builder addPrivateHdrs(Iterable<Artifact> privateHdrs) {
      this.privateHdrs = Iterables.concat(this.privateHdrs, privateHdrs);
      return this;
    }

    /**
     * Adds precompiled sources (.o files).
     */
    Builder addPrecompiledSrcs(Iterable<Artifact> precompiledSrcs) {
      // TODO(ulfjack): These are ignored *except* for a check below whether they are empty.
      this.precompiledSrcs = Iterables.concat(this.precompiledSrcs, precompiledSrcs);
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
      if (!Iterables.isEmpty(srcs)
          || !Iterables.isEmpty(nonArcSrcs)
          || !Iterables.isEmpty(precompiledSrcs)) {
        archive = Optional.of(intermediateArtifacts.archive());
      }
      return new CompilationArtifacts(srcs, nonArcSrcs, additionalHdrs, privateHdrs, archive);
    }
  }

  private final Iterable<Artifact> srcs;
  private final Iterable<Artifact> nonArcSrcs;
  private final Optional<Artifact> archive;
  private final Iterable<Artifact> additionalHdrs;
  private final Iterable<Artifact> privateHdrs;

  private CompilationArtifacts(
      Iterable<Artifact> srcs,
      Iterable<Artifact> nonArcSrcs,
      Iterable<Artifact> additionalHdrs,
      Iterable<Artifact> privateHdrs,
      Optional<Artifact> archive) {
    this.srcs = Preconditions.checkNotNull(srcs);
    this.nonArcSrcs = Preconditions.checkNotNull(nonArcSrcs);
    this.additionalHdrs = Preconditions.checkNotNull(additionalHdrs);
    this.privateHdrs = Preconditions.checkNotNull(privateHdrs);
    this.archive = Preconditions.checkNotNull(archive);
  }

  Iterable<Artifact> getSrcs() {
    return srcs;
  }

  Iterable<Artifact> getNonArcSrcs() {
    return nonArcSrcs;
  }

  /** Returns the public headers that aren't included in the hdrs attribute. */
  Iterable<Artifact> getAdditionalHdrs() {
    return additionalHdrs;
  }

  /**
   * Returns the private headers from the srcs attribute, which may by imported by any source or
   * header in this target, but not by sources or headers of dependers.
   */
  Iterable<Artifact> getPrivateHdrs() {
    return privateHdrs;
  }

  /**
   * Returns the output archive library (.a) file created by combining object files of the srcs, non
   * arc srcs, and precompiled srcs of this artifact collection. Returns absent if there are no such
   * source files for which to create an archive library.
   */
  Optional<Artifact> getArchive() {
    return archive;
  }
}
