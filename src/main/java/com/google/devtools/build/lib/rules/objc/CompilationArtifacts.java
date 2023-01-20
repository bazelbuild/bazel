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

import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.COMPILABLE_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PRECOMPILED_SRCS_TYPE;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.util.FileType;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/**
 * Artifacts related to compilation. Any rule containing compilable sources will create an instance
 * of this class.
 */
final class CompilationArtifacts implements StarlarkValue {
  private final Iterable<Artifact> srcs;
  private final Iterable<Artifact> nonArcSrcs;
  private final Iterable<Artifact> additionalHdrs;
  private final Optional<Artifact> archive;

  public CompilationArtifacts() {
    this.srcs = ImmutableList.<Artifact>of();
    this.nonArcSrcs = ImmutableList.<Artifact>of();
    this.additionalHdrs = ImmutableList.<Artifact>of();
    this.archive = Optional.absent();
  }

  public CompilationArtifacts(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts) {
    this(
        ruleContext.getPrerequisiteArtifacts("srcs").list(),
        ruleContext.getPrerequisiteArtifacts("non_arc_srcs").list(),
        ImmutableList.<Artifact>of(),
        intermediateArtifacts);
  }

  public CompilationArtifacts(
      Iterable<Artifact> srcs,
      Iterable<Artifact> nonArcSrcs,
      Iterable<Artifact> additionalHdrs,
      IntermediateArtifacts intermediateArtifacts) {
    this.srcs = srcs;
    this.nonArcSrcs = nonArcSrcs;
    this.additionalHdrs = additionalHdrs;

    // Note: the condition under which we set an archive artifact needs to match the condition for
    // which we create the archive in compilation_support.bzl.  In particular, if srcs are all
    // headers, we don't generate an archive.
    if (!Iterables.isEmpty(FileType.filter(srcs, COMPILABLE_SRCS_TYPE))
        || !Iterables.isEmpty(FileType.filter(srcs, PRECOMPILED_SRCS_TYPE))
        || Iterables.any(srcs, Artifact::isTreeArtifact)
        || !Iterables.isEmpty(nonArcSrcs)) {
      this.archive = Optional.of(intermediateArtifacts.archive());
    } else {
      this.archive = Optional.absent();
    }
  }

  Iterable<Artifact> getSrcs() {
    return srcs;
  }

  @StarlarkMethod(name = "srcs", documented = false, structField = true)
  public Sequence<Artifact> getSrcsForStarlark() {
    return StarlarkList.immutableCopyOf(getSrcs());
  }

  Iterable<Artifact> getNonArcSrcs() {
    return nonArcSrcs;
  }

  @StarlarkMethod(name = "non_arc_srcs", documented = false, structField = true)
  public Sequence<Artifact> getNonArcSrcsForStarlark() {
    return StarlarkList.immutableCopyOf(getNonArcSrcs());
  }

  /** Returns the public headers that aren't included in the hdrs attribute. */
  Iterable<Artifact> getAdditionalHdrs() {
    return additionalHdrs;
  }

  @StarlarkMethod(name = "additional_hdrs", documented = false, structField = true)
  public Sequence<Artifact> getAdditionalHdrsForStarlark() {
    return StarlarkList.immutableCopyOf(getAdditionalHdrs());
  }

  /**
   * Returns the output archive library (.a) file created by combining object files of the srcs, non
   * arc srcs, and precompiled srcs of this artifact collection. Returns absent if there are no such
   * source files for which to create an archive library.
   */
  Optional<Artifact> getArchive() {
    return archive;
  }

  @StarlarkMethod(name = "archive", documented = false, allowReturnNones = true, structField = true)
  @Nullable
  public Artifact getArchiveForStarlark() {
    return archive.orNull();
  }
}
