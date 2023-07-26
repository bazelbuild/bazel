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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcDebugInfoContextApi;
import java.util.Collection;
import java.util.Objects;

/**
 * A struct that stores .dwo files which can be combined into a .dwp in the packaging step. See
 * https://gcc.gnu.org/wiki/DebugFission for details.
 */
@Immutable
public final class CcDebugInfoContext implements CcDebugInfoContextApi {

  public static final CcDebugInfoContext EMPTY =
      new CcDebugInfoContext(
          /* transitiveDwoFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          /* transitivePicDwoFiles= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  private final NestedSet<Artifact> transitiveDwoFiles;
  private final NestedSet<Artifact> transitivePicDwoFiles;

  public CcDebugInfoContext(
      NestedSet<Artifact> transitiveDwoFiles, NestedSet<Artifact> transitivePicDwoFiles) {
    this.transitiveDwoFiles = transitiveDwoFiles;
    this.transitivePicDwoFiles = transitivePicDwoFiles;
  }

  /** Merge multiple {@link CcDebugInfoContext}s into one. */
  public static CcDebugInfoContext merge(Collection<CcDebugInfoContext> contexts) {
    if (contexts.isEmpty()) {
      return EMPTY;
    }
    NestedSetBuilder<Artifact> transitiveDwoFiles = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> transitivePicDwoFiles = NestedSetBuilder.stableOrder();

    for (CcDebugInfoContext context : contexts) {
      transitiveDwoFiles.addTransitive(context.getTransitiveDwoFiles());
      transitivePicDwoFiles.addTransitive(context.getTransitivePicDwoFiles());
    }

    return new CcDebugInfoContext(transitiveDwoFiles.build(), transitivePicDwoFiles.build());
  }

  public static CcDebugInfoContext from(CcCompilationOutputs outputs) {
    return new CcDebugInfoContext(
        NestedSetBuilder.wrap(Order.STABLE_ORDER, outputs.getDwoFiles()),
        NestedSetBuilder.wrap(Order.STABLE_ORDER, outputs.getPicDwoFiles()));
  }

  /**
   * Returns the .dwo files that should be included in this target's .dwp packaging (if this
   * target is linked) or passed through to a dependant's .dwp packaging (e.g. if this is a
   * cc_library depended on by a statically linked cc_binary).
   *
   * Assumes the corresponding link consumes .o files (vs. .pic.o files).
   */
  public NestedSet<Artifact> getTransitiveDwoFiles() {
    return transitiveDwoFiles;
  }

  /**
   * Same as above, but assumes the corresponding link consumes pic.o files.
   */
  public NestedSet<Artifact> getTransitivePicDwoFiles() {
    return transitivePicDwoFiles;
  }

  @Override
  public Depset getStarlarkTransitiveFiles() {
    return Depset.of(Artifact.class, getTransitiveDwoFiles());
  }

  @Override
  public Depset getStarlarkTransitivePicFiles() {
    return Depset.of(Artifact.class, getTransitivePicDwoFiles());
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CcDebugInfoContext that = (CcDebugInfoContext) o;
    return Objects.equals(transitiveDwoFiles, that.transitiveDwoFiles)
        && Objects.equals(transitivePicDwoFiles, that.transitivePicDwoFiles);
  }

  @Override
  public int hashCode() {
    return Objects.hash(transitiveDwoFiles, transitivePicDwoFiles);
  }
}
