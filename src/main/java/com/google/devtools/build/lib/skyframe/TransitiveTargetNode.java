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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire
 * transitive closure of a target.
 *
 * This will probably be unnecessary once other refactorings occur throughout the codebase
 * which make loading/analysis interleaving more feasible, or we will migrate "blaze query" to
 * use this to evaluate its Target graph.
 */
@Immutable
@ThreadSafe
public class TransitiveTargetNode implements Node {

  private final NestedSet<PathFragment> transitiveSuccessfulPkgs;
  private final NestedSet<PathFragment> transitiveUnsuccessfulPkgs;
  private final NestedSet<Label> transitiveTargets;
  @Nullable private final NestedSet<Label> transitiveRootCauses;
  @Nullable private final NoSuchTargetException errorLoadingTarget;

  private TransitiveTargetNode(NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets,
      @Nullable NestedSet<Label> transitiveRootCauses,
      @Nullable NoSuchTargetException errorLoadingTarget) {
    this.transitiveSuccessfulPkgs = transitiveSuccessfulPkgs;
    this.transitiveUnsuccessfulPkgs = transitiveUnsuccessfulPkgs;
    this.transitiveTargets = transitiveTargets;
    this.transitiveRootCauses = transitiveRootCauses;
    this.errorLoadingTarget = errorLoadingTarget;
  }

  static TransitiveTargetNode unsuccessfulTransitiveLoading(
      NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets,
      NestedSet<Label> rootCauses, @Nullable NoSuchTargetException errorLoadingTarget) {
    return new TransitiveTargetNode(transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs,
        transitiveTargets, rootCauses, errorLoadingTarget);
  }

  static TransitiveTargetNode successfulTransitiveLoading(
      NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets) {
    return new TransitiveTargetNode(transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs,
        transitiveTargets, null, null);
  }

  /** Returns the error, if any, from loading the target. */
  @Nullable
  public NoSuchTargetException getErrorLoadingTarget() {
    return errorLoadingTarget;
  }

  /** Returns the packages that were transitively successfully loaded. */
  public NestedSet<PathFragment> getTransitiveSuccessfulPackages() {
    return transitiveSuccessfulPkgs;
  }

  /** Returns the packages that were transitively successfully loaded. */
  public NestedSet<PathFragment> getTransitiveUnsuccessfulPackages() {
    return transitiveUnsuccessfulPkgs;
  }

  /** Returns the targets that were transitively loaded. */
  public NestedSet<Label> getTransitiveTargets() {
    return transitiveTargets;
  }

  /** Returns the root causes, if any, of why targets weren't loaded. */
  @Nullable
  public NestedSet<Label> getTransitiveRootCauses() {
    return transitiveRootCauses;
  }

  @ThreadSafe
  public static NodeKey key(Label label) {
    return new NodeKey(NodeTypes.TRANSITIVE_TARGET, label);
  }
}
