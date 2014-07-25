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
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Set;

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
public class TransitiveTargetValue implements SkyValue {

  // Non-final for serialization purposes.
  private NestedSet<PathFragment> transitiveSuccessfulPkgs;
  private NestedSet<PathFragment> transitiveUnsuccessfulPkgs;
  private NestedSet<Label> transitiveTargets;
  @Nullable private NestedSet<Label> transitiveRootCauses;
  @Nullable private NoSuchTargetException errorLoadingTarget;

  private TransitiveTargetValue(NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets,
      @Nullable NestedSet<Label> transitiveRootCauses,
      @Nullable NoSuchTargetException errorLoadingTarget) {
    this.transitiveSuccessfulPkgs = transitiveSuccessfulPkgs;
    this.transitiveUnsuccessfulPkgs = transitiveUnsuccessfulPkgs;
    this.transitiveTargets = transitiveTargets;
    this.transitiveRootCauses = transitiveRootCauses;
    this.errorLoadingTarget = errorLoadingTarget;
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    // It helps to flatten the transitiveSuccessfulPkgs nested set as it has lots of duplicates.
    Set<PathFragment> successfulPkgs = transitiveSuccessfulPkgs.toSet();
    out.writeInt(successfulPkgs.size());
    for (PathFragment pkg : successfulPkgs) {
      out.writeUTF(pkg.toString());
    }

    out.writeObject(transitiveUnsuccessfulPkgs);
    // Deliberately do not write out transitiveTargets. There is a lot of those and they drive
    // serialization costs through the roof, both in terms of space and of time.
    // TODO(bazel-team): Deal with this properly once we have efficient serialization of NestedSets.
    out.writeObject(transitiveRootCauses);
    out.writeObject(errorLoadingTarget);
  }

  @SuppressWarnings("unchecked")
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    int successfulPkgCount = in.readInt();
    NestedSetBuilder<PathFragment> pkgs = NestedSetBuilder.stableOrder();
    for (int i = 0; i < successfulPkgCount; i++) {
      pkgs.add(new PathFragment(in.readUTF()));
    }
    transitiveSuccessfulPkgs = pkgs.build();
    transitiveUnsuccessfulPkgs = (NestedSet<PathFragment>) in.readObject();
    // TODO(bazel-team): Deal with transitiveTargets properly.
    transitiveTargets = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    transitiveRootCauses = (NestedSet<Label>) in.readObject();
    errorLoadingTarget = (NoSuchTargetException) in.readObject();
  }

  static TransitiveTargetValue unsuccessfulTransitiveLoading(
      NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets,
      NestedSet<Label> rootCauses, @Nullable NoSuchTargetException errorLoadingTarget) {
    return new TransitiveTargetValue(transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs,
        transitiveTargets, rootCauses, errorLoadingTarget);
  }

  static TransitiveTargetValue successfulTransitiveLoading(
      NestedSet<PathFragment> transitiveSuccessfulPkgs,
      NestedSet<PathFragment> transitiveUnsuccessfulPkgs, NestedSet<Label> transitiveTargets) {
    return new TransitiveTargetValue(transitiveSuccessfulPkgs, transitiveUnsuccessfulPkgs,
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
  public static SkyKey key(Label label) {
    return new SkyKey(SkyFunctions.TRANSITIVE_TARGET, label);
  }
}
