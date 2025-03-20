// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * This class encapsulates logic behind computing final target set based on separate results from a
 * list of target patterns (eg, //foo:all -//bar/... //foo:test).
 */
class TargetPatternsResultBuilder {
  private final Set<Label> resolvedLabelsBuilder = CompactHashSet.create();
  private Map<PackageIdentifier, Package> packages;

  /** Returns final set of targets and sets error flag if required. */
  public Collection<Target> build(WalkableGraph walkableGraph) throws InterruptedException {
    precomputePackages(walkableGraph);
    return transformLabelsIntoTargets(resolvedLabelsBuilder);
  }

  /**
   * Transforms {@code ResolvedTargets<Label>} to {@code ResolvedTargets<Target>}. Note that this
   * method is using information about packages, so {@link #precomputePackages} has to be called
   * before this method.
   */
  private Collection<Target> transformLabelsIntoTargets(Set<Label> resolvedLabels) {
    // precomputePackages has to be called before this method.
    Set<Target> targets = CompactHashSet.create();
    Preconditions.checkNotNull(packages);
    for (Label label : resolvedLabels) {
      targets.add(getExistingTarget(label));
    }
    return targets;
  }

  private void precomputePackages(WalkableGraph walkableGraph) throws InterruptedException {
    Set<PackageIdentifier> packagesToRequest = getPackagesIdentifiers();      
    packages = Maps.newHashMapWithExpectedSize(packagesToRequest.size());
    for (PackageIdentifier pkgIdentifier : packagesToRequest) {
      packages.put(pkgIdentifier, findPackageInGraph(pkgIdentifier, walkableGraph));
    }
  }

  private Target getExistingTarget(Label label) {
    Package pkg = Preconditions.checkNotNull(packages.get(label.getPackageIdentifier()), label);
    try {
      return pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      // This exception should not raise, because we are processing it during TargetPatternValues
      // evaluation in SkyframeTargetPatternEvaluator#parseTargetPatternKeys and values with errors
      // are not added to final result.
      throw new IllegalStateException(e);
    }
  }

  private Set<PackageIdentifier> getPackagesIdentifiers() {
    Set<PackageIdentifier> packagesIdentifiers = new HashSet<>();
    for (Label label : resolvedLabelsBuilder) {
      packagesIdentifiers.add(label.getPackageIdentifier());
    }
    return packagesIdentifiers;
  }

  private static Package findPackageInGraph(
      PackageIdentifier pkgIdentifier, WalkableGraph walkableGraph) throws InterruptedException {
    return Preconditions.checkNotNull(
            (PackageValue) walkableGraph.getValue(pkgIdentifier), pkgIdentifier)
        .getPackage();
  }

  /** Adds the result from expansion of negative target pattern (eg, "-//foo:all"). */
  void addLabelsOfPositivePattern(ResolvedTargets<Label> labels) {
    Preconditions.checkArgument(labels.getFilteredTargets().isEmpty());
    resolvedLabelsBuilder.addAll(labels.getTargets());
  }
}
