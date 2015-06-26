// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * This class encapsulates logic behind computing final target set based on separate results from a
 * list of target patterns (eg, //foo:all -//bar/... //foo:test).
 */
abstract class TargetPatternsResultBuilder {
  private Map<PackageIdentifier, Package> packages = null;
  private boolean hasError = false;

  /**
   * Sets that there was an error, during evaluation. 
   */
  public void setError() {
    hasError = true;
  }

  /**
   * Returns final set of targets and sets error flag if required.
   */
  public ResolvedTargets<Target> build(WalkableGraph walkableGraph) throws TargetParsingException {
    precomputePackages(walkableGraph);
    ResolvedTargets.Builder<Target> resolvedTargetsBuilder = buildInternal();
    if (hasError) {
      resolvedTargetsBuilder.setError();
    }
    return resolvedTargetsBuilder.build();
  }

  /**
   * Transforms {@code ResolvedTargets<Label>} to {@code ResolvedTargets<Target>}. Note that this
   * method is using information about packages, so {@link #precomputePackages} has to be called
   * before this method.
   */
  protected ResolvedTargets.Builder<Target> transformLabelsIntoTargets(
      ResolvedTargets<Label> resolvedLabels) {
    // precomputePackages has to be called before this method.
    ResolvedTargets.Builder<Target> resolvedTargetsBuilder = ResolvedTargets.builder();
    Preconditions.checkNotNull(packages);
    for (Label label : resolvedLabels.getTargets()) {
      resolvedTargetsBuilder.add(getExistingTarget(label));
    }
    for (Label label : resolvedLabels.getFilteredTargets()) {
      resolvedTargetsBuilder.remove(getExistingTarget(label));
    }
    return resolvedTargetsBuilder;
  }

  private void precomputePackages(WalkableGraph walkableGraph) {
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
    for (Label label : getLabels()) {
      packagesIdentifiers.add(label.getPackageIdentifier());
    }
    return packagesIdentifiers;
  }

  private Package findPackageInGraph(PackageIdentifier pkgIdentifier,
      WalkableGraph walkableGraph) {
    SkyKey key = PackageValue.key(pkgIdentifier);
    Package pkg = null;
    NoSuchPackageException nspe = (NoSuchPackageException) walkableGraph.getException(key);
    if (nspe != null) {
      pkg = nspe.getPackage();
    } else {
      pkg = ((PackageValue) walkableGraph.getValue(key)).getPackage();
    }
    Preconditions.checkNotNull(pkg, pkgIdentifier);
    return pkg;
  }

  /**
   * Adds the result from expansion of positive target pattern (eg, "//foo:all").
   */
  abstract void addLabelsOfNegativePattern(ResolvedTargets<Label> labels);

  /**
   * Adds the result from expansion of negative target pattern (eg, "-//foo:all").
   */
  abstract void addLabelsOfPositivePattern(ResolvedTargets<Label> labels);

  /**
   * Returns {@code ResolvedTargets.Builder<Target>} with final set of targets. Note that this
   * method doesn't set error flag in result.
   */
  abstract ResolvedTargets.Builder<Target> buildInternal() throws TargetParsingException;

  /**
   * Returns target labels from all individual results.
   */
  protected abstract Iterable<Label> getLabels();
}
