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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;

import java.util.Set;

/**
 * Visits a set of Targets and Labels transitively.
 */
public interface TransitivePackageLoader {

  /**
   * Visit the specified labels and follow the transitive closure of their
   * outbound dependencies. If the targets have previously been visited,
   * may do an up-to-date check which will not trigger any of the observers.
   *
   * @param eventHandler the error and warnings eventHandler; must be thread-safe
   * @param targetsToVisit the targets to visit
   * @param labelsToVisit the labels to visit in addition to the targets
   * @param keepGoing if false, stop visitation upon first error.
   * @param parallelThreads number of threads to use in the visitation.
   * @param maxDepth the maximum depth to traverse to.
   */
  boolean sync(EventHandler eventHandler,
               Set<Target> targetsToVisit,
               Set<Label> labelsToVisit,
               boolean keepGoing,
               int parallelThreads,
               int maxDepth) throws InterruptedException;

  /**
   * Returns a read-only view of the set of packages visited since this visitor
   * was constructed.
   *
   * <p>Not thread-safe; do not call during visitation.
   */
  Set<PackageIdentifier> getVisitedPackageNames();

  /**
   * Returns a read-only view of the set of the actual packages visited without error since this
   * visitor was constructed.
   *
   * <p>Use {@link #getVisitedPackageNames()} instead when possible.
   *
   * <p>Not thread-safe; do not call during visitation.
   */
  Set<Package> getErrorFreeVisitedPackages(EventHandler eventHandler);

  /**
   * Return a mapping between the specified top-level targets and root causes. Note that targets in
   * the input that are transitively error free will not be in the output map. "Top-level" targets
   * are the targetsToVisit and labelsToVisit specified in the last sync.
   *
   * <p>May only be called once a keep_going visitation is complete, and prior to
   * trimErrorTracking().
   *
   * @return a mapping of targets to root causes
   */
  Multimap<Label, Label> getRootCauses();
}
