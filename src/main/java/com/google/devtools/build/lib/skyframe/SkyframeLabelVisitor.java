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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.SkyframeTransitivePackageLoader;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * Skyframe-based transitive package loader.
 */
final class SkyframeLabelVisitor implements TransitivePackageLoader {

  private final SkyframeTransitivePackageLoader transitivePackageLoader;
  private final AtomicReference<CyclesReporter> skyframeCyclesReporter;

  private Set<PackageIdentifier> allVisitedPackages;
  private Set<PackageIdentifier> errorFreeVisitedPackages;
  private Set<TransitiveTargetValue> previousBuildTargetValueSet = null;
  private boolean lastBuildKeepGoing;
  private final Multimap<Label, Label> rootCauses = HashMultimap.create();

  SkyframeLabelVisitor(SkyframeTransitivePackageLoader transitivePackageLoader,
      AtomicReference<CyclesReporter> skyframeCyclesReporter) {
    this.transitivePackageLoader = transitivePackageLoader;
    this.skyframeCyclesReporter = skyframeCyclesReporter;
  }

  @Override
  public boolean sync(EventHandler eventHandler, Set<Target> targetsToVisit,
      Set<Label> labelsToVisit, boolean keepGoing, int parallelThreads, int maxDepth)
      throws InterruptedException {
    rootCauses.clear();
    lastBuildKeepGoing = false;
    EvaluationResult<TransitiveTargetValue> result = transitivePackageLoader.loadTransitiveTargets(
        eventHandler, targetsToVisit, labelsToVisit, keepGoing, parallelThreads);
    updateVisitedValues(result.values());
    lastBuildKeepGoing = keepGoing;

    if (!hasErrors(result)) {
      return true;
    }

    Set<Entry<SkyKey, ErrorInfo>> errors = result.errorMap().entrySet();
    if (!keepGoing) {
      // We may have multiple errors, but in non keep_going builds, we're obligated to print only
      // one of them.
      Preconditions.checkState(!errors.isEmpty(), result);
      Entry<SkyKey, ErrorInfo> error = errors.iterator().next();
      ErrorInfo errorInfo = error.getValue();
      SkyKey topLevel = error.getKey();
      Label topLevelLabel = (Label) topLevel.argument();
      if (!Iterables.isEmpty(errorInfo.getCycleInfo())) {
        skyframeCyclesReporter.get().reportCycles(errorInfo.getCycleInfo(), topLevel, eventHandler);
        errorAboutLoadingFailure(topLevelLabel, null, eventHandler);
      } else if (isDirectErrorFromTopLevelLabel(topLevelLabel, labelsToVisit, errorInfo)) {
        // An error caused by a non-top-level label has already been reported during error
        // bubbling but an error caused by the top-level non-target label itself hasn't been
        // reported yet. Note that errors from top-level targets have already been reported
        // during target parsing.
        errorAboutLoadingFailure(topLevelLabel, errorInfo.getException(), eventHandler);
      }
      return false;
    }

    for (Entry<SkyKey, ErrorInfo> errorEntry : errors) {
      SkyKey key = errorEntry.getKey();
      ErrorInfo errorInfo = errorEntry.getValue();
      Preconditions.checkState(key.functionName().equals(SkyFunctions.TRANSITIVE_TARGET), errorEntry);
      Label topLevelLabel = (Label) key.argument();
      if (!Iterables.isEmpty(errorInfo.getCycleInfo())) {
        skyframeCyclesReporter.get().reportCycles(errorInfo.getCycleInfo(), key, eventHandler);
        rootCauses.putAll(
            topLevelLabel, getRootCausesOfCycles(topLevelLabel, errorInfo.getCycleInfo()));
      }
      if (isDirectErrorFromTopLevelLabel(topLevelLabel, labelsToVisit, errorInfo)) {
        // Unlike top-level targets, which have already gone through target parsing,
        // errors directly coming from top-level labels have not been reported yet.
        //
        // See the note in the --nokeep_going case above.
        eventHandler.handle(Event.error(errorInfo.getException().getMessage()));
      }
      warnAboutLoadingFailure(topLevelLabel, eventHandler);
      for (SkyKey badKey : errorInfo.getRootCauses()) {
        if (badKey.functionName().equals(SkyFunctions.PACKAGE)) {
          // Transitive target function may ask for a Package, but don't include this in the root
          // causes. We'll get more precise information from dependencies on transitive and direct
          // target dependencies.
          continue;
        }
        Preconditions.checkState(badKey.argument() instanceof Label,
            "%s %s %s", key, errorInfo, badKey);
        rootCauses.put(topLevelLabel, (Label) badKey.argument());
      }
    }
    for (Label topLevelLabel : result.<Label>keyNames()) {
      SkyKey topLevelTransitiveTargetKey = TransitiveTargetValue.key(topLevelLabel);
      TransitiveTargetValue topLevelTransitiveTargetValue = result.get(topLevelTransitiveTargetKey);
      if (topLevelTransitiveTargetValue.getTransitiveRootCauses() != null) {
        rootCauses.putAll(topLevelLabel, topLevelTransitiveTargetValue.getTransitiveRootCauses());
        warnAboutLoadingFailure(topLevelLabel, eventHandler);
      }
    }
    return false;
  }

  private static boolean hasErrors(EvaluationResult<TransitiveTargetValue> result) {
    if (result.hasError()) {
      return true;
    }
    for (TransitiveTargetValue transitiveTargetValue : result.values()) {
      if (transitiveTargetValue.getTransitiveRootCauses() != null) {
        return true;
      }
    }
    return false;
  }

  private static boolean isDirectErrorFromTopLevelLabel(Label label, Set<Label> topLevelLabels,
      ErrorInfo errorInfo) {
    return errorInfo.getException() != null && topLevelLabels.contains(label)
        && Iterables.contains(errorInfo.getRootCauses(), TransitiveTargetValue.key(label));
  }

  private static void errorAboutLoadingFailure(Label topLevelLabel, @Nullable Throwable throwable,
      EventHandler eventHandler) {
    eventHandler.handle(Event.error(
        "Loading of target '" + topLevelLabel + "' failed; build aborted" +
            (throwable == null ? "" : ": " + throwable.getMessage())));
  }

  private static void warnAboutLoadingFailure(Label label, EventHandler eventHandler) {
    eventHandler.handle(Event.warn(
        // TODO(bazel-team): We use 'analyzing' here so that we print the same message as legacy
        // Blaze. Once we get rid of legacy we should be able to change to 'loading' or
        // similar.
        "errors encountered while analyzing target '" + label + "': it will not be built"));
  }

  private static Set<Label> getRootCausesOfCycles(Label labelToLoad, Iterable<CycleInfo> cycles) {
    ImmutableSet.Builder<Label> builder = ImmutableSet.builder();
    for (CycleInfo cycleInfo : cycles) {
      // The root cause of a cycle depends on the type of a cycle.

      SkyKey culprit = Iterables.getFirst(cycleInfo.getCycle(), null);
      if (culprit == null) {
        continue;
      }
      if (culprit.functionName().equals(SkyFunctions.TRANSITIVE_TARGET)) {
        // For a cycle between build targets, the root cause is the first element of the cycle.
        builder.add((Label) culprit.argument());
      } else {
        // For other types of cycles (e.g. file symlink cycles), the root cause is the furthest
        // target dependency that itself depended on the cycle.
        Label furthestTarget = labelToLoad;
        for (SkyKey skyKey : cycleInfo.getPathToCycle()) {
          if (skyKey.functionName().equals(SkyFunctions.TRANSITIVE_TARGET)) {
            furthestTarget = (Label) skyKey.argument();
          } else {
            break;
          }
        }
        builder.add(furthestTarget);
      }
    }
    return builder.build();
  }

  // Unfortunately we have to do an effective O(TC) visitation after the eval() call above to
  // determine all of the packages in the closure.
  private void updateVisitedValues(Collection<TransitiveTargetValue> targetValues) {
    Set<TransitiveTargetValue> currentBuildTargetValueSet = new HashSet<>(targetValues);
    if (Objects.equals(previousBuildTargetValueSet, currentBuildTargetValueSet)) {
      // The next stanza is slow (and scales with the edge count of the target graph), so avoid
      // the computation if the previous build already did it.
      return;
    }
    NestedSetBuilder<PackageIdentifier> nestedAllPkgsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<PackageIdentifier> nestedErrorFreePkgsBuilder = NestedSetBuilder.stableOrder();
    for (TransitiveTargetValue value : targetValues) {
      nestedAllPkgsBuilder.addTransitive(value.getTransitiveSuccessfulPackages());
      nestedAllPkgsBuilder.addTransitive(value.getTransitiveUnsuccessfulPackages());
      nestedErrorFreePkgsBuilder.addTransitive(value.getTransitiveSuccessfulPackages());
    }
    allVisitedPackages = nestedAllPkgsBuilder.build().toSet();
    errorFreeVisitedPackages = nestedErrorFreePkgsBuilder.build().toSet();
    previousBuildTargetValueSet = currentBuildTargetValueSet;
  }

  @Override
  public Set<PackageIdentifier> getVisitedPackageNames() {
    return allVisitedPackages;
  }

  @Override
  public Set<Package> getErrorFreeVisitedPackages(EventHandler eventHandler) {
    return transitivePackageLoader.retrievePackages(eventHandler, errorFreeVisitedPackages);
  }

  @Override
  public Multimap<Label, Label> getRootCauses() {
    Preconditions.checkState(lastBuildKeepGoing);
    return rootCauses;
  }
}
