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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;

import java.util.Collection;

/**
 * This event is fired after the loading phase is complete.
 */
public class LoadingPhaseCompleteEvent {
  private final Collection<Target> targets;
  private final Collection<Target> filteredTargets;
  private final PackageManager.PackageManagerStatistics pkgManagerStats;
  private final long timeInMs;

  /**
   * Construct the event.
   *
   * @param targets the set of active targets that remain
   * @param pkgManagerStats statistics about the package cache
   */
  public LoadingPhaseCompleteEvent(Collection<Target> targets, Collection<Target> filteredTargets,
      PackageManager.PackageManagerStatistics pkgManagerStats, long timeInMs) {
    this.targets = targets;
    this.filteredTargets = filteredTargets;
    this.pkgManagerStats = pkgManagerStats;
    this.timeInMs = timeInMs;
  }

  /**
   * @return The set of active targets remaining, which is a subset of the
   *         targets we attempted to load.
   */
  public Collection<Target> getTargets() {
    return targets;
  }

  /**
   * @return The set of filtered targets.
   */
  public Collection<Target> getFilteredTargets() {
    return filteredTargets;
  }

  /**
   * @return The set of active target labels remaining, which is a subset of the
   *         targets we attempted to load.
   */
  public Iterable<Label> getLabels() {
    return Iterables.transform(targets, TO_LABEL);
  }
  
  public long getTimeInMs() {
    return timeInMs;
  }

  /**
   * Returns the PackageCache statistics.
   */
  public PackageManager.PackageManagerStatistics getPkgManagerStats() {
    return pkgManagerStats;
  }

  private static final Function<Target, Label> TO_LABEL = new Function<Target, Label>() {
    @Override
    public Label apply(Target input) {
      return input.getLabel();
    }
  };
}
