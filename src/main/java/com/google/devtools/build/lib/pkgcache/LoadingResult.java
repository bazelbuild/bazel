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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Target;

import java.util.Collection;

/**
 * The result of the loading phase, i.e., whether there were errors, and which targets were
 * successfully loaded, plus some related metadata.
 */
public final class LoadingResult {
  private final boolean hasTargetPatternError;
  private final boolean hasLoadingError;
  private final ImmutableSet<Target> targetsToAnalyze;
  private final ImmutableSet<Target> testsToRun;

  public LoadingResult(boolean hasTargetPatternError, boolean hasLoadingError,
      Collection<Target> targetsToAnalyze, Collection<Target> testsToRun) {
    this.hasTargetPatternError = hasTargetPatternError;
    this.hasLoadingError = hasLoadingError;
    this.targetsToAnalyze =
        targetsToAnalyze == null ? null : ImmutableSet.copyOf(targetsToAnalyze);
    this.testsToRun = testsToRun == null ? null : ImmutableSet.copyOf(testsToRun);
  }

  /** Whether there were errors during target pattern evaluation. */
  public boolean hasTargetPatternError() {
    return hasTargetPatternError;
  }

  /** Whether there were errors during the loading phase. */
  public boolean hasLoadingError() {
    return hasLoadingError;
  }

  /** Successfully loaded targets that should be built. */
  public Collection<Target> getTargets() {
    return targetsToAnalyze;
  }

  /** Successfully loaded targets that should be run as tests. Must be a subset of the targets. */
  public Collection<Target> getTestsToRun() {
    return testsToRun;
  }
}