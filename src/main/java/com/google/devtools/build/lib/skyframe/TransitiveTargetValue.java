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

import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire
 * transitive closure of a target.
 */
@Immutable
@ThreadSafe
public class TransitiveTargetValue implements SkyValue {
  private final NestedSet<Label> transitiveTargets;
  private final boolean encounteredLoadingError;
  @Nullable private NoSuchTargetException errorLoadingTarget;
  private NestedSet<Class<? extends Fragment>> transitiveConfigFragments;

  private TransitiveTargetValue(
      NestedSet<Label> transitiveTargets,
      boolean encounteredLoadingError,
      @Nullable NoSuchTargetException errorLoadingTarget,
      NestedSet<Class<? extends Fragment>> transitiveConfigFragments) {
    this.transitiveTargets = transitiveTargets;
    this.encounteredLoadingError = encounteredLoadingError;
    this.errorLoadingTarget = errorLoadingTarget;
    this.transitiveConfigFragments = transitiveConfigFragments;
  }

  static TransitiveTargetValue unsuccessfulTransitiveLoading(
      NestedSet<Label> transitiveTargets,
      @Nullable NoSuchTargetException errorLoadingTarget,
      NestedSet<Class<? extends Fragment>> transitiveConfigFragments) {
    return new TransitiveTargetValue(
        transitiveTargets,
        /*encounteredLoadingError=*/ true,
        errorLoadingTarget,
        transitiveConfigFragments);
  }

  static TransitiveTargetValue successfulTransitiveLoading(
      NestedSet<Label> transitiveTargets,
      NestedSet<Class<? extends Fragment>> transitiveConfigFragments) {
    return new TransitiveTargetValue(
        transitiveTargets, /*encounteredLoadingError=*/ false, null, transitiveConfigFragments);
  }

  /** Returns the error, if any, from loading the target. */
  @Nullable
  public NoSuchTargetException getErrorLoadingTarget() {
    return errorLoadingTarget;
  }

  /** Returns the targets that were transitively loaded. */
  public NestedSet<Label> getTransitiveTargets() {
    return transitiveTargets;
  }

  public boolean encounteredLoadingError() {
    return encounteredLoadingError;
  }

  /**
   * Returns the set of {@link Fragment} classes required to configure a rule's transitive closure.
   * These are used to instantiate the right {@link BuildConfigurationValue}.
   *
   * <p>This provides the basis for rule-scoped configurations. For example, Java-related build
   * flags have nothing to do with C++. So changing a Java flag shouldn't invalidate a C++ rule
   * (unless it has transitive dependencies on other Java rules). Likewise, a C++ rule shouldn't
   * fail because the Java configuration doesn't recognize the chosen architecture.
   *
   * <p>The general principle is that a rule can be influenced by the configuration parameters it
   * directly uses and the configuration parameters its transitive dependencies use (since it reads
   * its dependencies as part of analysis). So we need to 1) determine which configuration fragments
   * provide these parameters, 2) load those fragments, then 3) create a configuration from them to
   * feed the rule's configured target. This provides the first step.
   *
   * <p>See {@link
   * com.google.devtools.build.lib.packages.RuleClass.Builder#requiresConfigurationFragments}
   */
  public NestedSet<Class<? extends Fragment>> getTransitiveConfigFragments() {
    return transitiveConfigFragments;
  }
}
