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
  @Nullable private final NoSuchTargetException errorLoadingTarget;

  private TransitiveTargetValue(
      NestedSet<Label> transitiveTargets,
      boolean encounteredLoadingError,
      @Nullable NoSuchTargetException errorLoadingTarget) {
    this.transitiveTargets = transitiveTargets;
    this.encounteredLoadingError = encounteredLoadingError;
    this.errorLoadingTarget = errorLoadingTarget;
  }

  static TransitiveTargetValue unsuccessfulTransitiveLoading(
      NestedSet<Label> transitiveTargets, @Nullable NoSuchTargetException errorLoadingTarget) {
    return new TransitiveTargetValue(
        transitiveTargets, /*encounteredLoadingError=*/ true, errorLoadingTarget);
  }

  static TransitiveTargetValue successfulTransitiveLoading(NestedSet<Label> transitiveTargets) {
    return new TransitiveTargetValue(transitiveTargets, /*encounteredLoadingError=*/ false, null);
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
}
