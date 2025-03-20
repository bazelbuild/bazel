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
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue.UnsuccessfulTransitiveTargetValue;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire transitive
 * closure of a target.
 *
 * <p>Use {@link #unsuccessfulTransitiveLoading(NestedSet, NoSuchTargetException)} if this or any of
 * its transitive values failed to load.
 */
@Immutable
@ThreadSafe
public sealed class TransitiveTargetValue implements SkyValue
    permits UnsuccessfulTransitiveTargetValue {
  private final NestedSet<Label> transitiveTargets;

  private TransitiveTargetValue(NestedSet<Label> transitiveTargets) {
    this.transitiveTargets = transitiveTargets;
  }

  /**
   * A value that represents an unsuccessful target loading.
   *
   * <p>If this TransitiveTargetKey that failed to load, {@code errorLoadingTarget} is not null.
   *
   * <p>If this TransitiveTargetKey loaded successfully, but some other key in its transitive
   * dependencies has failed to load, then this value is {@link UnsuccessfulTransitiveTargetValue}
   * with a null `errorLoadingTarget`.
   *
   * <p>This is kept as a subclass so as to not burden the TransitiveTargetValue class with wasteful
   * fields for error handling.
   */
  static final class UnsuccessfulTransitiveTargetValue extends TransitiveTargetValue {

    private final NoSuchTargetException errorLoadingTarget;

    private UnsuccessfulTransitiveTargetValue(
        NestedSet<Label> transitiveTargets, NoSuchTargetException errorLoadingTarget) {
      super(transitiveTargets);
      this.errorLoadingTarget = errorLoadingTarget;
    }

    @Override
    @Nullable
    public NoSuchTargetException getErrorLoadingTarget() {
      return errorLoadingTarget;
    }

    @Override
    public boolean encounteredLoadingError() {
      return true;
    }
  }

  static TransitiveTargetValue unsuccessfulTransitiveLoading(
      NestedSet<Label> transitiveTargets, @Nullable NoSuchTargetException errorLoadingTarget) {
    return new UnsuccessfulTransitiveTargetValue(transitiveTargets, errorLoadingTarget);
  }

  static TransitiveTargetValue successfulTransitiveLoading(NestedSet<Label> transitiveTargets) {
    return new TransitiveTargetValue(transitiveTargets);
  }

  /** Returns the targets that were transitively loaded. */
  public NestedSet<Label> getTransitiveTargets() {
    return transitiveTargets;
  }

  public boolean encounteredLoadingError() {
    return false;
  }

  @Nullable
  public NoSuchTargetException getErrorLoadingTarget() {
    return null;
  }
}
