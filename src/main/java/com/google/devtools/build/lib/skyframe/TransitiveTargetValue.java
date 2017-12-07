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

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire
 * transitive closure of a target.
 *
 * <p>Note that this class drops transitive targets during serialization!
 */
@Immutable
@ThreadSafe
public class TransitiveTargetValue implements SkyValue {
  // Non-final for serialization purposes.
  private NestedSet<Label> transitiveTargets;
  @Nullable private NestedSet<Label> transitiveRootCauses;
  @Nullable private NoSuchTargetException errorLoadingTarget;
  private NestedSet<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments;

  private TransitiveTargetValue(
      NestedSet<Label> transitiveTargets,
      @Nullable NestedSet<Label> transitiveRootCauses,
      @Nullable NoSuchTargetException errorLoadingTarget,
      NestedSet<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments) {
    this.transitiveTargets = transitiveTargets;
    this.transitiveRootCauses = transitiveRootCauses;
    this.errorLoadingTarget = errorLoadingTarget;
    this.transitiveConfigFragments = transitiveConfigFragments;
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    // Deliberately do not write out transitiveTargets. There is a lot of those and they drive
    // serialization costs through the roof, both in terms of space and of time.
    // TODO(bazel-team): Deal with this properly once we have efficient serialization of NestedSets.
    out.writeObject(transitiveRootCauses);
    out.writeObject(errorLoadingTarget);
    out.writeObject(transitiveConfigFragments);
  }

  @SuppressWarnings("unchecked")
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    // TODO(bazel-team): Deal with transitiveTargets properly.
    transitiveTargets = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    transitiveRootCauses = (NestedSet<Label>) in.readObject();
    errorLoadingTarget = (NoSuchTargetException) in.readObject();
    transitiveConfigFragments =
        (NestedSet<Class<? extends BuildConfiguration.Fragment>>) in.readObject();
  }

  static TransitiveTargetValue unsuccessfulTransitiveLoading(
      NestedSet<Label> transitiveTargets,
      NestedSet<Label> rootCauses, @Nullable NoSuchTargetException errorLoadingTarget,
      NestedSet<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments) {
    return new TransitiveTargetValue(
        transitiveTargets, rootCauses, errorLoadingTarget, transitiveConfigFragments);
  }

  static TransitiveTargetValue successfulTransitiveLoading(
      NestedSet<Label> transitiveTargets,
      NestedSet<Class<? extends BuildConfiguration.Fragment>> transitiveConfigFragments) {
    return new TransitiveTargetValue(transitiveTargets, null, null, transitiveConfigFragments);
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

  /** Returns the root causes, if any, of why targets weren't loaded. */
  @Nullable
  public NestedSet<Label> getTransitiveRootCauses() {
    return transitiveRootCauses;
  }

  /**
   * Returns the set of {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment} classes required to
   * configure a rule's transitive closure. These are used to instantiate the right {@link
   * ConfigurationFragmentValue} instances for a rule's {@link BuildConfigurationValue}.
   *
   * <p>This provides the basis for rule-scoped configurations. For example, Java-related build
   * flags have nothing to do with C++. So changing a Java flag shouldn't invalidate a C++ rule
   * (unless it has transitive dependencies on other Java rules). Likewise, a C++ rule shouldn't
   * fail because the Java configuration doesn't recognize the chosen architecture.
   *
   * <p>The general principle is that a rule can be influenced by the configuration parameters it
   * directly uses and the configuration parameters its transitive dependencies use (since it
   * reads its dependencies as part of analysis). So we need to 1) determine which configuration
   * fragments provide these parameters, 2) load those fragments, then 3) create a configuration
   * from them to feed the rule's configured target. This provides the first step.
   *
   * <p>See
   * {@link com.google.devtools.build.lib.packages.RuleClass.Builder#requiresConfigurationFragments}
   */
  public NestedSet<Class<? extends BuildConfiguration.Fragment>> getTransitiveConfigFragments() {
    return transitiveConfigFragments;
  }
}
