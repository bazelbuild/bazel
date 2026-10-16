// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Objects;

/**
 * Provides the transitive visibility groups that a target belongs to. If a target belongs to a
 * transitive visibility group, it may only be depended on by other targets that also belong to the
 * same group.
 */
public class TransitiveVisibilityProvider implements TransitiveInfoProvider {

  /** A pairing of a transitive visibility restriction set and the package group that defined it. */
  public static final class Requirement {
    private final PackageSpecificationProvider allowedPackages;
    private final Label label;

    public Requirement(PackageSpecificationProvider allowedPackages, Label label) {
      this.allowedPackages = checkNotNull(allowedPackages);
      this.label = checkNotNull(label);
    }

    public PackageSpecificationProvider getAllowedPackages() {
      return allowedPackages;
    }

    public Label getLabel() {
      return label;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Requirement that)) {
        return false;
      }
      return allowedPackages.equals(that.allowedPackages) && label.equals(that.label);
    }

    @Override
    public int hashCode() {
      return Objects.hash(allowedPackages, label);
    }
  }

  private final ImmutableSet<Requirement> transitiveVisibility;

  /**
   * Creates a new {@link TransitiveVisibilityProvider} from a set of transitive visibility
   * declarations.
   */
  public TransitiveVisibilityProvider(ImmutableSet<Requirement> transitiveVisibility) {
    // We should only try to create a provider if there is a non-empty transitive visibility.
    checkNotNull(transitiveVisibility);
    checkArgument(!transitiveVisibility.isEmpty());

    this.transitiveVisibility = transitiveVisibility;
  }

  /** Returns the set of transitive visibility declarations for the target. */
  ImmutableSet<Requirement> getTransitiveVisibility() {
    return transitiveVisibility;
  }

  /** Returns the set of labels of the package groups that define this transitive visibility. */
  public ImmutableSet<Label> getTransitiveVisibilityLabels() {
    return transitiveVisibility.stream().map(Requirement::getLabel).collect(toImmutableSet());
  }
}
