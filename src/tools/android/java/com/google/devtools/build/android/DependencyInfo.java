// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.auto.value.AutoValue;
import java.util.Comparator;

/** Encapsulates information about a Bazel dependency with respect to the current target. */
@AutoValue
public abstract class DependencyInfo {
  public abstract String label();

  public abstract DependencyType dependencyType();

  /** Placeholder for when dependency information is unknown (or irrelevant). */
  public static final DependencyInfo UNKNOWN = create("", DependencyType.UNKNOWN);

  static DependencyInfo create(String label, DependencyType dependencyType) {
    return new AutoValue_DependencyInfo(label, dependencyType);
  }

  /**
   * The type of dependency relationship, in terms of its distance from the current target. If a
   * resource is combined from multiple sources, we use the DependencyType of the closest source.
   */
  public static enum DependencyType {
    /**
     * The current target. This corresponds to the "--primaryData" option of
     * ResourceProcessorBusyBox.
     */
    PRIMARY,

    /** Direct dependency. */
    DIRECT,

    /** Transitive dependency. */
    TRANSITIVE,

    /** Unknown dependency. */
    UNKNOWN
  }

  /** Compares two {@link DependencyInfo}s by their distance from the current target. */
  public static final Comparator<DependencyInfo> DISTANCE_COMPARATOR =
      Comparator.comparing(DependencyInfo::dependencyType);
}
