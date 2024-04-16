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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
@VisibleForTesting
public final class TestsForTargetPatternValue implements SkyValue {
  private final ResolvedTargets<Label> labels;

  TestsForTargetPatternValue(ResolvedTargets<Label> labels) {
    this.labels = Preconditions.checkNotNull(labels);
  }

  public ResolvedTargets<Label> getLabels() {
    return labels;
  }

  /**
   * Create a target pattern value key.
   *
   * @param targets the set of targets to be expanded
   */
  @ThreadSafe
  public static SkyKey key(Collection<Label> targets) {
    return new TestsForTargetPatternKey(ImmutableSortedSet.copyOf(targets));
  }

  /** A list of targets of which all test suites should be expanded. */
  @ThreadSafe
  static final class TestsForTargetPatternKey implements SkyKey {
    private final ImmutableSortedSet<Label> targets;

    TestsForTargetPatternKey(ImmutableSortedSet<Label> targets) {
      this.targets = targets;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TEST_SUITE_EXPANSION;
    }

    public ImmutableSet<Label> getTargets() {
      return targets;
    }

    @Override
    public String toString() {
      return "ExpandTestSuites(" + targets.toString() + ")";
    }

    @Override
    public int hashCode() {
      return targets.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TestsForTargetPatternKey other)) {
        return false;
      }
      return other.targets.equals(targets);
    }
  }
}
