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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
final class TestExpansionValue implements SkyValue {
  private final ResolvedTargets<Label> labels;

  TestExpansionValue(ResolvedTargets<Label> labels) {
    this.labels = Preconditions.checkNotNull(labels);
  }

  public ResolvedTargets<Label> getLabels() {
    return labels;
  }

  /**
   * Create a target pattern value key.
   *
   * @param target the target to be expanded
   */
  @ThreadSafe
  public static SkyKey key(Target target, boolean strict) {
    Preconditions.checkState(TargetUtils.isTestSuiteRule(target));
    return new TestExpansionKey(target.getLabel(), strict);
  }

  /** A list of targets of which all test suites should be expanded. */
  @ThreadSafe
  static final class TestExpansionKey implements SkyKey {
    private final Label label;
    private final boolean strict;

    public TestExpansionKey(Label label, boolean strict) {
      this.label = label;
      this.strict = strict;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TESTS_IN_SUITE;
    }

    public Label getLabel() {
      return label;
    }

    public boolean isStrict() {
      return strict;
    }

    @Override
    public String toString() {
      return "TestsInSuite(" + label + ", strict=" + strict + ")";
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, strict);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TestExpansionKey other)) {
        return false;
      }
      return other.label.equals(label) && other.strict == strict;
    }
  }
}
