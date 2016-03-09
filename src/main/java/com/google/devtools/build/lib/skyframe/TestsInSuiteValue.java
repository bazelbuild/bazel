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

import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Objects;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
final class TestsInSuiteValue implements SkyValue {
  private ResolvedTargets<Target> targets;

  TestsInSuiteValue(ResolvedTargets<Target> targets) {
    this.targets = Preconditions.checkNotNull(targets);
  }

  public ResolvedTargets<Target> getTargets() {
    return targets;
  }

  @SuppressWarnings("unused")
  private void writeObject(ObjectOutputStream out) {
    throw new UnsupportedOperationException();
  }

  @SuppressWarnings("unused")
  private void readObject(ObjectInputStream in) {
    throw new UnsupportedOperationException();
  }

  @SuppressWarnings("unused")
  private void readObjectNoData() {
    throw new IllegalStateException();
  }

  /**
   * Create a target pattern value key.
   *
   * @param testSuite the test suite target to be expanded
   */
  @ThreadSafe
  public static SkyKey key(Target testSuite, boolean strict) {
    Preconditions.checkState(TargetUtils.isTestSuiteRule(testSuite));
    return SkyKey.create(SkyFunctions.TESTS_IN_SUITE, new TestsInSuite((Rule) testSuite, strict));
  }

  /**
   * A list of targets of which all test suites should be expanded.
   */
  @ThreadSafe
  static final class TestsInSuite implements Serializable {
    private final Rule testSuite;
    private final boolean strict;

    public TestsInSuite(Rule testSuite, boolean strict) {
      this.testSuite = testSuite;
      this.strict = strict;
    }

    public Rule getTestSuite() {
      return testSuite;
    }

    public boolean isStrict() {
      return strict;
    }

    @Override
    public String toString() {
      return "TestsInSuite(" + testSuite.toString() + ", strict=" + strict + ")";
    }

    @Override
    public int hashCode() {
      return Objects.hash(testSuite, strict);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TestsInSuite)) {
        return false;
      }
      TestsInSuite other = (TestsInSuite) obj;
      return other.testSuite.equals(testSuite) && other.strict == strict;
    }
  }
}
