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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadingResult;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
@VisibleForTesting
public final class TargetPatternPhaseValue implements SkyValue {

  private final ImmutableSet<Target> targets;
  @Nullable private final ImmutableSet<Target> testsToRun;
  private final boolean hasError;
  private final boolean hasPostExpansionError;

  private final ImmutableSet<Target> filteredTargets;
  private final ImmutableSet<Target> testFilteredTargets;

  // This field is only for the purposes of generating the LoadingPhaseCompleteEvent.
  // TODO(ulfjack): Support EventBus event posting in Skyframe, and remove this code again.
  private final ImmutableSet<Target> testSuiteTargets;
  private final String workspaceName;

  TargetPatternPhaseValue(ImmutableSet<Target> targets, @Nullable ImmutableSet<Target> testsToRun,
      boolean hasError, boolean hasPostExpansionError, ImmutableSet<Target> filteredTargets,
      ImmutableSet<Target> testFilteredTargets, ImmutableSet<Target> testSuiteTargets,
      String workspaceName) {
    this.targets = Preconditions.checkNotNull(targets);
    this.testsToRun = testsToRun;
    this.hasError = hasError;
    this.hasPostExpansionError = hasPostExpansionError;
    this.filteredTargets = Preconditions.checkNotNull(filteredTargets);
    this.testFilteredTargets = Preconditions.checkNotNull(testFilteredTargets);
    this.testSuiteTargets = Preconditions.checkNotNull(testSuiteTargets);
    this.workspaceName = workspaceName;
  }

  public ImmutableSet<Target> getTargets() {
    return targets;
  }

  @Nullable
  public ImmutableSet<Target> getTestsToRun() {
    return testsToRun;
  }

  public boolean hasError() {
    return hasError;
  }

  public boolean hasPostExpansionError() {
    return hasPostExpansionError;
  }

  public ImmutableSet<Target> getFilteredTargets() {
    return filteredTargets;
  }

  public ImmutableSet<Target> getTestFilteredTargets() {
    return testFilteredTargets;
  }

  public ImmutableSet<Target> getTestSuiteTargets() {
    return testSuiteTargets;
  }

  public String getWorkspaceName() {
    return workspaceName;
  }

  public LoadingResult toLoadingResult() {
    return new LoadingResult(
        hasError(), hasPostExpansionError(), getTargets(), getTestsToRun(), getWorkspaceName());
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

  /** Create a target pattern phase value key. */
  @ThreadSafe
  public static SkyKey key(
      ImmutableList<String> targetPatterns,
      String offset,
      boolean compileOneDependency,
      boolean buildTestsOnly,
      boolean determineTests,
      ImmutableList<String> buildTargetFilter,
      boolean buildManualTests,
      @Nullable TestFilter testFilter) {
    return new TargetPatternPhaseKey(
        targetPatterns,
        offset,
        compileOneDependency,
        buildTestsOnly,
        determineTests,
        buildTargetFilter,
        buildManualTests,
        testFilter);
  }

  /** The configuration needed to run the target pattern evaluation phase. */
  @ThreadSafe
  static final class TargetPatternPhaseKey implements SkyKey, Serializable {
    private final ImmutableList<String> targetPatterns;
    private final String offset;
    private final boolean compileOneDependency;
    private final boolean buildTestsOnly;
    private final boolean determineTests;
    private final ImmutableList<String> buildTargetFilter;
    private final boolean buildManualTests;
    @Nullable private final TestFilter testFilter;

    public TargetPatternPhaseKey(
        ImmutableList<String> targetPatterns,
        String offset,
        boolean compileOneDependency,
        boolean buildTestsOnly,
        boolean determineTests,
        ImmutableList<String> buildTargetFilter,
        boolean buildManualTests,
        @Nullable TestFilter testFilter) {
      this.targetPatterns = Preconditions.checkNotNull(targetPatterns);
      this.offset = Preconditions.checkNotNull(offset);
      this.compileOneDependency = compileOneDependency;
      this.buildTestsOnly = buildTestsOnly;
      this.determineTests = determineTests;
      this.buildTargetFilter = Preconditions.checkNotNull(buildTargetFilter);
      this.buildManualTests = buildManualTests;
      this.testFilter = testFilter;
      if (buildTestsOnly || determineTests) {
        Preconditions.checkNotNull(testFilter);
      }
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN_PHASE;
    }

    @Override
    public Object argument() {
      return this;
    }

    public ImmutableList<String> getTargetPatterns() {
      return targetPatterns;
    }

    public String getOffset() {
      return offset;
    }

    public boolean getCompileOneDependency() {
      return compileOneDependency;
    }

    public boolean getBuildTestsOnly() {
      return buildTestsOnly;
    }

    public boolean getDetermineTests() {
      return determineTests;
    }

    public ImmutableList<String> getBuildTargetFilter() {
      return buildTargetFilter;
    }

    public boolean getBuildManualTests() {
      return buildManualTests;
    }

    public TestFilter getTestFilter() {
      return testFilter;
    }

    @Override
    public String toString() {
      StringBuilder result = new StringBuilder();
      result.append(targetPatterns);
      if (!offset.isEmpty()) {
        result.append(" OFFSET=").append(offset);
      }
      result.append(compileOneDependency ? " COMPILE_ONE_DEPENDENCY" : "");
      result.append(buildTestsOnly ? " BUILD_TESTS_ONLY" : "");
      result.append(determineTests ? " DETERMINE_TESTS" : "");
      result.append(testFilter != null ? testFilter : "");
      return result.toString();
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          targetPatterns, offset, compileOneDependency, buildTestsOnly, determineTests,
          buildManualTests, testFilter);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TargetPatternPhaseKey)) {
        return false;
      }
      TargetPatternPhaseKey other = (TargetPatternPhaseKey) obj;
      return other.targetPatterns.equals(this.targetPatterns)
          && other.offset.equals(this.offset)
          && other.compileOneDependency == compileOneDependency
          && other.buildTestsOnly == buildTestsOnly
          && other.determineTests == determineTests
          && other.buildTargetFilter.equals(buildTargetFilter)
          && other.buildManualTests == buildManualTests
          && Objects.equals(other.testFilter, testFilter);
    }
  }
}
