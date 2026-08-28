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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
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

  private final ImmutableSet<Label> targetLabels;
  @Nullable private final ImmutableSet<Label> testsToRunLabels;
  private final ImmutableSet<Label> nonExpandedLabels;
  private final ImmutableSet<Label> explicitTargetLabels;
  private final boolean hasError;
  private final boolean hasPostExpansionError;

  TargetPatternPhaseValue(
      ImmutableSet<Label> targetLabels,
      ImmutableSet<Label> testsToRunLabels,
      ImmutableSet<Label> nonExpandedLabels,
      ImmutableSet<Label> explicitTargetLabels,
      boolean hasError,
      boolean hasPostExpansionError) {
    this.targetLabels = targetLabels;
    this.testsToRunLabels = testsToRunLabels;
    this.nonExpandedLabels = nonExpandedLabels;
    this.explicitTargetLabels = explicitTargetLabels;
    this.hasError = hasError;
    this.hasPostExpansionError = hasPostExpansionError;
  }

  /** Expensive. Results in a Skyframe evaluation. */
  private static ImmutableSet<Target> getTargetsFromLabels(
      Collection<Label> labels, ExtendedEventHandler eventHandler, PackageManager packageManager)
      throws InterruptedException {
    ImmutableSet.Builder<Target> result = ImmutableSet.builderWithExpectedSize(labels.size());
    for (Label label : labels) {
      try {
        result.add(
            packageManager
                .getPackage(eventHandler, label.getPackageIdentifier())
                .getTarget(label.getName()));
      } catch (NoSuchTargetException | NoSuchPackageException e) {
        throw new IllegalStateException(
            "Failed to get preloaded package from TargetPatternPhaseValue for " + label, e);
      }
    }
    return result.build();
  }

  public ImmutableSet<Target> getTargets(
      ExtendedEventHandler eventHandler, PackageManager packageManager)
      throws InterruptedException {
    return getTargetsFromLabels(targetLabels, eventHandler, packageManager);
  }

  public ImmutableSet<Label> getNonExpandedLabels() {
    return nonExpandedLabels;
  }

  /** Returns labels resolved from explicit (non-wildcard) target patterns. */
  public ImmutableSet<Label> getExplicitTargetLabels() {
    return explicitTargetLabels;
  }

  public ImmutableSet<Target> getTestsToRun(
      ExtendedEventHandler eventHandler, PackageManager packageManager)
      throws InterruptedException {
    return getTargetsFromLabels(testsToRunLabels, eventHandler, packageManager);
  }

  public ImmutableSet<Label> getTargetLabels() {
    return targetLabels;
  }

  @Nullable
  public ImmutableSet<Label> getTestsToRunLabels() {
    return testsToRunLabels;
  }

  public boolean hasError() {
    return hasError;
  }

  public boolean hasPostExpansionError() {
    return hasPostExpansionError;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof TargetPatternPhaseValue that)) {
      return false;
    }
    return Objects.equals(this.targetLabels, that.targetLabels)
        && Objects.equals(this.testsToRunLabels, that.testsToRunLabels)
        && Objects.equals(this.explicitTargetLabels, that.explicitTargetLabels)
        && this.hasError == that.hasError
        && this.hasPostExpansionError == that.hasPostExpansionError;
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        this.targetLabels,
        this.testsToRunLabels,
        this.explicitTargetLabels,
        this.hasError,
        this.hasPostExpansionError);
  }

  /** Create a target pattern phase value key. */
  @ThreadSafe
  public static TargetPatternPhaseKey key(
      ImmutableList<String> targetPatterns,
      PathFragment offset,
      boolean compileOneDependency,
      boolean buildTestsOnly,
      boolean determineTests,
      ImmutableList<String> buildTargetFilter,
      boolean buildManualTests,
      boolean expandTestSuites,
      @Nullable TestFilter testFilter,
      boolean keepGoing) {
    return new TargetPatternPhaseKey(
        targetPatterns,
        offset,
        compileOneDependency,
        buildTestsOnly,
        determineTests,
        buildTargetFilter,
        buildManualTests,
        expandTestSuites,
        testFilter,
        keepGoing);
  }

  /**
   * Creates a new target pattern sky key which represents the given target patterns without
   * attempting to filter them in any way (for example, ignores options such as only loading tests).
   *
   * @param targetPatterns list of targets to evaluate
   * @param offset relative path to the working directory
   */
  @ThreadSafe
  public static SkyKey keyWithoutFilters(
      ImmutableList<String> targetPatterns, PathFragment offset, boolean keepGoing) {
    return new TargetPatternPhaseKey(
        targetPatterns,
        offset,
        false,
        false,
        false,
        ImmutableList.of(),
        false,
        false,
        null,
        keepGoing);
  }

  /** The configuration needed to run the target pattern evaluation phase. */
  @ThreadSafe
  @Immutable
  static final class TargetPatternPhaseKey implements SkyKey {
    private final ImmutableList<String> targetPatterns;
    private final PathFragment offset;
    private final boolean compileOneDependency;
    private final boolean buildTestsOnly;
    private final boolean determineTests;
    private final ImmutableList<String> buildTargetFilter;
    private final boolean buildManualTests;
    private final boolean expandTestSuites;
    @Nullable private final TestFilter testFilter;
    private final boolean keepGoing;

    private TargetPatternPhaseKey(
        ImmutableList<String> targetPatterns,
        PathFragment offset,
        boolean compileOneDependency,
        boolean buildTestsOnly,
        boolean determineTests,
        ImmutableList<String> buildTargetFilter,
        boolean buildManualTests,
        boolean expandTestSuites,
        @Nullable TestFilter testFilter,
        boolean keepGoing) {
      this.targetPatterns = Preconditions.checkNotNull(targetPatterns);
      this.offset = Preconditions.checkNotNull(offset);
      this.compileOneDependency = compileOneDependency;
      this.buildTestsOnly = buildTestsOnly;
      this.determineTests = determineTests;
      this.buildTargetFilter = Preconditions.checkNotNull(buildTargetFilter);
      this.buildManualTests = buildManualTests;
      this.expandTestSuites = expandTestSuites;
      this.testFilter = testFilter;
      this.keepGoing = keepGoing;
      if (buildTestsOnly || determineTests) {
        Preconditions.checkNotNull(testFilter);
      }
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN_PHASE;
    }

    public ImmutableList<String> getTargetPatterns() {
      return targetPatterns;
    }

    public PathFragment getOffset() {
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

    public boolean isExpandTestSuites() {
      return expandTestSuites;
    }

    public boolean isKeepGoing() {
      return keepGoing;
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
      result.append(expandTestSuites ? " EXPAND_TEST_SUITES" : "");
      result.append(testFilter != null ? " " + testFilter : "");
      result.append(keepGoing ? " KEEP_GOING" : "");
      return result.toString();
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          targetPatterns,
          offset,
          compileOneDependency,
          buildTestsOnly,
          determineTests,
          buildManualTests,
          expandTestSuites,
          testFilter,
          keepGoing);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TargetPatternPhaseKey other)) {
        return false;
      }
      return other.targetPatterns.equals(this.targetPatterns)
          && other.offset.equals(this.offset)
          && other.compileOneDependency == compileOneDependency
          && other.buildTestsOnly == buildTestsOnly
          && other.determineTests == determineTests
          && other.buildTargetFilter.equals(buildTargetFilter)
          && other.buildManualTests == buildManualTests
          && other.expandTestSuites == expandTestSuites
          && Objects.equals(other.testFilter, testFilter)
          && other.keepGoing == keepGoing;
    }
  }
}
