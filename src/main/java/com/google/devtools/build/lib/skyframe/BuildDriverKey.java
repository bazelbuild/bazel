// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.skyframe.CPUHeavySkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.Objects;

/**
 * Wraps an {@link ActionLookupKey}. The evaluation of this SkyKey is the entry point of analyzing
 * the {@link ActionLookupKey} and executing the associated actions.
 */
public final class BuildDriverKey implements CPUHeavySkyKey {
  private final ActionLookupKey actionLookupKey;
  private final TopLevelArtifactContext topLevelArtifactContext;
  private final TestType testType;
  private final boolean strictActionConflictCheck;
  private final boolean explicitlyRequested;
  private final boolean skipIncompatibleExplicitTargets;
  private final boolean isTopLevelAspectDriver;

  private BuildDriverKey(
      ActionLookupKey actionLookupKey,
      TopLevelArtifactContext topLevelArtifactContext,
      boolean strictActionConflictCheck,
      boolean explicitlyRequested,
      boolean skipIncompatibleExplicitTargets,
      boolean isTopLevelAspectDriver,
      TestType testType) {
    this.actionLookupKey = actionLookupKey;
    this.topLevelArtifactContext = topLevelArtifactContext;
    this.strictActionConflictCheck = strictActionConflictCheck;
    this.explicitlyRequested = explicitlyRequested;
    this.skipIncompatibleExplicitTargets = skipIncompatibleExplicitTargets;
    this.isTopLevelAspectDriver = isTopLevelAspectDriver;
    this.testType = testType;
  }

  public static BuildDriverKey ofTopLevelAspect(
      ActionLookupKey actionLookupKey,
      TopLevelArtifactContext topLevelArtifactContext,
      boolean strictActionConflictCheck,
      boolean explicitlyRequested,
      boolean skipIncompatibleExplicitTargets) {
    return new BuildDriverKey(
        actionLookupKey,
        topLevelArtifactContext,
        strictActionConflictCheck,
        explicitlyRequested,
        skipIncompatibleExplicitTargets,
        /* isTopLevelAspectDriver= */ true,
        TestType.NOT_TEST);
  }

  public static BuildDriverKey ofConfiguredTarget(
      ActionLookupKey actionLookupKey,
      TopLevelArtifactContext topLevelArtifactContext,
      boolean strictActionConflictCheck,
      boolean explicitlyRequested,
      boolean skipIncompatibleExplicitTargets,
      TestType testType) {
    return new BuildDriverKey(
        actionLookupKey,
        topLevelArtifactContext,
        strictActionConflictCheck,
        explicitlyRequested,
        skipIncompatibleExplicitTargets,
        /* isTopLevelAspectDriver= */ false,
        testType);
  }

  public TopLevelArtifactContext getTopLevelArtifactContext() {
    return topLevelArtifactContext;
  }

  public ActionLookupKey getActionLookupKey() {
    return actionLookupKey;
  }

  public boolean isTest() {
    return !TestType.NOT_TEST.equals(testType);
  }

  public TestType getTestType() {
    return testType;
  }

  public boolean strictActionConflictCheck() {
    return strictActionConflictCheck;
  }

  public boolean isExplicitlyRequested() {
    return explicitlyRequested;
  }

  public boolean shouldSkipIncompatibleExplicitTargets() {
    return skipIncompatibleExplicitTargets;
  }

  public boolean isTopLevelAspectDriver() {
    return isTopLevelAspectDriver;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.BUILD_DRIVER;
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof BuildDriverKey) {
      BuildDriverKey otherBuildDriverKey = (BuildDriverKey) other;
      return actionLookupKey.equals(otherBuildDriverKey.actionLookupKey)
          && topLevelArtifactContext.equals(otherBuildDriverKey.topLevelArtifactContext)
          && testType.equals(otherBuildDriverKey.testType)
          && strictActionConflictCheck == otherBuildDriverKey.strictActionConflictCheck
          && explicitlyRequested == otherBuildDriverKey.explicitlyRequested;
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        actionLookupKey,
        topLevelArtifactContext,
        testType,
        strictActionConflictCheck,
        explicitlyRequested);
  }

  @Override
  public final String toString() {
    return String.format("ActionLookupKey: %s; TestType: %s", actionLookupKey, testType);
  }

  @Override
  public boolean valueIsShareable() {
    // BuildDriverValue is just a wrapper value that signals that the building of a top level target
    // was concluded. It's meant to be created anew each build, since BuildDriverFunction must be
    // run every build.
    return false;
  }

  enum TestType {
    NOT_TEST("not-test"),
    PARALLEL("parallel"),
    EXCLUSIVE("exclusive"),
    EXCLUSIVE_IF_LOCAL("exclusive-if-local");

    private final String msg;

    TestType(String msg) {
      this.msg = msg;
    }

    public String getMsg() {
      return msg;
    }
  }
}
