// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.util.Set;

/** Contains information about the results of test execution. */
@AutoValue
public abstract class StandaloneTestResult {

  /** Returns the SpawnResults created by the test, if any. */
  public abstract Set<SpawnResult> spawnResults();

  /** Returns the TestResultData for the test. */
  public abstract TestResultData testResultData();

  /** Returns a builder that can be used to construct a {@link StandaloneTestResult} object. */
  public static Builder builder() {
    return new AutoValue_StandaloneTestResult.Builder();
  }

  /** Creates a StandaloneTestResult, given spawnResults and testResultData. */
  public static StandaloneTestResult create(
      Set<SpawnResult> spawnResults, TestResultData testResultData) {
    return builder().setSpawnResults(spawnResults).setTestResultData(testResultData).build();
  }

  /** Builder for a {@link StandaloneTestResult} instance, which is immutable once built. */
  @AutoValue.Builder
  public abstract static class Builder {

    /** Returns the SpawnResults for the test, if any. */
    abstract Set<SpawnResult> spawnResults();

    /** Sets the SpawnResults for the test. */
    public abstract Builder setSpawnResults(Set<SpawnResult> spawnResults);

    /** Sets the TestResultData for the test. */
    public abstract Builder setTestResultData(TestResultData testResultData);

    abstract StandaloneTestResult realBuild();

    /**
     * Returns an immutable StandaloneTestResult object.
     *
     * <p>The set of SpawnResults is also made immutable here.
     */
    public StandaloneTestResult build() {
      return this.setSpawnResults(ImmutableSet.copyOf(spawnResults())).realBuild();
    }
  }
}
