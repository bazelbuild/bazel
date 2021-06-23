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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.TestTimeout;

/** A {@link TransitiveInfoProvider} for configured targets that implement test rules. */
@Immutable
public final class TestProvider implements TransitiveInfoProvider {
  private final TestParams testParams;

  public TestProvider(TestParams testParams) {
    this.testParams = testParams;
  }

  /**
   * Returns the {@link TestParams} object for the test represented by the corresponding configured
   * target.
   */
  public TestParams getTestParams() {
    return testParams;
  }

  /**
   * Returns the test status artifacts for a specified configured target
   *
   * @param target the configured target. Should belong to a test rule.
   * @return the test status artifacts
   */
  public static ImmutableList<Artifact.DerivedArtifact> getTestStatusArtifacts(
      TransitiveInfoCollection target) {
    return target.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
  }

  /** A value class describing the properties of a test. */
  // Non-final only for mocking.
  public static class TestParams {
    private final int runs;
    private final int shards;
    private final boolean runsDetectsFlakes;
    private final TestTimeout timeout;
    private final String testRuleClass;
    private final ImmutableList<Artifact.DerivedArtifact> testStatusArtifacts;
    private final ImmutableList<Artifact> coverageArtifacts;
    private final FilesToRunProvider coverageReportGenerator;
    private final ImmutableList<ActionInput> outputs;

    /**
     * Don't call this directly. Instead use {@link
     * com.google.devtools.build.lib.analysis.test.TestActionBuilder}.
     */
    TestParams(
        int runs,
        int shards,
        boolean runsDetectsFlakes,
        TestTimeout timeout,
        String testRuleClass,
        ImmutableList<Artifact.DerivedArtifact> testStatusArtifacts,
        ImmutableList<Artifact> coverageArtifacts,
        FilesToRunProvider coverageReportGenerator,
        ImmutableList<ActionInput> outputs) {
      this.runs = runs;
      this.shards = shards;
      this.runsDetectsFlakes = runsDetectsFlakes;
      this.timeout = timeout;
      this.testRuleClass = testRuleClass;
      this.testStatusArtifacts = testStatusArtifacts;
      this.coverageArtifacts = coverageArtifacts;
      this.coverageReportGenerator = coverageReportGenerator;
      this.outputs = outputs;
    }

    /** Returns the number of times this test should be run. */
    public int getRuns() {
      return runs;
    }

    /** Returns the number of shards for this test. */
    public int getShards() {
      return shards;
    }

    /** Returns true iff multiple runs per shard should be aggregated for flake detection. */
    public boolean runsDetectsFlakes() {
      return runsDetectsFlakes;
    }

    /** Returns the timeout of this test. */
    public TestTimeout getTimeout() {
      return timeout;
    }

    /** Returns the test rule class. */
    public String getTestRuleClass() {
      return testRuleClass;
    }

    /**
     * Returns a list of test status artifacts that represent serialized test status protobuffers
     * produced by testing this target.
     */
    public ImmutableList<Artifact.DerivedArtifact> getTestStatusArtifacts() {
      return testStatusArtifacts;
    }

    /** Returns the coverageArtifacts. */
    public ImmutableList<Artifact> getCoverageArtifacts() {
      return coverageArtifacts;
    }

    /** Returns the coverage report generator tool. */
    public FilesToRunProvider getCoverageReportGenerator() {
      return coverageReportGenerator;
    }

    /** Returns the list of mandatory and optional test outputs. */
    public ImmutableList<ActionInput> getOutputs() {
      return outputs;
    }
  }
}
