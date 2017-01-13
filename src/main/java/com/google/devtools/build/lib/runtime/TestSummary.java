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
package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter.Mode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.FailedTestCasesStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Test summary entry. Stores summary information for a single test rule. Also used to sort summary
 * output by status.
 *
 * <p>Invariant: All TestSummary mutations should be performed through the Builder. No direct
 * TestSummary methods (except the constructor) may mutate the object.
 */
@VisibleForTesting // Ideally package-scoped.
public class TestSummary implements Comparable<TestSummary>, BuildEvent {
  /**
   * Builder class responsible for creating and altering TestSummary objects.
   */
  public static class Builder {
    private TestSummary summary;
    private boolean built;

    private Builder() {
      summary = new TestSummary();
      built = false;
    }

    private void mergeFrom(TestSummary existingSummary) {
      // Yuck, manually fill in fields.
      summary.shardRunStatuses =
          MultimapBuilder.hashKeys().arrayListValues().build(existingSummary.shardRunStatuses);
      setTarget(existingSummary.target);
      setStatus(existingSummary.status);
      addCoverageFiles(existingSummary.coverageFiles);
      addPassedLogs(existingSummary.passedLogs);
      addFailedLogs(existingSummary.failedLogs);

      if (existingSummary.failedTestCasesStatus != null) {
        addFailedTestCases(existingSummary.getFailedTestCases(),
            existingSummary.getFailedTestCasesStatus());
      }

      addTestTimes(existingSummary.testTimes);
      addWarnings(existingSummary.warnings);
      setActionRan(existingSummary.actionRan);
      setNumCached(existingSummary.numCached);
      setRanRemotely(existingSummary.ranRemotely);
      setWasUnreportedWrongSize(existingSummary.wasUnreportedWrongSize);
    }

    // Implements copy on write logic, allowing reuse of the same builder.
    private void checkMutation() {
      // If mutating the builder after an object was built, create another copy.
      if (built) {
        built = false;
        TestSummary lastSummary = summary;
        summary = new TestSummary();
        mergeFrom(lastSummary);
      }
    }

    // This used to return a reference to the value on success.
    // However, since it can alter the summary member, inlining it in an
    // assignment to a property of summary was unsafe.
    private void checkMutation(Object value) {
      Preconditions.checkNotNull(value);
      checkMutation();
    }

    public Builder setTarget(ConfiguredTarget target) {
      checkMutation(target);
      summary.target = target;
      return this;
    }

    public Builder setStatus(BlazeTestStatus status) {
      checkMutation(status);
      summary.status = status;
      return this;
    }

    public Builder addCoverageFiles(List<Path> coverageFiles) {
      checkMutation(coverageFiles);
      summary.coverageFiles.addAll(coverageFiles);
      return this;
    }

    public Builder addPassedLogs(List<Path> passedLogs) {
      checkMutation(passedLogs);
      summary.passedLogs.addAll(passedLogs);
      return this;
    }

    public Builder addFailedLogs(List<Path> failedLogs) {
      checkMutation(failedLogs);
      summary.failedLogs.addAll(failedLogs);
      return this;
    }

    public Builder collectFailedTests(TestCase testCase) {
      if (testCase == null) {
        summary.failedTestCasesStatus = FailedTestCasesStatus.NOT_AVAILABLE;
        return this;
      }
      summary.failedTestCasesStatus = FailedTestCasesStatus.FULL;
      return collectFailedTestCases(testCase);
    }

    private Builder collectFailedTestCases(TestCase testCase) {
      if (testCase.getChildCount() > 0) {
        // This is a non-leaf result. Traverse its children, but do not add its
        // name to the output list. It should not contain any 'failure' or
        // 'error' tags, but we want to be lax here, because the syntax of the
        // test.xml file is also lax.
        for (TestCase child : testCase.getChildList()) {
          collectFailedTestCases(child);
        }
      } else {
        // This is a leaf result. If it passed, don't add it.
        if (testCase.getStatus() == TestCase.Status.PASSED) {
          return this;
        }

        String name = testCase.getName();
        String className = testCase.getClassName();
        if (name == null || className == null) {
          // A test case detail is not really interesting if we cannot tell which
          // one it is.
          this.summary.failedTestCasesStatus = FailedTestCasesStatus.PARTIAL;
          return this;
        }

        this.summary.failedTestCases.add(testCase);
      }
      return this;
    }

    public Builder addFailedTestCases(List<TestCase> testCases, FailedTestCasesStatus status) {
      checkMutation(status);
      checkMutation(testCases);

      if (summary.failedTestCasesStatus == null) {
        summary.failedTestCasesStatus = status;
      } else if (summary.failedTestCasesStatus != status) {
        summary.failedTestCasesStatus = FailedTestCasesStatus.PARTIAL;
      }

      if (testCases.isEmpty()) {
        return this;
      }

      // union of summary.failedTestCases, testCases
      Map<String, TestCase> allCases = new TreeMap<>();
      if (summary.failedTestCases != null) {
        for (TestCase detail : summary.failedTestCases) {
          allCases.put(detail.getClassName() + "." + detail.getName(), detail);
        }
      }
      for (TestCase detail : testCases) {
        allCases.put(detail.getClassName() + "." + detail.getName(), detail);
      }

      summary.failedTestCases = new ArrayList<>(allCases.values());
      return this;
    }

    public Builder addTestTimes(List<Long> testTimes) {
      checkMutation(testTimes);
      summary.testTimes.addAll(testTimes);
      return this;
    }

    public Builder addWarnings(List<String> warnings) {
      checkMutation(warnings);
      summary.warnings.addAll(warnings);
      return this;
    }

    public Builder setActionRan(boolean actionRan) {
      checkMutation();
      summary.actionRan = actionRan;
      return this;
    }

    /**
     * Set the number of results cached, locally or remotely.
     * 
     * @param numCached number of results cached locally or remotely
     * @return this Builder
     */
    public Builder setNumCached(int numCached) {
      checkMutation();
      summary.numCached = numCached;
      return this;
    }

    public Builder setNumLocalActionCached(int numLocalActionCached) {
      checkMutation();
      summary.numLocalActionCached = numLocalActionCached;
      return this;
    }

    public Builder setRanRemotely(boolean ranRemotely) {
      checkMutation();
      summary.ranRemotely = ranRemotely;
      return this;
    }

    public Builder setWasUnreportedWrongSize(boolean wasUnreportedWrongSize) {
      checkMutation();
      summary.wasUnreportedWrongSize = wasUnreportedWrongSize;
      return this;
    }

    /**
     * Records a new result for the given shard of the test.
     *
     * @return an immutable view of the statuses associated with the shard, with the new element.
     */
    public List<BlazeTestStatus> addShardStatus(int shardNumber, BlazeTestStatus status) {
      Preconditions.checkState(summary.shardRunStatuses.put(shardNumber, status),
          "shardRunStatuses must allow duplicate statuses");
      return ImmutableList.copyOf(summary.shardRunStatuses.get(shardNumber));
    }

    /**
     * Returns the created TestSummary object.
     * Any actions following a build() will create another copy of the same values.
     * Since no mutators are provided directly by TestSummary, a copy will not
     * be produced if two builds are invoked in a row without calling a setter.
     */
    public TestSummary build() {
      peek();
      if (!built) {
        makeSummaryImmutable();
        // else: it is already immutable.
      }
      Preconditions.checkState(built, "Built flag was not set");
      return summary;
    }

    /**
     * Within-package, it is possible to read directly from an
     * incompletely-built TestSummary. Used to pass Builders around directly.
     */
    TestSummary peek() {
      Preconditions.checkNotNull(summary.target, "Target cannot be null");
      Preconditions.checkNotNull(summary.status, "Status cannot be null");
      return summary;
    }

    private void makeSummaryImmutable() {
      // Once finalized, the list types are immutable.
      summary.passedLogs = Collections.unmodifiableList(summary.passedLogs);
      summary.failedLogs = Collections.unmodifiableList(summary.failedLogs);
      summary.warnings = Collections.unmodifiableList(summary.warnings);
      summary.coverageFiles = Collections.unmodifiableList(summary.coverageFiles);
      summary.testTimes = Collections.unmodifiableList(summary.testTimes);

      built = true;
    }
  }

  private ConfiguredTarget target;
  private BlazeTestStatus status;
  // Currently only populated if --runs_per_test_detects_flakes is enabled.
  private Multimap<Integer, BlazeTestStatus> shardRunStatuses = ArrayListMultimap.create();
  private int numCached;
  private int numLocalActionCached;
  private boolean actionRan;
  private boolean ranRemotely;
  private boolean wasUnreportedWrongSize;
  private List<TestCase> failedTestCases = new ArrayList<>();
  private List<Path> passedLogs = new ArrayList<>();
  private List<Path> failedLogs = new ArrayList<>();
  private List<String> warnings = new ArrayList<>();
  private List<Path> coverageFiles = new ArrayList<>();
  private List<Long> testTimes = new ArrayList<>();
  private FailedTestCasesStatus failedTestCasesStatus = null;

  // Don't allow public instantiation; go through the Builder.
  private TestSummary() {
  }

  /**
   * Creates a new Builder allowing construction of a new TestSummary object.
   */
  public static Builder newBuilder() {
    return new Builder();
  }

  /**
   * Creates a new Builder initialized with a copy of the existing object's values.
   */
  public static Builder newBuilderFromExisting(TestSummary existing) {
    Builder builder = new Builder();
    builder.mergeFrom(existing);
    return builder;
  }

  public ConfiguredTarget getTarget() {
    return target;
  }

  public BlazeTestStatus getStatus() {
    return status;
  }

  /**
   * Whether or not any results associated with this test were cached locally
   * or remotely.
   * 
   * @return true if any results were cached, false if not
   */
  public boolean isCached() {
    return numCached > 0;
  }

  public boolean isLocalActionCached() {
    return numLocalActionCached > 0;
  }

  public int numLocalActionCached() {
    return numLocalActionCached;
  }

  /**
   * @return number of results that were cached locally or remotely
   */
  public int numCached() {
    return numCached;
  }

  private int numUncached() {
    return totalRuns() - numCached;
  }

  /**
   * Whether or not any action was taken for this test, that is there was some
   * result that was <em>not cached</em>.
   * 
   * @return true if some action was taken for this test, false if not
   */
  public boolean actionRan() {
    return actionRan;
  }

  public boolean ranRemotely() {
    return ranRemotely;
  }

  public boolean wasUnreportedWrongSize() {
    return wasUnreportedWrongSize;
  }

  public List<TestCase> getFailedTestCases() {
    return failedTestCases;
  }

  public List<Path> getCoverageFiles() {
    return coverageFiles;
  }

  public List<Path> getPassedLogs() {
    return passedLogs;
  }

  public List<Path> getFailedLogs() {
    return failedLogs;
  }

  public FailedTestCasesStatus getFailedTestCasesStatus() {
    return failedTestCasesStatus;
  }

  /**
   * Returns an immutable view of the warnings associated with this test.
   */
  public List<String> getWarnings() {
    return Collections.unmodifiableList(warnings);
  }

  private static int getSortKey(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED ? -1 : status.ordinal();
  }

  @Override
  public int compareTo(TestSummary that) {
    if (this.isCached() != that.isCached()) {
      return this.isCached() ? -1 : 1;
    } else if ((this.isCached() && that.isCached()) && (this.numUncached() != that.numUncached())) {
      return this.numUncached() - that.numUncached();
    } else if (this.status != that.status) {
      return getSortKey(this.status) - getSortKey(that.status);
    } else {
      Artifact thisExecutable = this.target.getProvider(FilesToRunProvider.class).getExecutable();
      Artifact thatExecutable = that.target.getProvider(FilesToRunProvider.class).getExecutable();
      return thisExecutable.getPath().compareTo(thatExecutable.getPath());
    }
  }

  public List<Long> getTestTimes() {
    // The return result is unmodifiable (UnmodifiableList instance)
    return testTimes;
  }

  public int getNumCached() {
    return numCached;
  }

  public int totalRuns() {
    return testTimes.size();
  }

  static Mode getStatusMode(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED
        ? Mode.INFO
        : (status == BlazeTestStatus.FLAKY ? Mode.WARNING : Mode.ERROR);
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.testSummary(target.getTarget().getLabel());
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto() {
    BuildEventStreamProtos.TestSummary.Builder summaryBuilder =
        BuildEventStreamProtos.TestSummary.newBuilder().setTotalRunCount(totalRuns());
    for (Path path : getFailedLogs()) {
      summaryBuilder.addFailed(
          BuildEventStreamProtos.File.newBuilder().setUri(path.toString()).build());
    }
    for (Path path : getPassedLogs()) {
      summaryBuilder.addPassed(
          BuildEventStreamProtos.File.newBuilder().setUri(path.toString()).build());
    }
    return GenericBuildEvent.protoChaining(this).setTestSummary(summaryBuilder.build()).build();
  }
}
