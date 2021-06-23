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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider.TestParams;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter.Mode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.FailedTestCasesStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.protobuf.util.Durations;
import com.google.protobuf.util.Timestamps;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Test summary entry. Stores summary information for a single test rule. Also used to sort summary
 * output by status.
 *
 * <p>Invariant: All TestSummary mutations should be performed through the Builder. No direct
 * TestSummary methods (except the constructor) may mutate the object.
 */
public class TestSummary implements Comparable<TestSummary>, BuildEventWithOrderConstraint {
  /**
   * Builder class responsible for creating and altering TestSummary objects.
   */
  public static class Builder {
    private TestSummary summary;
    private boolean built;

    private Builder(ConfiguredTarget target) {
      summary = new TestSummary(target);
      built = false;
    }

    void mergeFrom(TestSummary existingSummary) {
      // Yuck, manually fill in fields.
      for (int i = 0; i < existingSummary.shardRunStatuses.size(); i++) {
        summary.shardRunStatuses.get(i).addAll(existingSummary.shardRunStatuses.get(i));
      }
      summary.firstStartTimeMillis = existingSummary.firstStartTimeMillis;
      summary.lastStopTimeMillis = existingSummary.lastStopTimeMillis;
      summary.totalRunDurationMillis = existingSummary.totalRunDurationMillis;
      setConfiguration(existingSummary.configuration);
      setStatus(existingSummary.status);
      addCoverageFiles(existingSummary.coverageFiles);
      addPassedLogs(existingSummary.passedLogs);
      addFailedLogs(existingSummary.failedLogs);
      summary.totalTestCases += existingSummary.totalTestCases;
      summary.totalUnknownTestCases += existingSummary.totalUnknownTestCases;

      if (existingSummary.failedTestCasesStatus != null) {
        addFailedTestCases(
            existingSummary.getFailedTestCases(), existingSummary.getFailedTestCasesStatus());
      }

      addTestTimes(existingSummary.testTimes);
      addWarnings(existingSummary.warnings);
      setActionRan(existingSummary.actionRan);
      setNumCached(existingSummary.numCached);
      setRanRemotely(existingSummary.ranRemotely);
      setWasUnreportedWrongSize(existingSummary.wasUnreportedWrongSize);
      mergeSystemFailure(existingSummary.getSystemFailure());
    }

    // Implements copy on write logic, allowing reuse of the same builder.
    private void checkMutation() {
      // If mutating the builder after an object was built, create another copy.
      if (built) {
        built = false;
        TestSummary lastSummary = summary;
        summary = new TestSummary(lastSummary.target);
        mergeFrom(lastSummary);
      }
    }

    // This used to return a reference to the value on success.
    // However, since it can alter the summary member, inlining it in an
    // assignment to a property of summary was unsafe.
    private void checkMutation(Object value) {
      checkNotNull(value);
      checkMutation();
    }

    public Builder setConfiguration(BuildConfiguration configuration) {
      checkMutation(configuration);
      summary.configuration = checkNotNull(configuration, summary);
      return this;
    }

    public Builder setStatus(BlazeTestStatus status) {
      checkMutation(status);
      summary.status = status;
      return this;
    }

    public Builder setSkipped(boolean skipped) {
      checkMutation(skipped);
      summary.skipped = skipped;
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

    public Builder addPassedLog(Path passedLog) {
      checkMutation(passedLog);
      summary.passedLogs.add(passedLog);
      return this;
    }

    public Builder addFailedLogs(List<Path> failedLogs) {
      checkMutation(failedLogs);
      summary.failedLogs.addAll(failedLogs);
      return this;
    }

    public Builder addFailedLog(Path failedLog) {
      checkMutation(failedLog);
      summary.failedLogs.add(failedLog);
      return this;
    }

    public Builder collectTestCases(@Nullable TestCase testCase) {
      // Maintain the invariant: failedTestCases + totalUnknownTestCases <= totalTestCases
      if (testCase == null) {
        // If we don't have test case information, count each test as one case with unknown status.
        summary.failedTestCasesStatus = FailedTestCasesStatus.NOT_AVAILABLE;
        summary.totalTestCases++;
        summary.totalUnknownTestCases++;
      } else {
        summary.failedTestCasesStatus = FailedTestCasesStatus.FULL;
        summary.totalTestCases += traverseTestCases(testCase);
      }
      return this;
    }

    private int traverseTestCases(TestCase testCase) {
      if (testCase.getChildCount() > 0) {
        // This is a non-leaf result. Traverse its children, but do not add its
        // name to the output list. It should not contain any 'failure' or
        // 'error' tags, but we want to be lax here, because the syntax of the
        // test.xml file is also lax.
        // don't count container of test cases as test
        int res = 0;
        for (TestCase child : testCase.getChildList()) {
          res += traverseTestCases(child);
        }
        return res;
      } else if (testCase.getType() != TestCase.Type.TEST_CASE) {
        return 0;
      }

      // This is a leaf result.
      if (!testCase.getRun()) {
        // Don't count test cases that were not run.
        return 0;
      }
      if (testCase.getStatus() != TestCase.Status.PASSED) {
        this.summary.failedTestCases.add(testCase);
      }
      return 1;
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

    public Builder mergeTiming(long startTimeMillis, long runDurationMillis) {
      checkMutation();
      summary.firstStartTimeMillis = Math.min(summary.firstStartTimeMillis, startTimeMillis);
      summary.lastStopTimeMillis =
          Math.max(summary.lastStopTimeMillis, startTimeMillis + runDurationMillis);
      summary.totalRunDurationMillis += runDurationMillis;
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

    public Builder mergeSystemFailure(@Nullable DetailedExitCode systemFailure) {
      checkMutation();
      summary.systemFailure =
          DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
              summary.systemFailure, systemFailure);
      return this;
    }

    /**
     * Records a new result for the given shard of the test.
     *
     * @return an immutable view of the statuses associated with the shard, with the new element.
     */
    public ImmutableList<BlazeTestStatus> addShardStatus(int shardNumber, BlazeTestStatus status) {
      List<BlazeTestStatus> statuses = summary.shardRunStatuses.get(shardNumber);
      statuses.add(status);
      return ImmutableList.copyOf(statuses);
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
      checkNotNull(summary.target, "Target cannot be null");
      checkNotNull(summary.status, "Status cannot be null");
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

  private final ConfiguredTarget target;
  // Currently only populated if --runs_per_test_detects_flakes is enabled.
  private final ImmutableList<ArrayList<BlazeTestStatus>> shardRunStatuses;

  private BuildConfiguration configuration;
  private BlazeTestStatus status;
  private boolean skipped;
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
  private long totalRunDurationMillis;
  private long firstStartTimeMillis = Long.MAX_VALUE;
  private long lastStopTimeMillis = Long.MIN_VALUE;
  private FailedTestCasesStatus failedTestCasesStatus = null;
  private int totalTestCases;
  private int totalUnknownTestCases;
  @Nullable private DetailedExitCode systemFailure;

  // Don't allow public instantiation; go through the Builder.
  private TestSummary(ConfiguredTarget target) {
    this.target = target;
    TestParams testParams = getTestParams();
    shardRunStatuses =
        createAndInitialize(
            testParams.runsDetectsFlakes() ? Math.max(testParams.getShards(), 1) : 0);
  }

  private static ImmutableList<ArrayList<BlazeTestStatus>> createAndInitialize(int sz) {
    return Stream.generate(() -> new ArrayList<BlazeTestStatus>(1))
        .limit(sz)
        .collect(toImmutableList());
  }

  /** Creates a new Builder allowing construction of a new TestSummary object. */
  public static Builder newBuilder(ConfiguredTarget target) {
    return new Builder(target);
  }

  public Label getLabel() {
    return AliasProvider.getDependencyLabel(target);
  }

  public ConfiguredTarget getTarget() {
    return target;
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public BlazeTestStatus getStatus() {
    return status;
  }

  public boolean isSkipped() {
    return skipped;
  }

  /**
   * Whether or not any results associated with this test were cached locally or remotely.
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
   * Whether or not any action was taken for this test, that is there was some result that was
   * <em>not cached</em>.
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

  public int getTotalTestCases() {
    return totalTestCases;
  }

  public int getUnkownTestCases() {
    return totalUnknownTestCases;
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

  @Nullable
  public DetailedExitCode getSystemFailure() {
    return systemFailure;
  }

  /**
   * Returns an immutable view of the warnings associated with this test.
   */
  public List<String> getWarnings() {
    return Collections.unmodifiableList(warnings);
  }

  private static int getSortKey(BlazeTestStatus status) {
    return status == BlazeTestStatus.PASSED ? -1 : status.getNumber();
  }

  @Override
  public int compareTo(TestSummary that) {
    return ComparisonChain.start()
        .compareTrueFirst(this.isCached(), that.isCached())
        .compare(this.numUncached(), that.numUncached())
        .compare(getSortKey(this.status), getSortKey(that.status))
        .compare(this.getLabel(), that.getLabel())
        .compare(
            this.getTarget().getConfigurationChecksum(),
            that.getTarget().getConfigurationChecksum())
        .compare(this.getTotalTestCases(), that.getTotalTestCases())
        .result();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("target", this.getTarget())
        .add("status", status)
        .add("numCached", numCached)
        .add("numLocalActionCached", numLocalActionCached)
        .add("actionRan", actionRan)
        .add("ranRemotely", ranRemotely)
        .toString();
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

  public long getTotalRunDurationMillis() {
    return totalRunDurationMillis;
  }

  public long getFirstStartTimeMillis() {
    return firstStartTimeMillis;
  }

  public long getLastStopTimeMillis() {
    return lastStopTimeMillis;
  }

  Mode getStatusMode() {
    if (skipped) {
      return Mode.WARNING;
    }
    return status == BlazeTestStatus.PASSED
        ? Mode.INFO
        : (status == BlazeTestStatus.FLAKY ? Mode.WARNING : Mode.ERROR);
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.testSummary(
        AliasProvider.getDependencyLabel(target),
        BuildEventIdUtil.configurationId(target.getConfigurationChecksum()));
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(
        BuildEventIdUtil.targetCompleted(
            AliasProvider.getDependencyLabel(target),
            BuildEventIdUtil.configurationId(target.getConfigurationChecksum())));
  }

  @Override
  public ImmutableList<LocalFile> referencedLocalFiles() {
    ImmutableList.Builder<LocalFile> localFiles = ImmutableList.builder();
    for (Path path : getFailedLogs()) {
      localFiles.add(new LocalFile(path, LocalFileType.FAILED_TEST_OUTPUT));
    }
    for (Path path : getPassedLogs()) {
      localFiles.add(new LocalFile(path, LocalFileType.SUCCESSFUL_TEST_OUTPUT));
    }
    return localFiles.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    PathConverter pathConverter = converters.pathConverter();
    TestParams testParams = getTestParams();
    BuildEventStreamProtos.TestSummary.Builder summaryBuilder =
        BuildEventStreamProtos.TestSummary.newBuilder()
            .setOverallStatus(BuildEventStreamerUtils.bepStatus(status))
            .setTotalNumCached(getNumCached())
            .setTotalRunCount(totalRuns())
            .setRunCount(testParams.getRuns())
            .setShardCount(testParams.getShards())
            .setFirstStartTime(Timestamps.fromMillis(firstStartTimeMillis))
            .setFirstStartTimeMillis(firstStartTimeMillis)
            .setLastStopTime(Timestamps.fromMillis(lastStopTimeMillis))
            .setLastStopTimeMillis(lastStopTimeMillis)
            .setTotalRunDuration(Durations.fromMillis(totalRunDurationMillis))
            .setTotalRunDurationMillis(totalRunDurationMillis);
    for (Path path : getFailedLogs()) {
      String uri = pathConverter.apply(path);
      if (uri != null) {
        summaryBuilder.addFailed(BuildEventStreamProtos.File.newBuilder().setUri(uri).build());
      }
    }
    for (Path path : getPassedLogs()) {
      String uri = pathConverter.apply(path);
      if (uri != null) {
        summaryBuilder.addPassed(BuildEventStreamProtos.File.newBuilder().setUri(uri).build());
      }
    }
    return GenericBuildEvent.protoChaining(this).setTestSummary(summaryBuilder.build()).build();
  }

  private TestParams getTestParams() {
    return checkNotNull(target.getProvider(TestProvider.class).getTestParams(), target);
  }
}
