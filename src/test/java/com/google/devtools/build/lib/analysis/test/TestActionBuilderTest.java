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
package com.google.devtools.build.lib.analysis.test;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.TestTimeout;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link com.google.devtools.build.lib.analysis.test.TestActionBuilder} tests. */
@RunWith(JUnit4.class)
public class TestActionBuilderTest extends BuildViewTestCase {

  @Before
  public final void createBuildFile() throws Exception {
    analysisMock.pySupport().setup(mockToolsConfig);

    scratch.file(
        "tests/BUILD",
        "py_test(name = 'small_test_1',",
        "        srcs = ['small_test_1.py'],",
        "        data = [':xUnit'],",
        "        size = 'small',",
        "        tags = ['tag1'])",
        "",
        "sh_test(name = 'small_test_2',",
        "        srcs = ['small_test_2.sh'],",
        "        data = ['//testing/shbase:googletest.sh'],",
        "        size = 'small',",
        "        tags = ['tag2'])",
        "",
        "sh_test(name = 'large_test_1',",
        "        srcs = ['large_test_1.sh'],",
        "        data = ['//testing/shbase:googletest.sh', ':xUnit'],",
        "        size = 'large',",
        "        tags = ['tag1'])",
        "",
        "py_binary(name = 'notest',",
        "        srcs = ['notest.py'])",
        "cc_library(name = 'xUnit')",
        "",
        "test_suite(name = 'smallTests', tags=['small'])");
  }

  @Test
  public void testFlakyAttributeValidation() throws Exception {
    scratch.file("flaky/BUILD",
        "sh_test(name = 'good_test',",
        "        srcs = ['a.sh'])",
        "sh_test(name = 'flaky_test',",
        "        srcs = ['a.sh'],",
        "        flaky = 1)");
    Artifact testStatus = Iterables.getOnlyElement(getTestStatusArtifacts("//flaky:good_test"));
    assertThat(testStatus).isNotNull();
    TestRunnerAction action = (TestRunnerAction) getGeneratingAction(testStatus);
    assertThat(action.getTestProperties().isFlaky()).isFalse();

    testStatus = Iterables.getOnlyElement(getTestStatusArtifacts("//flaky:flaky_test"));
    assertThat(testStatus).isNotNull();
    action = (TestRunnerAction) getGeneratingAction(testStatus);
    assertThat(action.getTestProperties().isFlaky()).isTrue();
  }

  @Test
  public void testIllegalBooleanFlakySetting() throws Exception {
    checkError("flaky", "bad_test",
        "boolean is not one of [0, 1]",
        "sh_test(name = 'bad_test',",
        "        srcs = ['a.sh'],",
        "        flaky = 2)");
  }

  @Test
  public void testRunsPerTest() throws Exception {
    useConfiguration("--runs_per_test=2");
    ImmutableList<Artifact.DerivedArtifact> testStatusList =
        getTestStatusArtifacts("//tests:small_test_1");
    assertThat(testStatusList).hasSize(2);
    Artifact testStatus1 = testStatusList.get(0);
    Artifact testStatus2 = testStatusList.get(1);
    assertThat(testStatus1).isNotNull();
    assertThat(testStatus2).isNotNull();
    assertThat(testStatus2).isNotSameInstanceAs(testStatus1);
    assertThat(getGeneratingAction(testStatus2))
        .isNotSameInstanceAs(getGeneratingAction(testStatus1));
    assertThat(testStatus1.getRootRelativePath().getPathString())
        .contains("tests/small_test_1/run_1_of_2/test");
    assertThat(testStatus2.getRootRelativePath().getPathString())
        .contains("tests/small_test_1/run_2_of_2/test");
  }

  @Test
  public void testRunsPerTestCanBeOverridden() throws Exception {
    useConfiguration("--runs_per_test=1", "--runs_per_test=2");
    ImmutableList<Artifact.DerivedArtifact> testStatusList =
        getTestStatusArtifacts("//tests:small_test_1");
    assertThat(testStatusList).hasSize(2);
    Artifact testStatus1 = testStatusList.get(0);
    Artifact testStatus2 = testStatusList.get(1);
    assertThat(testStatus1).isNotNull();
    assertThat(testStatus2).isNotNull();
    assertThat(testStatus2).isNotSameInstanceAs(testStatus1);
    assertThat(getGeneratingAction(testStatus2))
        .isNotSameInstanceAs(getGeneratingAction(testStatus1));
    assertThat(testStatus1.getRootRelativePath().getPathString())
        .contains("tests/small_test_1/run_1_of_2/test");
    assertThat(testStatus2.getRootRelativePath().getPathString())
        .contains("tests/small_test_1/run_2_of_2/test");
  }

  /**
   * Test that test rules always construct with a standard timeout, either
   * inferred from size or explicitly set by attribute.
   */
  @Test
  public void testTestTimeoutFlagOverridesTimeoutDefaultsValues() throws Exception {
    scratch.file("javatests/timeouts/BUILD",
        "java_test(name = 'small_no_timeout',",
        "          srcs = [],",
        "          size = 'small')",
        "java_test(name = 'small_with_timeout',",
        "          srcs = [],",
        "          size = 'small',",
        "          timeout = 'long')");
    ImmutableList<Artifact.DerivedArtifact> testStatusList =
        getTestStatusArtifacts("//javatests/timeouts:small_no_timeout");
    TestRunnerAction testAction = (TestRunnerAction)
        getGeneratingAction(Iterables.get(testStatusList, 0));
    Integer timeout = testAction.getTestProperties().getTimeout().getTimeoutSeconds();
    assertThat(timeout).isEqualTo(TestTimeout.SHORT.getTimeoutSeconds());

    testStatusList = getTestStatusArtifacts("//javatests/timeouts:small_with_timeout");
    testAction = (TestRunnerAction) getGeneratingAction(Iterables.get(testStatusList, 0));
    timeout = testAction.getTestProperties().getTimeout().getTimeoutSeconds();
    assertThat(timeout).isEqualTo(TestTimeout.LONG.getTimeoutSeconds());
  }

  @Test
  public void testRunsPerTestWithSharding() throws Exception {
    useConfiguration("--runs_per_test=2");
    scratch.file(
        "javatests/jt/BUILD",
        "java_test(name = 'RGT',",
        "          shard_count = 10,",
        "          srcs = ['RGT.java'])");
    ImmutableList<Artifact.DerivedArtifact> testStatusList =
        getTestStatusArtifacts("//javatests/jt:RGT");
    assertThat(testStatusList).hasSize(20);
    Artifact testStatus1 = testStatusList.get(0);
    Artifact testStatus10 = testStatusList.get(9);
    Artifact testStatus11 = testStatusList.get(10);
    assertThat(testStatus1).isNotNull();
    assertThat(testStatus10).isNotNull();
    assertThat(testStatus11).isNotNull();
    assertThat(testStatus1.getRootRelativePath().getPathString())
        .contains("javatests/jt/RGT/shard_1_of_10_run_1_of_2/test");
    assertThat(testStatus10.getRootRelativePath().getPathString())
        .contains("javatests/jt/RGT/shard_5_of_10_run_2_of_2/test");
    assertThat(testStatus11.getRootRelativePath().getPathString())
        .contains("javatests/jt/RGT/shard_6_of_10_run_1_of_2/test");
  }

  @Test
  public void testAspectOverNonExpandingTestSuitesVisitsImplicitTests() throws Exception {
    scratch.file(
        "BUILD",
        "sh_test(name = 'test_a',",
        "        srcs = [':a.sh'])",
        "",
        "sh_test(name = 'test_b',",
        "        srcs = [':b.sh'])",
        "",
        "test_suite(name = 'suite'",
        ")");
    writeLabelCollectionAspect();

    useLoadingOptions("--noexpand_test_suites");
    AnalysisResult analysisResult =
        update(
            ImmutableList.of("//:suite"),
            ImmutableList.of("//:aspect.bzl%a"),
            /* keepGoing= */ false,
            /* loadingPhaseThreads= */ 1,
            /* doAnalysis= */ true,
            new EventBus());
    ConfiguredAspect aspectValue =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute(
                "//:aspect.bzl",
                /* defaultToMain= */ true,
                /* repositoryMapping= */ ImmutableMap.of()),
            "StructImpl");
    StructImpl info = (StructImpl) aspectValue.get(key);
    assertThat(((Depset) info.getValue("labels")).getSet(String.class).toList())
        .containsExactly("//:suite", "//:test_a", "//:test_b");
  }

  @Test
  public void testAspectOverNonExpandingTestSuitesVisitsExplicitTests() throws Exception {
    scratch.file(
        "BUILD",
        "sh_test(name = 'test_a',",
        "        srcs = [':a.sh'])",
        "",
        "sh_test(name = 'test_b',",
        "        srcs = [':b.sh'])",
        "",
        "test_suite(name = 'suite',",
        "           tests = [':test_b']",
        ")");
    writeLabelCollectionAspect();

    useLoadingOptions("--noexpand_test_suites");
    AnalysisResult analysisResult =
        update(
            ImmutableList.of("//:suite"),
            ImmutableList.of("//:aspect.bzl%a"),
            /* keepGoing= */ false,
            /* loadingPhaseThreads= */ 1,
            /* doAnalysis= */ true,
            new EventBus());
    ConfiguredAspect aspectValue =
        Iterables.getOnlyElement(analysisResult.getAspectsMap().values());
    StarlarkProvider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute(
                "//:aspect.bzl",
                /* defaultToMain= */ true,
                /* repositoryMapping= */ ImmutableMap.of()),
            "StructImpl");
    StructImpl info = (StructImpl) aspectValue.get(key);
    assertThat(((Depset) info.getValue("labels")).getSet(String.class).toList())
        .containsExactly("//:suite", "//:test_b");
  }

  @Test
  public void testAspectOverExpandingTestSuitesDoesNotVisitSuite() throws Exception {
    scratch.file(
        "BUILD",
        "sh_test(name = 'test_a',",
        "        srcs = [':a.sh'])",
        "",
        "sh_test(name = 'test_b',",
        "        srcs = [':b.sh'])",
        "",
        "test_suite(name = 'suite',",
        ")");
    writeLabelCollectionAspect();

    AnalysisResult analysisResult =
        update(
            ImmutableList.of("//:suite"),
            ImmutableList.of("//:aspect.bzl%a"),
            /* keepGoing= */ false,
            /* loadingPhaseThreads= */ 1,
            /* doAnalysis= */ true,
            new EventBus());
    final StarlarkProvider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute(
                "//:aspect.bzl",
                /* defaultToMain= */ true,
                /* repositoryMapping= */ ImmutableMap.of()),
            "StructImpl");

    List<String> labels = new ArrayList<>();
    for (ConfiguredAspect a : analysisResult.getAspectsMap().values()) {
      StructImpl info = (StructImpl) a.get(key);
      labels.addAll(((Depset) info.getValue("labels")).getSet(String.class).toList());
    }
    assertThat(labels).containsExactly("//:test_a", "//:test_b");
  }

  private void writeLabelCollectionAspect() throws IOException {
    scratch.file(
        "aspect.bzl",
        "StructImpl = provider(fields = ['labels'])",
        "def _impl(target,ctx):",
        "    print(target.label)",
        "    transitive = []",
        "    if hasattr(ctx.rule.attr, 'tests'):",
        "      transitive += [dep[StructImpl].labels for dep in ctx.rule.attr.tests]",
        "    if hasattr(ctx.rule.attr, '_implicit_tests'):",
        "      transitive += [dep[StructImpl].labels for dep in ctx.rule.attr._implicit_tests]",
        "    return [StructImpl(labels = depset([str(target.label)], transitive = transitive))]",
        "",
        "a = aspect(_impl, attr_aspects = ['tests', '_implicit_tests'])");
  }

  /**
   * Regression test for bug {@link "http://b/2644860"}.
   */
  @Test
  public void testIllegalTestSizeAttributeDoesNotCrashTestSuite() throws Exception {
    checkError("bad_size", "illegal_size_test",
        "In rule 'illegal_size_test', size 'bad' is not a valid size",
        "sh_test(name = 'illegal_size_test',",
        "        srcs = ['illegal.sh'],",
        "        size = 'bad')",
        "test_suite(name = 'everything')");
  }

  /**
   * Regression test for bug {@link "http://b/2644860"} but with an illegal Timeout.
   */
  @Test
  public void testIllegalTestTimeoutAttributeDoesNotCrashTestSuite() throws Exception {
    checkError("bad_timeout", "illegal_timeout_test",
        "In rule 'illegal_timeout_test', timeout 'unreasonable' is not a valid timeout",
        "sh_test(name = 'illegal_timeout_test',",
        "        srcs = ['illegal.sh'],",
        "        timeout = 'unreasonable')",
        "test_suite(name = 'everything')");
  }

  private ImmutableList<Artifact.DerivedArtifact> getTestStatusArtifacts(String label)
      throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    return target.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
  }

}
