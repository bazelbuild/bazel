// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.metrics;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ActionSummary.ActionData;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ArtifactMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.BuildGraphMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.CumulativeMetrics;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerMetricsCollector;
import java.util.List;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests metric collection. */
@RunWith(JUnit4.class)
public class MetricsCollectorTest extends BuildIntegrationTestCase {

  static class BuildMetricsEventListener extends BlazeModule {

    private BuildMetricsEvent event;

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void onBuildMetrics(BuildMetricsEvent event) {
      this.event = event;
    }
  }

  private BuildMetricsEventListener buildMetricsEventListener = new BuildMetricsEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new MetricsModule())
        .addBlazeModule(buildMetricsEventListener);
  }

  @Before
  public void writeTrivialFooTarget() throws Exception {
    write(
        "foo/BUILD",
        "genrule(",
        "    name = 'foo',",
        "    outs = ['dir'],",
        "    cmd = '/bin/mkdir $(location dir)',",
        "    srcs = [],",
        ")");
  }

  @Before
  public void setUpWorkerMetricsCollecto() {
    WorkerMetricsCollector.instance().setClock(new JavaClock());
  }

  @After
  public void resetProfilers() throws Exception {
    MemoryProfiler.instance().stop();
    PostGCMemoryUseRecorder.get().reset();
  }

  @Test
  public void testActionsCreated() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getActionSummary().getActionsCreated()).isGreaterThan(0L);
  }

  @Test
  public void testActionsCreatedIsZeroOnSecondBuild() throws Exception {
    buildTarget("//foo:foo");
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getActionSummary().getActionsCreated()).isEqualTo(0);
  }

  @Test
  public void testActionsExecuted() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getActionSummary().getActionsExecuted()).isGreaterThan(0L);
  }

  @Test
  public void testActionsExecutedIsZeroOnSecondBuild() throws Exception {
    buildTarget("//foo:foo");
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getActionSummary().getActionsExecuted()).isEqualTo(0);
  }

  @Test
  public void buildGraphAndArtifactMetrics() throws Exception {
    write(
        "a/BUILD",
        "genrule(name = 'a', srcs = ['//b:b', '//b:c'], outs = ['a.out'], cmd = 'cat $(SRCS) >"
            + " $@')");
    write(
        "b/BUILD",
        "genrule(name = 'b', srcs = ['b.in', 'c.in'], outs = ['b.out'], cmd = 'cat $(SRCS) > $@')",
        "genrule(name = 'c', srcs = ['c.in', 'c2.in'], outs = ['c.out'], cmd = 'cat $(SRCS) >"
            + " $@')");
    if (OS.getCurrent() == OS.WINDOWS) {
      // On Windows we have \r\n line endings while on other platforms only \n. So make the file one
      // byte shorter on Windows so that the byte counts below match.
      write("b/b.in", "1234");
      write("b/c.in", "1");
    } else {
      write("b/b.in", "12345");
      write("b/c.in", "12");
    }
    createSymlink("c.in", "b/c2.in");
    write(
        "e/BUILD",
        "alias(name = 'facade', actual = ':e.out')",
        "genrule(name = 'e', srcs = ['e.in'], outs = ['e.out'], cmd = 'cat $(SRCS) > $@')");
    if (OS.getCurrent() == OS.WINDOWS) {
      // On Windows we have \r\n line endings while on other platforms only \n. So make the file one
      // byte shorter on Windows so that the byte counts below match.
      write("e/e.in", "ab");
    } else {
      write("e/e.in", "abc");
    }

    // Do one build of a target in a standalone package. Gets us a baseline for analysis/execution.
    buildTarget("//e:facade");
    boolean skymeldWasInvolved =
        getCommandEnvironment().withMergedAnalysisAndExecutionSourceOfTruth();
    BuildGraphMetrics buildGraphMetrics =
        buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics();
    int actionLookupValueCount = buildGraphMetrics.getActionLookupValueCount();
    // All these numbers should be big, but just want a basic check.
    assertThat(actionLookupValueCount).isGreaterThan(0);
    assertThat(buildGraphMetrics.getActionLookupValueCountNotIncludingAspects())
        .isEqualTo(actionLookupValueCount);
    int actionCount = buildGraphMetrics.getActionCount();
    assertThat(actionCount).isGreaterThan(0);
    assertThat(buildGraphMetrics.getActionCountNotIncludingAspects()).isEqualTo(actionCount);
    assertThat(buildGraphMetrics)
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                .setOutputFileConfiguredTargetCount(1)
                .setOtherConfiguredTargetCount(1)
                .build());
    int outputArtifactCount = buildGraphMetrics.getOutputArtifactCount();
    assertThat(outputArtifactCount).isGreaterThan(0);
    int graphSize = buildGraphMetrics.getPostInvocationSkyframeNodeCount();
    assertThat(graphSize).isGreaterThan(0);
    ArtifactMetrics artifactMetrics =
        buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics();
    assertThat(artifactMetrics.getSourceArtifactsRead().getSizeInBytes()).isGreaterThan(0L);
    assertThat(artifactMetrics.getOutputArtifactsSeen())
        .isEqualTo(ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(4L).setCount(3).build());
    assertThat(artifactMetrics.getOutputArtifactsFromActionCache().getCount()).isEqualTo(0);
    assertThat(artifactMetrics.getTopLevelArtifacts())
        .isEqualTo(ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(4L).setCount(1).build());
    // Adjust for the "alias", "input" and "output" configured targets, which won't be in play
    // later.
    actionLookupValueCount -= 3;

    // Now do a build of a target with non-trivial transitive deps, and verify the metrics. Blaze
    // won't redo analysis of dependencies or re-read their sources.
    buildTarget("//a");

    buildGraphMetrics = buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics();
    assertThat(buildGraphMetrics)
        .comparingExpectedFieldsOnly()
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                // Two dependencies and three source files for action lookup values.
                .setActionLookupValueCount(5 + actionLookupValueCount)
                .setActionCount(2 + actionCount)
                .setInputFileConfiguredTargetCount(4)
                .setOutputArtifactCount(2 + outputArtifactCount)
                .build());

    int newGraphSize = buildGraphMetrics.getPostInvocationSkyframeNodeCount();
    assertThat(newGraphSize).isGreaterThan(graphSize);

    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            ArtifactMetrics.newBuilder()
                // 2 distinct artifacts of 6 and 3 bytes, with a symlink to the 3-byte one.
                .setSourceArtifactsRead(
                    ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(12).setCount(3))
                // b outputs 9 bytes, c outputs 6, a outputs 15, 30 total.
                .setOutputArtifactsSeen(
                    ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(30).setCount(3).build())
                .setTopLevelArtifacts(
                    ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(15).setCount(1).build())
                .build());

    // Do a null build. No useful analysis stats.
    buildTarget("//a");
    if (skymeldWasInvolved) {
      // The BuildDriverKey of //e:facade is gone.
      newGraphSize -= 1;
    }

    // For null build, we don't do any conflict checking. As the metrics are collected during the
    // traversal that's part of conflict checking, these analysis-related numbers are 0.
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                .setActionLookupValueCount(0)
                .setActionLookupValueCountNotIncludingAspects(0)
                .setActionCount(0)
                .setActionCountNotIncludingAspects(0)
                .setInputFileConfiguredTargetCount(0)
                .setOutputArtifactCount(0)
                .setPostInvocationSkyframeNodeCount(newGraphSize)
                .build());
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(ArtifactMetrics.getDefaultInstance());

    // Change a BUILD file and rebuild: no source artifacts read, but analysis stats present.
    write(
        "a/BUILD",
        "genrule(name = 'a', srcs = ['//b:c', '//b:b'], outs = ['a.out'], cmd = 'cat $(SRCS) >"
            + " $@')");
    buildTarget("//a");

    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                .setActionLookupValueCount(5 + actionLookupValueCount)
                .setActionLookupValueCountNotIncludingAspects(5 + actionLookupValueCount)
                .setActionCount(2 + actionCount)
                .setActionCountNotIncludingAspects(2 + actionCount)
                .setInputFileConfiguredTargetCount(4)
                .setOutputArtifactCount(2 + outputArtifactCount)
                // ArtifactNestedSet node for stale nested set is still in graph, since it is
                // technically still valid (even though nobody wants that nested set anymore).
                .setPostInvocationSkyframeNodeCount(newGraphSize + 1)
                .build());
    ArtifactMetrics.FilesMetric singleFileMetric =
        ArtifactMetrics.FilesMetric.newBuilder().setSizeInBytes(15L).setCount(1).build();
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            ArtifactMetrics.newBuilder()
                .setOutputArtifactsSeen(singleFileMetric)
                .setTopLevelArtifacts(singleFileMetric)
                .build());

    // Change BUILD file back, but don't do a full build.
    write(
        "a/BUILD",
        "genrule(name = 'a', srcs = ['//b:c', '//b:b'], outs = ['a.out'], cmd = 'cat $(SRCS) >"
            + " $@')");
    addOptions("--nobuild");
    buildTarget("//a");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                .setActionLookupValueCount(5 + actionLookupValueCount)
                .setActionLookupValueCountNotIncludingAspects(5 + actionLookupValueCount)
                .setActionCount(2 + actionCount)
                .setActionCountNotIncludingAspects(2 + actionCount)
                .setInputFileConfiguredTargetCount(4)
                .setOutputArtifactCount(2 + outputArtifactCount)
                .setPostInvocationSkyframeNodeCount(newGraphSize + 1)
                .build());
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(ArtifactMetrics.getDefaultInstance());

    // Null --nobuild.
    buildTarget("//a");
    if (skymeldWasInvolved) {
      // When doing --nobuild, no new BuildDriverKey entry is put in the graph while the old one is
      // deleted.
      newGraphSize -= 1;
    }
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                // Stale action execution nodes have been GC'ed.
                .setPostInvocationSkyframeNodeCount(newGraphSize - 1)
                .build());

    // Do a null full build. Back to baseline.
    addOptions("--build");
    buildTarget("//a");
    if (skymeldWasInvolved) {
      // Extra BuildDriverKey
      newGraphSize += 1;
    }
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                // We now have three copies of the ArtifactNestedSetKey, since the re-analysis
                // didn't re-use the old nested set.
                .setPostInvocationSkyframeNodeCount(newGraphSize + 2)
                .build());
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            ArtifactMetrics.newBuilder()
                .setOutputArtifactsSeen(singleFileMetric)
                .setOutputArtifactsFromActionCache(singleFileMetric)
                .setTopLevelArtifacts(singleFileMetric)
                .build());

    // Change a source file. It and its symlink are both re-read.
    write("b/c.in", "1234");
    buildTarget("//a");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            BuildGraphMetrics.newBuilder()
                // Analysis not re-triggered, even of the input file that was changed.
                .setInputFileConfiguredTargetCount(0)
                .setPostInvocationSkyframeNodeCount(newGraphSize + 2)
                .build());
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            ArtifactMetrics.newBuilder()
                .setSourceArtifactsRead(
                    ArtifactMetrics.FilesMetric.newBuilder()
                        .setSizeInBytes(OS.getCurrent() == OS.WINDOWS ? 6 : 10)
                        .setCount(OS.getCurrent() == OS.WINDOWS ? 1 : 2))
                .setOutputArtifactsSeen(
                    ArtifactMetrics.FilesMetric.newBuilder()
                        .setSizeInBytes(42L)
                        .setCount(3)
                        .build())
                .setTopLevelArtifacts(
                    ArtifactMetrics.FilesMetric.newBuilder()
                        .setSizeInBytes(21L)
                        .setCount(1)
                        .build())
                .build());
  }

  @Test
  public void treeArtifactAndTopLevelMetrics() throws Exception {
    write(
        "foo/tree_artifact_rule.bzl",
        "def _tree_artifact_files_impl(ctx):",
        "    directory = ctx.actions.declare_directory(ctx.attr.name + '_artifact')",
        "    ctx.actions.run_shell(",
        "      outputs = [directory],",
        "      command = 'cd %s && echo a > file1 && echo bcde > file2}' % (directory.path))",
        "    return [DefaultInfo(files = depset([directory]))]",
        "def _several_outputs_impl(ctx):",
        "    file = ctx.actions.declare_file(ctx.attr.name + '_file')",
        "    ctx.actions.write(output = file, content = 'abc')",
        "    return [DefaultInfo(files = depset([file])),",
        "            OutputGroupInfo(dep_files = ctx.attr.dep[DefaultInfo].files)]",
        "my_tree = rule(implementation = _tree_artifact_files_impl)",
        "my_rule = rule(",
        "               implementation = _several_outputs_impl,",
        "               attrs = { 'dep': attr.label()},",
        " )");
    write(
        "foo/BUILD",
        "load('//foo:tree_artifact_rule.bzl', 'my_rule', 'my_tree')",
        "my_tree(name = 'tree')",
        "my_rule(name = 'top', dep = ':tree')");
    // Null build to populate silly things like fake build-info artifact.
    buildTarget();
    addOptions("--output_groups=+dep_files");
    buildTarget("//foo:top");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getArtifactMetrics())
        .ignoringFieldAbsence()
        .isEqualTo(
            ArtifactMetrics.newBuilder()
                .setOutputArtifactsSeen(
                    ArtifactMetrics.FilesMetric.newBuilder()
                        .setSizeInBytes(10L)
                        .setCount(3)
                        .build())
                .setTopLevelArtifacts(
                    ArtifactMetrics.FilesMetric.newBuilder()
                        .setSizeInBytes(10L)
                        .setCount(3)
                        .build())
                .build());
  }

  @Test
  public void testTargetCounts() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTargetMetrics().getTargetsConfigured()).isGreaterThan(0L);
    assertThat(buildMetrics.getTargetMetrics().getTargetsConfiguredNotIncludingAspects())
        .isGreaterThan(0L);
  }

  @Test
  public void testTargetsCountsAreZeroOnSecondBuild() throws Exception {
    buildTarget("//foo:foo");
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTargetMetrics().getTargetsLoaded()).isEqualTo(0);
    assertThat(buildMetrics.getTargetMetrics().getTargetsConfigured()).isEqualTo(0);
  }

  @Test
  public void aspectLoadedMetric() throws Exception {
    write(
        "foo/foo.bzl",
        "def _aspect_impl(target, ctx):",
        "  outfile = ctx.actions.declare_file(ctx.rule.attr.name + 'aspect.out')",
        "  ctx.actions.run_shell(",
        "    outputs = [outfile],",
        "    command = 'echo \"1\" > ' + outfile.path,",
        "  )",
        "  return [OutputGroupInfo(files = [outfile])]",
        "",
        "def _impl(ctx):",
        "    return []",
        "",
        "rule_aspect = aspect(implementation = _aspect_impl, attr_aspects = ['deps'])",
        "",
        "aspected = rule(",
        "    implementation = _impl,",
        "    attrs = { 'deps': attr.label_list(aspects = [rule_aspect]) })");
    write(
        "foo/BUILD",
        "load('//foo:foo.bzl', 'aspected')",
        "aspected(name = 'top', deps = [':dep'])",
        "aspected(name = 'dep')");
    buildTarget("//foo:top");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    // 2 additional: aspect and workspace status action.
    assertThat(buildMetrics.getTargetMetrics().getTargetsConfigured())
        .isEqualTo(2L + buildMetrics.getTargetMetrics().getTargetsConfiguredNotIncludingAspects());
    assertThat(buildMetrics.getActionSummary().getActionsCreated())
        .isEqualTo(2L + buildMetrics.getActionSummary().getActionsCreatedNotIncludingAspects());
    // Traversing the Skyframe graph doesn't hit the workspace status action.
    assertThat(buildMetrics.getBuildGraphMetrics().getActionLookupValueCount())
        .isEqualTo(
            1L
                + buildMetrics
                    .getBuildGraphMetrics()
                    .getActionLookupValueCountNotIncludingAspects());
    assertThat(buildMetrics.getBuildGraphMetrics().getActionCount())
        .isEqualTo(1L + buildMetrics.getBuildGraphMetrics().getActionCountNotIncludingAspects());

    // Analyzing a new target makes the aspect drop out of the target metric, but the build graph
    // metric still knows about it.
    write("bar/BUILD", "genrule(name = 'bar', outs = ['out'], cmd = 'touch $@')");
    buildTarget("//foo:top", "//bar:bar");
    buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTargetMetrics().getTargetsConfigured())
        .isEqualTo(buildMetrics.getTargetMetrics().getTargetsConfiguredNotIncludingAspects());
    assertThat(buildMetrics.getActionSummary().getActionsCreated())
        .isEqualTo(buildMetrics.getActionSummary().getActionsCreatedNotIncludingAspects());
    assertThat(buildMetrics.getBuildGraphMetrics().getActionLookupValueCount())
        .isEqualTo(
            1L
                + buildMetrics
                    .getBuildGraphMetrics()
                    .getActionLookupValueCountNotIncludingAspects());
    assertThat(buildMetrics.getBuildGraphMetrics().getActionCount())
        .isEqualTo(1L + buildMetrics.getBuildGraphMetrics().getActionCountNotIncludingAspects());
  }

  @Test
  public void testPackagesLoaded() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getPackageMetrics().getPackagesLoaded()).isGreaterThan(0L);
  }

  @Test
  public void testPackagesLoadedIsZeroOnSecondBuild() throws Exception {
    buildTarget("//foo:foo");
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getPackageMetrics().getPackagesLoaded()).isEqualTo(0);
  }

  @Test
  public void testAnalysisTimeInMs() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTimingMetrics().getAnalysisPhaseTimeInMs()).isGreaterThan(0);
  }

  @Test
  public void testExecutionTimeInMs() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTimingMetrics().getExecutionPhaseTimeInMs()).isGreaterThan(0);
  }

  @Test
  public void testUsedHeapSizePostBuild() throws Exception {
    // TODO(bazel-team): Fix recording used heap size on Windows.
    Assume.assumeTrue(OS.getCurrent() != OS.WINDOWS);
    addOptions("--memory_profile=/dev/null");

    // The options from above do not get added to the initial command environment,
    // so it has to be recreated here.
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getMemoryMetrics().getUsedHeapSizePostBuild()).isGreaterThan(0L);
    // Note that we cannot test peak heap size here since the tiny builds that we do here don't
    // trigger a full GC.
  }

  @Test
  public void testUsedHeapSizePostBuildCollectionOffByDefault() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getMemoryMetrics().getUsedHeapSizePostBuild()).isEqualTo(0);
    assertThat(buildMetrics.getMemoryMetrics().getPeakPostGcHeapSize()).isEqualTo(0);
    assertThat(buildMetrics.getMemoryMetrics().getPeakPostGcTenuredSpaceHeapSize()).isEqualTo(0);
  }

  @Test
  public void testWallTimePostBuild() throws Exception {
    buildTarget("//foo:foo");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    assertThat(buildMetrics.getTimingMetrics().getWallTimeInMs()).isGreaterThan(0);
  }

  @Test
  public void cumulativeMetrics() throws Exception {
    buildTarget("//foo:foo");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getCumulativeMetrics())
        .isEqualTo(CumulativeMetrics.newBuilder().setNumAnalyses(1).setNumBuilds(1).build());

    addOptions("--nobuild");
    buildTarget("//foo:foo");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getCumulativeMetrics())
        .isEqualTo(CumulativeMetrics.newBuilder().setNumAnalyses(2).setNumBuilds(1).build());

    addOptions("--build", "--noanalyze");
    buildTarget("//foo:foo");
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getCumulativeMetrics())
        .isEqualTo(CumulativeMetrics.newBuilder().setNumAnalyses(2).setNumBuilds(1).build());

    write(
        "foo/BUILD",
        "genrule(",
        "    name = 'foo',",
        "    outs = ['dir'],",
        "    srcs = ['//noexist:noexist'],",
        "    cmd = '/bin/mkdir $(location dir)',",
        ")");

    addOptions("--analyze");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:foo"));
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getCumulativeMetrics())
        .isEqualTo(CumulativeMetrics.newBuilder().setNumAnalyses(3).setNumBuilds(1).build());

    write(
        "foo/BUILD",
        "genrule(",
        "    name = 'foo',",
        "    outs = ['dir'],",
        "    cmd = '/bin/false',",
        ")");

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getCumulativeMetrics())
        .isEqualTo(CumulativeMetrics.newBuilder().setNumAnalyses(4).setNumBuilds(2).build());
  }

  @Test
  public void testActionData() throws Exception {
    write(
        "bar/BUILD",
        "genrule(name='bar', srcs=[':dep1',':dep2'], outs=['out'], cmd='touch $@')",
        "genrule(name='dep1', outs=['out1'], cmd='touch $@')",
        "genrule(name='dep2', outs=['out2'], cmd='touch $@')");
    buildTarget("//bar");
    BuildMetrics buildMetrics = buildMetricsEventListener.event.getBuildMetrics();
    List<ActionData> actionDataList = buildMetrics.getActionSummary().getActionDataList();
    assertThat(actionDataList).hasSize(2);
    assertThat(actionDataList.get(0).getMnemonic()).isEqualTo("Genrule");
    assertThat(actionDataList.get(0).getActionsExecuted()).isEqualTo(3);

    assertThat(actionDataList.get(1).getMnemonic()).isEqualTo("DummyBuildInfoAction");
    assertThat(actionDataList.get(1).getActionsExecuted()).isEqualTo(1);

    for (ActionData actionData : actionDataList) {
      assertThat(actionData.getFirstStartedMs()).isAtMost(actionData.getLastEndedMs());
    }
  }

  @Test
  public void skymeldNullIncrementalBuild_buildGraphMetricsNotCollected() throws Exception {
    write(
        "foo/BUILD",
        "genrule(",
        "    name = 'foo',",
        "    outs = ['dir'],",
        "    cmd = '/bin/mkdir $(location dir)',",
        "    srcs = [],",
        ")",
        "genrule(",
        "    name = 'bar',",
        "    outs = ['dir2'],",
        "    cmd = '/bin/mkdir $(location dir2)',",
        "    srcs = [],",
        ")");
    addOptions("--experimental_merged_skyframe_analysis_execution");
    BuildGraphMetrics expected =
        BuildGraphMetrics.newBuilder()
            .setActionLookupValueCount(8)
            .setActionLookupValueCountNotIncludingAspects(8)
            .setActionCount(2)
            .setActionCountNotIncludingAspects(2)
            .setInputFileConfiguredTargetCount(1)
            .setOutputArtifactCount(2)
            .build();
    buildTarget("//foo:foo", "//foo:bar");

    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .comparingExpectedFieldsOnly()
        .isEqualTo(expected);

    // Null build.
    buildTarget("//foo:foo", "//foo:bar");

    BuildGraphMetrics expectedNullBuild =
        BuildGraphMetrics.newBuilder()
            .setActionLookupValueCount(0)
            .setActionLookupValueCountNotIncludingAspects(0)
            .setActionCount(0)
            .setActionCountNotIncludingAspects(0)
            .setInputFileConfiguredTargetCount(0)
            .setOutputArtifactCount(0)
            .build();
    assertThat(buildMetricsEventListener.event.getBuildMetrics().getBuildGraphMetrics())
        .comparingExpectedFieldsOnly()
        .isEqualTo(expectedNullBuild);
  }
}
