// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Analysis failure reporting tests. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class AnalysisFailureReportingTest extends AnalysisTestCase {
  private final AnalysisFailureEventCollector collector = new AnalysisFailureEventCollector();

  // TODO(mschaller): The below is closer now because of e.g. DetailedExitCode/FailureDetail.
  // original(ulfjack): Don't check for exact error message wording; instead, add machine-readable
  // details to the events, and check for those. Also check if we can remove duplicate test coverage
  // for these errors, i.e., consolidate the failure reporting tests in this class.

  @Before
  public void setup() {
    // We only test failure cases in this class.
    reporter.removeHandler(failFastHandler);
    eventBus.register(collector);
  }

  private static ConfigurationId toId(BuildConfiguration config) {
    return config == null ? null : config.getEventId().getConfiguration();
  }

  @Test
  public void testMissingRequiredAttribute() throws Exception {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',", // missing "out" attribute
        "        cmd = '')");
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();
    Label topLevel = Label.parseAbsoluteUnchecked("//foo");

    assertThat(collector.events.keySet()).containsExactly(topLevel);

    Collection<Cause> topLevelCauses = collector.events.get(topLevel);
    assertThat(topLevelCauses).hasSize(1);

    Cause cause = Iterables.getOnlyElement(topLevelCauses);
    assertThat(cause).isInstanceOf(LoadingFailedCause.class);
    assertThat(cause.getLabel()).isEqualTo(topLevel);
    assertThat(((LoadingFailedCause) cause).getMessage())
        .isEqualTo("Target '//foo:foo' contains an error and its package is in error");
  }

  @Test
  public void testMissingDependency() throws Exception {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "        tools = ['//bar'],",
        "        cmd = 'command',",
        "        outs = ['foo.txt'])");
    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();
    Label topLevel = Label.parseAbsoluteUnchecked("//foo");
    Label causeLabel = Label.parseAbsoluteUnchecked("//bar");
    assertThat(collector.events.keySet()).containsExactly(topLevel);
    assertThat(collector.events.get(topLevel))
        .containsExactly(
            new AnalysisFailedCause(
                causeLabel,
                toId(
                    Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs())
                        .getConfiguration()),
                createPackageLoadingDetailedExitCode(
                    "BUILD file not found in any of the following"
                        + " directories. Add a BUILD file to a directory to mark it as a"
                        + " package.\n"
                        + " - /workspace/bar",
                    Code.BUILD_FILE_MISSING)));
  }

  /**
   * This error gets reported twice - once when we try to analyze the //cycles1 target, and the
   * other time when we analyze the //c target (which depends on //cycles1). This test checks that
   * both use the same error message.
   */
  @Test
  public void testSymlinkCycleReportedExactlyOnce() throws Exception {
    scratch.file("gp/BUILD", "sh_library(name = 'gp', deps = ['//p'])");
    scratch.file("p/BUILD", "sh_library(name = 'p', deps = ['//c'])");
    scratch.file("c/BUILD", "sh_library(name = 'c', deps = ['//cycles1'])");
    Path cycles1BuildFilePath =
        scratch.file("cycles1/BUILD", "sh_library(name = 'cycles1', srcs = glob(['*.sh']))");
    cycles1BuildFilePath
        .getParentDirectory()
        .getRelative("cycles1.sh")
        .createSymbolicLink(PathFragment.create("cycles1.sh"));

    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//gp");
    assertThat(result.hasError()).isTrue();

    Label topLevel = Label.parseAbsoluteUnchecked("//gp");
    String message =
        "Symlink issue while evaluating globs: Symlink cycle:" + " /workspace/cycles1/cycles1.sh";
    Code code = Code.EVAL_GLOBS_SYMLINK_ERROR;
    assertThat(collector.events.get(topLevel))
        .containsExactly(
            new AnalysisFailedCause(
                Label.parseAbsolute("//cycles1", ImmutableMap.of()),
                toId(
                    Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs())
                        .getConfiguration()),
                createPackageLoadingDetailedExitCode(message, code)));
  }

  @Test
  public void testVisibilityError() throws Exception {
    scratch.file("foo/BUILD", "sh_library(name = 'foo', deps = ['//bar'])");
    scratch.file("bar/BUILD", "sh_library(name = 'bar', visibility = ['//visibility:private'])");

    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();

    Label topLevel = Label.parseAbsoluteUnchecked("//foo");
    assertThat(collector.events.get(topLevel))
        .containsExactly(
            new AnalysisFailedCause(
                Label.parseAbsolute("//foo", ImmutableMap.of()),
                toId(
                    Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs())
                        .getConfiguration()),
                createAnalysisDetailedExitCode(
                    "in sh_library rule //foo:foo: target '//bar:bar' is not visible from"
                        + " target '//foo:foo'. Check the visibility declaration of the"
                        + " former target if you think the dependency is legitimate")));
  }

  @Test
  public void testFileVisibilityError() throws Exception {
    scratch.file("foo/BUILD", "sh_library(name = 'foo', srcs = ['//bar:bar.sh'])");
    scratch.file("bar/BUILD", "exports_files(['bar.sh'], visibility = ['//visibility:private'])");
    scratch.file("bar/bar.sh");

    AnalysisResult result = update(eventBus, defaultFlags().with(Flag.KEEP_GOING), "//foo");
    assertThat(result.hasError()).isTrue();

    Label topLevel = Label.parseAbsoluteUnchecked("//foo");
    assertThat(collector.events)
        .valuesForKey(topLevel)
        .containsExactly(
            new AnalysisFailedCause(
                Label.parseAbsolute("//foo", ImmutableMap.of()),
                toId(
                    Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs())
                        .getConfiguration()),
                DetailedExitCode.of(
                    FailureDetail.newBuilder()
                        .setMessage(
                            "in sh_library rule //foo:foo: target '//bar:bar.sh' is not visible"
                                + " from target '//foo:foo'. Check the visibility declaration of"
                                + " the former target if you think the dependency is legitimate."
                                + " To set the visibility of that source file target, use the"
                                + " exports_files() function")
                        .setAnalysis(
                            Analysis.newBuilder()
                                .setCode(Analysis.Code.CONFIGURED_VALUE_CREATION_FAILED))
                        .build())));
  }

  @Test
  public void testVisibilityErrorNoKeepGoing() throws Exception {
    scratch.file("foo/BUILD", "sh_test(name = 'foo', srcs = ['test.sh'], deps = ['//bar'])");
    scratch.file("bar/BUILD", "sh_library(name = 'bar', visibility = ['//visibility:private'])");

    try {
      update(eventBus, defaultFlags(), "//foo");
    } catch (ViewCreationFailedException e) {
      // Ignored; we check for the correct eventbus event below.
    }

    Label topLevel = Label.parseAbsoluteUnchecked("//foo");
    BuildConfiguration expectedConfig =
        Iterables.getOnlyElement(
            skyframeExecutor
                .getSkyframeBuildView()
                .getBuildConfigurationCollection()
                .getTargetConfigurations());
    String message =
        "in sh_test rule //foo:foo: target '//bar:bar' is not visible from"
            + " target '//foo:foo'. Check the visibility declaration of the"
            + " former target if you think the dependency is legitimate";
    assertThat(collector.events.get(topLevel))
        .containsExactly(
            new AnalysisFailedCause(
                Label.parseAbsolute("//foo", ImmutableMap.of()),
                toId(expectedConfig),
                createAnalysisDetailedExitCode(message)));
  }

  public static DetailedExitCode createPackageLoadingDetailedExitCode(String message, Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setPackageLoading(PackageLoading.newBuilder().setCode(code))
            .build());
  }

  public static DetailedExitCode createAnalysisDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(
                Analysis.newBuilder().setCode(Analysis.Code.CONFIGURED_VALUE_CREATION_FAILED))
            .build());
  }

  // TODO(ulfjack): Add more tests for
  // - a target that has multiple analysis errors (in the target itself)
  // - a visibility error in a dependency (not in the target itself)
  // - an error in a config condition
  // - a missing top-level target (does that even get this far?)
  // - a top-level target with an InvalidConfigurationException
  // - a top-level target with a ToolchainContextException
  // - a top-level target with a visibility attribute that points to a non-package_group
  // - a top-level target with a package_group that refers to a non-package_group
  // - aspect errors

  /** Class to collect analysis failures. */
  public static class AnalysisFailureEventCollector {
    private final Multimap<Label, Cause> events = HashMultimap.create();

    Multimap<Label, Cause> causesByLabel() {
      Multimap<Label, Cause> result = HashMultimap.create();
      return result;
    }

    @Subscribe
    public void failureEvent(AnalysisFailureEvent event) {
      events.putAll(event.getFailedTarget().getLabel(), event.getRootCauses().toList());
    }
  }
}
