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

package com.google.devtools.build.lib.analysis.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.OutputFilter.RegexOutputFilter;
import com.google.devtools.build.lib.pkgcache.LoadingFailureEvent;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.DeterministicHelper;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.NotifyingHelper.Listener;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Base class for BuildView test cases.
 */
public abstract class BuildViewTestBase extends AnalysisTestCase {

  protected static int getFrequencyOfErrorsWithLocation(
      PathFragment path, EventCollector eventCollector) {
    int frequency = 0;
    for (Event event : eventCollector) {
      if (event.getLocation() != null) {
        if (path.equals(event.getLocation().getPath())) {
          frequency++;
        }
      }
    }
    return frequency;
  }

  protected final void setupDummyRule() throws Exception {
    scratch.file("pkg/BUILD",
                "testing_dummy_rule(name='foo', ",
                "                   srcs=['a.src'],",
                "                   outs=['a.out'])");
  }

  protected void runAnalysisWithOutputFilter(Pattern outputFilter) throws Exception {
    scratch.file("java/a/BUILD",
        "exports_files(['A.java'])");
    scratch.file("java/b/BUILD",
        "java_library(name = 'b', srcs = ['//java/a:A.java'])");
    scratch.file("java/c/BUILD",
        "java_library(name = 'c', exports = ['//java/b:b'])");
    reporter.setOutputFilter(RegexOutputFilter.forPattern(outputFilter));
    update("//java/c:c");
  }

  protected Artifact getNativeDepsLibrary(ConfiguredTarget target) throws Exception {
    return ActionsTestUtil.getFirstArtifactEndingWith(target
        .getProvider(RunfilesProvider.class)
        .getDefaultRunfiles()
        .getAllArtifacts(), "_swigdeps.so");
  }

  protected void runTestDepOnGoodTargetInBadPkgAndTransitiveCycle(boolean incremental)
      throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("parent/BUILD",
        "sh_library(name = 'foo',",
        "           srcs = ['//badpkg:okay-target', '//okaypkg:transitively-a-cycle'])");
    Path symlinkcycleBuildFile = scratch.file("symlinkcycle/BUILD",
        "sh_library(name = 'cycle', srcs = glob(['*.sh']))");
    Path dirPath = symlinkcycleBuildFile.getParentDirectory();
    dirPath.getRelative("foo.sh").createSymbolicLink(PathFragment.create("foo.sh"));
    scratch.file("okaypkg/BUILD",
        "sh_library(name = 'transitively-a-cycle',",
        "           srcs = ['//symlinkcycle:cycle'])");
    Path badpkgBuildFile = scratch.file("badpkg/BUILD",
        "exports_files(['okay-target'])",
        "invalidbuildsyntax");
    if (incremental) {
      update(defaultFlags().with(Flag.KEEP_GOING), "//okaypkg:transitively-a-cycle");
      assertContainsEvent("circular symlinks detected");
      eventCollector.clear();
    }
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:foo");
    assertThat(getFrequencyOfErrorsWithLocation(badpkgBuildFile.asFragment(), eventCollector))
        .isEqualTo(1);
    // TODO(nharmata): This test currently only works because each BuildViewTest#update call
    // dirties all FileNodes that are in error. There is actually a skyframe bug with cycle
    // reporting on incremental builds (see b/14622820).
    assertContainsEvent("circular symlinks detected");
  }

  protected void injectGraphListenerForTesting(Listener listener, boolean deterministic) {
    InMemoryMemoizingEvaluator memoizingEvaluator =
        (InMemoryMemoizingEvaluator) skyframeExecutor.getEvaluatorForTesting();
    memoizingEvaluator.injectGraphTransformerForTesting(
        DeterministicHelper.makeTransformer(listener, deterministic));
  }

  protected void runTestForMultiCpuAnalysisFailure(String badCpu, String goodCpu) throws Exception {
    reporter.removeHandler(failFastHandler);
    useConfiguration("--experimental_multi_cpu=" + badCpu + "," + goodCpu);
    scratch.file("multi/BUILD",
        "config_setting(",
        "    name = 'config',",
        "    values = {'cpu': '" + badCpu + "'})",
        "cc_library(",
        "    name = 'cpu',",
        "    deps = select({",
        "        ':config': [':fail'],",
        "        '//conditions:default': []}))",
        "genrule(",
        "    name = 'fail',",
        "    outs = ['file1', 'file2'],",
        "    executable = 1,",
        "    cmd = 'touch $@')");
    update(defaultFlags().with(Flag.KEEP_GOING), "//multi:cpu");
    AnalysisResult result = getAnalysisResult();
    assertThat(result.getTargetsToBuild()).hasSize(1);
    ConfiguredTarget targetA = Iterables.get(result.getTargetsToBuild(), 0);
    assertThat(targetA.getConfiguration().getCpu()).isEqualTo(goodCpu);
    // Unfortunately, we get the same error twice - we can't distinguish the configurations.
    assertContainsEvent("if genrules produce executables, they are allowed only one output");
  }

  /**
   * Record analysis failures.
   */
  public static class AnalysisFailureRecorder {
    @Subscribe
    public void analysisFailure(AnalysisFailureEvent event) {
      events.add(event);
    }

    public final List<AnalysisFailureEvent> events = new ArrayList<>();
  }

  /**
   * Record loading failures.
   */
  public static class LoadingFailureRecorder {
    @Subscribe
    public void loadingFailure(LoadingFailureEvent event) {
      events.add(Pair.of(event.getFailedTarget(), event.getFailureReason()));
    }

    public final List<Pair<Label, Label>> events = new ArrayList<>();
  }
}
