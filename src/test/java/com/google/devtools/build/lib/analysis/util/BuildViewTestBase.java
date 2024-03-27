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


import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.AnalysisRootCauseEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.events.OutputFilter.RegexOutputFilter;
import com.google.devtools.build.lib.pkgcache.LoadingFailureEvent;
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

  protected final void setupDummyRule() throws Exception {
    scratch.file(
        "pkg/BUILD",
        """
        testing_dummy_rule(
            name = "foo",
            srcs = ["a.src"],
            outs = ["a.out"],
        )
        """);
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
    scratch.file(
        "parent/BUILD",
        """
        sh_library(
            name = "foo",
            srcs = [
                "//badpkg:okay-target",
                "//okaypkg:transitively-a-cycle",
            ],
        )
        """);
    Path symlinkcycleBuildFile = scratch.file("symlinkcycle/BUILD",
        "sh_library(name = 'cycle', srcs = glob(['*.sh']))");
    Path dirPath = symlinkcycleBuildFile.getParentDirectory();
    dirPath.getRelative("foo.sh").createSymbolicLink(PathFragment.create("foo.sh"));
    scratch.file(
        "okaypkg/BUILD",
        """
        sh_library(
            name = "transitively-a-cycle",
            srcs = ["//symlinkcycle:cycle"],
        )
        """);
    Path badpkgBuildFile =
        scratch.file(
            "badpkg/BUILD",
            """
            exports_files(["okay-target"])

            fail()
            """);
    if (incremental) {
      update(defaultFlags().with(Flag.KEEP_GOING), "//okaypkg:transitively-a-cycle");
      assertContainsEvent("circular symlinks detected");
      eventCollector.clear();
    }
    update(defaultFlags().with(Flag.KEEP_GOING), "//parent:foo");
    // Each event string may contain stack traces and error messages with multiple file names.
    assertContainsEventWithFrequency(badpkgBuildFile.asFragment().getPathString(), 1);
    // TODO(nharmata): This test currently only works because each BuildViewTest#update call
    // dirties all FileNodes that are in error. There is actually a skyframe bug with cycle
    // reporting on incremental builds (see b/14622820).
    assertContainsEvent("circular symlinks detected");
  }

  protected void injectGraphListenerForTesting(Listener listener, boolean deterministic) {
    InMemoryMemoizingEvaluator memoizingEvaluator =
        (InMemoryMemoizingEvaluator) skyframeExecutor.getEvaluator();
    memoizingEvaluator.injectGraphTransformerForTesting(
        DeterministicHelper.makeTransformer(listener, deterministic));
  }

  /**
   * Record analysis failures.
   */
  public static class AnalysisFailureRecorder {
    @Subscribe
    public void analysisFailure(AnalysisFailureEvent event) {
      events.add(event);
    }

    @Subscribe
    public void analysisFailureCause(AnalysisRootCauseEvent event) {
      causes.add(event);
    }

    public final List<AnalysisFailureEvent> events = new ArrayList<>();
    public final List<AnalysisRootCauseEvent> causes = new ArrayList<>();
  }

  /**
   * Record loading failures.
   */
  public static class LoadingFailureRecorder {
    @Subscribe
    public void loadingFailure(LoadingFailureEvent event) {
      events.add(event);
    }

    public final List<LoadingFailureEvent> events = new ArrayList<>();
  }
}
