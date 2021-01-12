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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link MiddlemanAction}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class MiddlemanActionTest extends BuildViewTestCase {

  private AnalysisTestUtil.CollectingAnalysisEnvironment analysisEnvironment;
  private MiddlemanFactory middlemanFactory;
  private Artifact a;
  private Artifact b;

  @Before
  public final void initializeMiddleman() throws Exception  {
    scratch.file("a/BUILD",
                "testing_dummy_rule(name='a', outs=['a.out'])");
    scratch.file("b/BUILD",
                "testing_dummy_rule(name='b', outs=['b.out'])");
    a = getFilesToBuild(getConfiguredTarget("//a")).toList().get(0);
    b = getFilesToBuild(getConfiguredTarget("//b")).toList().get(0);
    analysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(
            AnalysisTestUtil.STUB_ANALYSIS_ENVIRONMENT);
    middlemanFactory = new MiddlemanFactory(view.getArtifactFactory(), analysisEnvironment);
  }

  @Test
  public void testActionIsAMiddleman() throws Exception {
    Artifact middle =
        middlemanFactory.createRunfilesMiddleman(
            NULL_ACTION_OWNER,
            null,
            NestedSetBuilder.<Artifact>stableOrder().add(a).add(b).build(),
            targetConfig.getMiddlemanDirectory(RepositoryName.MAIN),
            "runfiles");
    analysisEnvironment.registerWith(getMutableActionGraph());
    Action middleman = getGeneratingAction(middle);

    assertWithMessage("Encountered instance of " + middleman.getClass())
        .that(middleman.getActionType().isMiddleman())
        .isTrue();
    assertThat(middleman.getInputs().toList()).containsExactly(a, b);
    assertThat(middleman.getOutputs()).containsExactly(middle);
  }

  @Test
  public void testDifferentExecutablesForRunfilesMiddleman() throws Exception {
    scratch.file("c/BUILD",
                "testing_dummy_rule(name='c', outs=['c.out', 'd.out', 'common.out'])");

    Artifact c = getFilesToBuild(getConfiguredTarget("//c:c.out")).toList().get(0);
    Artifact d = getFilesToBuild(getConfiguredTarget("//c:d.out")).toList().get(0);
    Artifact common = getFilesToBuild(getConfiguredTarget("//c:common.out")).toList().get(0);

    analysisEnvironment.clear();
    Artifact middlemanForC =
        middlemanFactory.createRunfilesMiddleman(
            NULL_ACTION_OWNER,
            c,
            NestedSetBuilder.<Artifact>stableOrder().add(c).add(common).build(),
            targetConfig.getMiddlemanDirectory(RepositoryName.MAIN),
            "runfiles");
    Artifact middlemanForD =
        middlemanFactory.createRunfilesMiddleman(
            NULL_ACTION_OWNER,
            d,
            NestedSetBuilder.<Artifact>stableOrder().add(d).add(common).build(),
            targetConfig.getMiddlemanDirectory(RepositoryName.MAIN),
            "runfiles");
    analysisEnvironment.registerWith(getMutableActionGraph());

    MiddlemanAction middlemanActionForC = (MiddlemanAction) getGeneratingAction(middlemanForC);
    MiddlemanAction middlemanActionForD = (MiddlemanAction) getGeneratingAction(middlemanForD);

    assertThat(Sets.newHashSet(middlemanActionForD.getInputs()))
        .isNotEqualTo(Sets.newHashSet(middlemanActionForC.getInputs()));
    assertThat(Sets.newHashSet(middlemanActionForD.getOutputs()))
        .isNotEqualTo(Sets.newHashSet(middlemanActionForC.getOutputs()));
  }
}
