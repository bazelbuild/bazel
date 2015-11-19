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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanAction;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * A test for {@link MiddlemanAction}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
public class MiddlemanActionTest extends BuildViewTestCase {

  private AnalysisTestUtil.CollectingAnalysisEnvironment analysisEnvironment;
  private MiddlemanFactory middlemanFactory;
  private Artifact a, b, middle;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    scratch.file("a/BUILD",
                "testing_dummy_rule(name='a', outs=['a.out'])");
    scratch.file("b/BUILD",
                "testing_dummy_rule(name='b', outs=['b.out'])");
    a = getFilesToBuild(getConfiguredTarget("//a")).iterator().next();
    b = getFilesToBuild(getConfiguredTarget("//b")).iterator().next();
    analysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(
            AnalysisTestUtil.STUB_ANALYSIS_ENVIRONMENT);
    middlemanFactory = new MiddlemanFactory(view.getArtifactFactory(), analysisEnvironment);
    middle = middlemanFactory.createAggregatingMiddleman(
        NULL_ACTION_OWNER, "middleman_test",
        Arrays.asList(a, b),
        targetConfig.getMiddlemanDirectory());
    analysisEnvironment.registerWith(getMutableActionGraph());
  }

  public void testActionIsAMiddleman() {
    Action middleman = getGeneratingAction(middle);
    assertTrue("Encountered instance of " + middleman.getClass(),
        middleman.getActionType().isMiddleman());
  }

  public void testAAndBAreInputsToMiddleman() {
    MiddlemanAction middleman = (MiddlemanAction) getGeneratingAction(middle);
    assertThat(middleman.getInputs()).containsExactly(a, b);
  }

  public void testMiddleIsOutputOfMiddleman() {
    MiddlemanAction middleman = (MiddlemanAction) getGeneratingAction(middle);
    assertThat(middleman.getOutputs()).containsExactly(middle);
  }

  public void testMiddlemanIsNullForEmptyInputs() throws Exception {
    assertNull(middlemanFactory.createAggregatingMiddleman(NULL_ACTION_OWNER,
        "middleman_test", new ArrayList<Artifact>(), targetConfig.getMiddlemanDirectory()));
  }

  public void testMiddlemanIsIdentityForLonelyInput() throws Exception {
    assertEquals(a,
        middlemanFactory.createAggregatingMiddleman(
            NULL_ACTION_OWNER, "middleman_test",
            Lists.newArrayList(a),
            targetConfig.getMiddlemanDirectory()));
  }

  public void testDifferentExecutablesForRunfilesMiddleman() throws Exception {
    scratch.file("c/BUILD",
                "testing_dummy_rule(name='c', outs=['c.out', 'd.out', 'common.out'])");

    Artifact c = getFilesToBuild(getConfiguredTarget("//c:c.out")).iterator().next();
    Artifact d = getFilesToBuild(getConfiguredTarget("//c:d.out")).iterator().next();
    Artifact common = getFilesToBuild(getConfiguredTarget("//c:common.out")).iterator().next();

    analysisEnvironment.clear();
    Artifact middlemanForC = middlemanFactory.createRunfilesMiddleman(
        NULL_ACTION_OWNER, c, Arrays.asList(c, common), targetConfig.getMiddlemanDirectory());
    Artifact middlemanForD = middlemanFactory.createRunfilesMiddleman(
        NULL_ACTION_OWNER, d, Arrays.asList(d, common), targetConfig.getMiddlemanDirectory());
    analysisEnvironment.registerWith(getMutableActionGraph());

    MiddlemanAction middlemanActionForC = (MiddlemanAction) getGeneratingAction(middlemanForC);
    MiddlemanAction middlemanActionForD = (MiddlemanAction) getGeneratingAction(middlemanForD);

    assertThat(Sets.newHashSet(middlemanActionForD.getInputs()))
        .isNotEqualTo(Sets.newHashSet(middlemanActionForC.getInputs()));
    assertThat(Sets.newHashSet(middlemanActionForD.getOutputs()))
        .isNotEqualTo(Sets.newHashSet(middlemanActionForC.getOutputs()));
  }
}
