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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.Scratch;
import java.util.Collection;
import java.util.Collections;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FailActionTest {

  private Scratch scratch = new Scratch();

  private String errorMessage;
  private Artifact anOutput;
  private Collection<Artifact> outputs;
  private FailAction failAction;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  protected MutableActionGraph actionGraph = new MapBasedActionGraph(actionKeyContext);

  @Before
  public final void setUp() throws Exception  {
    errorMessage = "An error just happened.";
    anOutput =
        new Artifact(
            scratch.file("/out/foo"),
            ArtifactRoot.asDerivedRoot(scratch.dir("/"), scratch.dir("/out")));
    outputs = ImmutableList.of(anOutput);
    failAction = new FailAction(NULL_ACTION_OWNER, outputs, errorMessage);
    actionGraph.registerAction(failAction);
    assertThat(actionGraph.getGeneratingAction(anOutput)).isSameAs(failAction);
  }

  @Test
  public void testExecutingItYieldsExceptionWithErrorMessage() {
    try {
      failAction.execute(null);
      fail();
    } catch (ActionExecutionException e) {
      assertThat(e).hasMessageThat().isEqualTo(errorMessage);
    }
  }

  @Test
  public void testInputsAreEmptySet() {
    assertThat(failAction.getInputs()).containsExactlyElementsIn(Collections.emptySet());
  }

  @Test
  public void testRetainsItsOutputs() {
    assertThat(failAction.getOutputs()).containsExactlyElementsIn(outputs);
  }

  @Test
  public void testPrimaryOutput() {
    assertThat(failAction.getPrimaryOutput()).isSameAs(anOutput);
  }
}
