// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link BasicActionLookupValue}. */
@RunWith(JUnit4.class)
public class BasicActionLookupValueTest {

  private FileSystem fs;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  @Before
  public void setUp() {
    fs = new InMemoryFileSystem();
  }

  @Test
  public void testActionPresentAfterEvaluation() {
    Action action = mock(Action.class);
    Artifact artifact = mock(Artifact.class);
    when(action.getOutputs()).thenReturn(ImmutableSet.of(artifact));
    when(action.canRemoveAfterExecution()).thenReturn(true);
    ActionLookupValue underTest = new BasicActionLookupValue(action, false);
    assertThat(underTest.getGeneratingActionIndex(artifact)).isEqualTo(0);
    assertThat(underTest.getAction(0)).isSameAs(action);
    underTest.actionEvaluated(0, action);
    assertThat(underTest.getAction(0)).isSameAs(action);
  }

  @Test
  public void testActionNotPresentAfterEvaluation() throws ActionConflictException {
    Path execRoot = fs.getPath("/execroot");
    Path outputRootPath = execRoot.getRelative("blaze-out");
    ArtifactRoot root = ArtifactRoot.asDerivedRoot(execRoot, outputRootPath);
    Action normalAction = mock(Action.class);
    Artifact normalArtifact = new Artifact(PathFragment.create("normal"), root);
    when(normalAction.getOutputs()).thenReturn(ImmutableSet.of(normalArtifact));
    when(normalAction.canRemoveAfterExecution()).thenReturn(true);
    Action persistentAction = mock(Action.class);
    Artifact persistentOutput = new Artifact(PathFragment.create("persistent"), root);
    when(persistentAction.getOutputs()).thenReturn(ImmutableSet.of(persistentOutput));
    when(persistentAction.canRemoveAfterExecution()).thenReturn(false);
    ActionLookupValue underTest =
        new BasicActionLookupValue(
            Actions.filterSharedActionsAndThrowActionConflict(
                actionKeyContext, ImmutableList.of(normalAction, persistentAction)),
            true);
    assertThat(underTest.getGeneratingActionIndex(normalArtifact)).isEqualTo(0);
    assertThat(underTest.getAction(0)).isSameAs(normalAction);
    assertThat(underTest.getGeneratingActionIndex(persistentOutput)).isEqualTo(1);
    assertThat(underTest.getAction(1)).isSameAs(persistentAction);
    underTest.actionEvaluated(0, normalAction);
    try {
      underTest.getAction(0);
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
    assertThat(underTest.getGeneratingActionIndex(persistentOutput)).isEqualTo(1);
    assertThat(underTest.getAction(1)).isSameAs(persistentAction);
    underTest.actionEvaluated(1, persistentAction);
    // Action that said not to clear it won't be cleared.
    assertThat(underTest.getGeneratingActionIndex(persistentOutput)).isEqualTo(1);
    assertThat(underTest.getAction(1)).isSameAs(persistentAction);
  }
}
