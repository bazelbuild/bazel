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
package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SymlinkAction}.
 */
@RunWith(JUnit4.class)
public class SymlinkActionTest extends BuildViewTestCase {

  private Path input;
  private Artifact inputArtifact;
  private Path output;
  private Artifact outputArtifact;
  private SymlinkAction action;

  @Before
  public final void setUp() throws Exception  {
    input = scratch.file("input.txt", "Hello, world.");
    inputArtifact = getSourceArtifact("input.txt");
    Path linkedInput =
        directories.getExecRoot(TestConstants.WORKSPACE_NAME).getRelative("input.txt");
    FileSystemUtils.createDirectoryAndParents(linkedInput.getParentDirectory());
    linkedInput.createSymbolicLink(input);
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    output = outputArtifact.getPath();
    FileSystemUtils.createDirectoryAndParents(output.getParentDirectory());
    action = new SymlinkAction(NULL_ACTION_OWNER,
        inputArtifact, outputArtifact, "Symlinking test");
  }

  @Test
  public void testInputArtifactIsInput() {
    Iterable<Artifact> inputs = action.getInputs();
    assertThat(inputs).containsExactly(inputArtifact);
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    Iterable<Artifact> outputs = action.getOutputs();
    assertThat(outputs).containsExactly(outputArtifact);
  }

  @Test
  public void testSymlink() throws Exception {
    Executor executor = new TestExecutorBuilder(directories, null).build();
    action.execute(new ActionExecutionContext(executor, null, ActionInputPrefetcher.NONE, null,
        null, ImmutableMap.<String, String>of(), null));
    assertThat(output.isSymbolicLink()).isTrue();
    assertThat(output.resolveSymbolicLinks()).isEqualTo(input);
    assertThat(action.getPrimaryInput()).isEqualTo(inputArtifact);
    assertThat(action.getPrimaryOutput()).isEqualTo(outputArtifact);
  }
}
