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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
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
    Path linkedInput = directories.getExecRoot().getRelative("input.txt");
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
    assertEquals(Sets.newHashSet(inputArtifact), Sets.newHashSet(inputs));
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    Iterable<Artifact> outputs = action.getOutputs();
    assertEquals(Sets.newHashSet(outputArtifact), Sets.newHashSet(outputs));
  }

  @Test
  public void testSymlink() throws Exception {
    Executor executor = new TestExecutorBuilder(directories, null).build();
    action.execute(new ActionExecutionContext(executor, null, null, null,
        ImmutableMap.<String, String>of(), null));
    assertTrue(output.isSymbolicLink());
    assertEquals(input, output.resolveSymbolicLinks());
    assertEquals(inputArtifact, action.getPrimaryInput());
    assertEquals(outputArtifact, action.getPrimaryOutput());
  }

  @Test
  public void testExecutableSymlink() throws Exception {
    Executor executor = new TestExecutorBuilder(directories, null).build();
    outputArtifact = getBinArtifactWithNoOwner("destination2.txt");
    output = outputArtifact.getPath();
    action = new ExecutableSymlinkAction(NULL_ACTION_OWNER, inputArtifact, outputArtifact);
    assertFalse(input.isExecutable());
    ActionExecutionContext actionExecutionContext =
      new ActionExecutionContext(executor, null, null, null,
          ImmutableMap.<String, String>of(), null);
    try {
      action.execute(actionExecutionContext);
      fail("Expected ActionExecutionException");
    } catch (ActionExecutionException e) {
      assertThat(e.getMessage()).containsMatch("'input.txt' is not executable");
    }
    input.setExecutable(true);
    action.execute(actionExecutionContext);
    assertTrue(output.isSymbolicLink());
    assertEquals(input, output.resolveSymbolicLinks());
  }
}
