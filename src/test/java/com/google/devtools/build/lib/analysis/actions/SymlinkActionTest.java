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
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
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
  private Artifact.DerivedArtifact outputArtifact;
  private SymlinkAction action;

  @Before
  public final void setUp() throws Exception {
    input = scratch.file("input.txt", "Hello, world.");
    inputArtifact = getSourceArtifact("input.txt");
    Path linkedInput =
        directories.getExecRoot(TestConstants.WORKSPACE_NAME).getRelative("input.txt");
    FileSystemUtils.createDirectoryAndParents(linkedInput.getParentDirectory());
    linkedInput.createSymbolicLink(input);
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    outputArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    output = outputArtifact.getPath();
    FileSystemUtils.createDirectoryAndParents(output.getParentDirectory());
    action = SymlinkAction.toArtifact(NULL_ACTION_OWNER,
        inputArtifact, outputArtifact, "Symlinking test");
  }

  @Test
  public void testInputArtifactIsInput() {
    Iterable<Artifact> inputs = action.getInputs().toList();
    assertThat(inputs).containsExactly(inputArtifact);
  }

  @Test
  public void testDestinationArtifactIsOutput() {
    Iterable<Artifact> outputs = action.getOutputs();
    assertThat(outputs).containsExactly(outputArtifact);
  }

  @Test
  public void testSymlink() throws Exception {
    Executor executor = new TestExecutorBuilder(fileSystem, directories, null).build();
    ActionResult actionResult =
        action.execute(
            new ActionExecutionContext(
                executor,
                /*actionInputFileCache=*/ null,
                ActionInputPrefetcher.NONE,
                actionKeyContext,
                /*metadataHandler=*/ null,
                LostInputsCheck.NONE,
                /*fileOutErr=*/ null,
                new StoredEventHandler(),
                /*clientEnv=*/ ImmutableMap.of(),
                /*topLevelFilesets=*/ ImmutableMap.of(),
                /*artifactExpander=*/ null,
                /*actionFileSystem=*/ null,
                /*skyframeDepsResult=*/ null,
                NestedSetExpander.DEFAULT));
    assertThat(actionResult.spawnResults()).isEmpty();
    assertThat(output.isSymbolicLink()).isTrue();
    assertThat(output.resolveSymbolicLinks()).isEqualTo(input);
    assertThat(action.getPrimaryInput()).isEqualTo(inputArtifact);
    assertThat(action.getPrimaryOutput()).isEqualTo(outputArtifact);
  }

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(action)
        .addDependency(FileSystem.class, scratch.getFileSystem())
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .setVerificationFunction(
            (in, out) -> {
              SymlinkAction inAction = (SymlinkAction) in;
              SymlinkAction outAction = (SymlinkAction) out;
              assertThat(inAction.getPrimaryInput().getFilename())
                  .isEqualTo(outAction.getPrimaryInput().getFilename());
              assertThat(inAction.getPrimaryOutput().getFilename())
                  .isEqualTo(outAction.getPrimaryOutput().getFilename());
              assertThat(inAction.getOwner()).isEqualTo(outAction.getOwner());
              assertThat(inAction.getProgressMessage()).isEqualTo(outAction.getProgressMessage());
            })
        .runTests();
  }
}
