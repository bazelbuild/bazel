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
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests {@link SymlinkAction}. */
@RunWith(TestParameterInjector.class)
public class SymlinkActionTest extends BuildViewTestCase {

  @TestParameter private boolean useExecRootForSources;

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
    linkedInput.getParentDirectory().createDirectoryAndParents();
    linkedInput.createSymbolicLink(input);
    outputArtifact = getBinArtifactWithNoOwner("destination.txt");
    outputArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    output = outputArtifact.getPath();
    output.getParentDirectory().createDirectoryAndParents();
    action =
        SymlinkAction.toArtifact(
            NULL_ACTION_OWNER,
            inputArtifact,
            outputArtifact,
            "Symlinking test: %{label}: %{input} -> %{output}",
            useExecRootForSources);
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
                mock(InputMetadataProvider.class),
                ActionInputPrefetcher.NONE,
                actionKeyContext,
                mock(OutputMetadataStore.class),
                /* rewindingEnabled= */ false,
                LostInputsCheck.NONE,
                /* fileOutErr= */ null,
                new StoredEventHandler(),
                /* clientEnv= */ ImmutableMap.of(),
                /* topLevelFilesets= */ ImmutableMap.of(),
                /* artifactExpander= */ null,
                /* actionFileSystem= */ null,
                /* skyframeDepsResult= */ null,
                DiscoveredModulesPruner.DEFAULT,
                SyscallCache.NO_CACHE,
                ThreadStateReceiver.NULL_INSTANCE));
    assertThat(actionResult.spawnResults()).isEmpty();
    assertThat(output.isSymbolicLink()).isTrue();
    assertThat(output.resolveSymbolicLinks()).isEqualTo(input);
    assertThat(action.getPrimaryInput()).isEqualTo(inputArtifact);
    assertThat(action.getPrimaryOutput()).isEqualTo(outputArtifact);
    assertThat(action.getProgressMessage())
        .isEqualTo("Symlinking test: //null/action:owner: input.txt -> destination.txt");
  }

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(action)
        .addDependency(FileSystem.class, scratch.getFileSystem())
        .addDependency(Root.RootCodecDependencies.class, new Root.RootCodecDependencies(root))
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
