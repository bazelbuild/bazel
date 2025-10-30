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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testing.vfs.SpiedFileSystem;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests {@link SymlinkAction}. */
@RunWith(TestParameterInjector.class)
public class SymlinkActionTest extends BuildViewTestCase {

  private Executor executor;
  private SpiedFileSystem fs;

  @Before
  public void setUp() throws Exception {
    executor =
        new TestExecutorBuilder(
                fileSystem,
                directories,
                BinTools.forEmbeddedBin(directories.getEmbeddedBinariesRoot(), ImmutableList.of()))
            .build();
  }

  @Override
  public FileSystem createFileSystem() {
    fs = SpiedFileSystem.createInMemorySpy();
    return fs;
  }

  @Test
  public void testSymlinkToSourceFile(@TestParameter boolean useExecRootForSource)
      throws Exception {
    Artifact inputArtifact = getSourceArtifact("input");
    Artifact outputArtifact = getBinArtifactWithNoOwner("output");

    Path inputPath = directories.getExecRoot(TestConstants.WORKSPACE_NAME).getRelative("input");
    inputPath.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.createEmptyFile(inputPath);
    inputArtifact.getPath().createSymbolicLink(inputPath);
    outputArtifact.getPath().getParentDirectory().createDirectoryAndParents();

    runSymlinkAction(inputArtifact, outputArtifact, useExecRootForSource);

    PathFragment expectedTarget =
        useExecRootForSource
            ? getExecRoot().getRelative(inputArtifact.getExecPath()).asFragment()
            : inputArtifact.getPath().asFragment();

    assertThat(outputArtifact.getPath().isSymbolicLink()).isTrue();
    assertThat(outputArtifact.getPath().readSymbolicLink()).isEqualTo(expectedTarget);

    verify(fs)
        .createSymbolicLink(
            outputArtifact.getPath().asFragment(), expectedTarget, SymlinkTargetType.FILE);
  }

  @Test
  public void testSymlinkToSourceDirectory(@TestParameter boolean useExecRootForSource)
      throws Exception {
    Artifact inputArtifact = getSourceArtifact("input");
    Artifact outputArtifact = getBinArtifactWithNoOwner("output");

    Path inputPath = directories.getExecRoot(TestConstants.WORKSPACE_NAME).getRelative("input");
    inputPath.createDirectoryAndParents();
    inputArtifact.getPath().createSymbolicLink(inputPath);
    outputArtifact.getPath().getParentDirectory().createDirectoryAndParents();

    runSymlinkAction(inputArtifact, outputArtifact, useExecRootForSource);

    PathFragment expectedTarget =
        useExecRootForSource
            ? getExecRoot().getRelative(inputArtifact.getExecPath()).asFragment()
            : inputArtifact.getPath().asFragment();

    assertThat(outputArtifact.getPath().isSymbolicLink()).isTrue();
    assertThat(outputArtifact.getPath().readSymbolicLink()).isEqualTo(expectedTarget);

    verify(fs)
        .createSymbolicLink(
            outputArtifact.getPath().asFragment(), expectedTarget, SymlinkTargetType.DIRECTORY);
  }

  @Test
  public void testSymlinkToOutputFile() throws Exception {
    Artifact inputArtifact = getBinArtifactWithNoOwner("input");
    Artifact outputArtifact = getBinArtifactWithNoOwner("output");

    inputArtifact.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(inputArtifact.getPath(), UTF_8, "hello world");
    outputArtifact.getPath().getParentDirectory().createDirectoryAndParents();

    runSymlinkAction(inputArtifact, outputArtifact);

    assertThat(outputArtifact.getPath().isSymbolicLink()).isTrue();
    assertThat(outputArtifact.getPath().readSymbolicLink())
        .isEqualTo(inputArtifact.getPath().asFragment());

    verify(fs)
        .createSymbolicLink(
            outputArtifact.getPath().asFragment(),
            inputArtifact.getPath().asFragment(),
            SymlinkTargetType.FILE);
  }

  @Test
  public void testSymlinkToOutputTree() throws Exception {
    Artifact inputArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            getTargetConfiguration().getBinDir(), "input");
    Artifact outputArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            getTargetConfiguration().getBinDir(), "output");

    inputArtifact.getPath().createDirectoryAndParents();
    outputArtifact.getPath().createDirectoryAndParents();

    runSymlinkAction(inputArtifact, outputArtifact);

    assertThat(outputArtifact.getPath().isSymbolicLink()).isTrue();
    assertThat(outputArtifact.getPath().readSymbolicLink())
        .isEqualTo(inputArtifact.getPath().asFragment());

    verify(fs)
        .createSymbolicLink(
            outputArtifact.getPath().asFragment(),
            inputArtifact.getPath().asFragment(),
            SymlinkTargetType.DIRECTORY);
  }

  @Test
  public void testSymlinkToAbsolutePath() throws Exception {
    Artifact outputArtifact = getBinArtifactWithNoOwner("output");

    outputArtifact.getPath().getParentDirectory().createDirectoryAndParents();

    runSymlinkAction(PathFragment.create("/some/path"), outputArtifact);

    assertThat(outputArtifact.getPath().isSymbolicLink()).isTrue();
    assertThat(outputArtifact.getPath().readSymbolicLink())
        .isEqualTo(PathFragment.create("/some/path"));

    verify(fs)
        .createSymbolicLink(
            outputArtifact.getPath().asFragment(),
            PathFragment.create("/some/path"),
            SymlinkTargetType.UNSPECIFIED);
  }

  @Test
  public void testCodec(@TestParameter boolean useExecRootForSource) throws Exception {
    Artifact inputArtifact = getSourceArtifact("input");
    Artifact.DerivedArtifact outputArtifact = getBinArtifactWithNoOwner("output");
    outputArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);

    SymlinkAction action =
        SymlinkAction.toArtifact(
            NULL_ACTION_OWNER,
            inputArtifact,
            outputArtifact,
            "Test symlink action",
            useExecRootForSource);

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

  private void runSymlinkAction(
      Artifact inputArtifact, Artifact outputArtifact, boolean useExecRootForSource)
      throws Exception {
    var action =
        SymlinkAction.toArtifact(
            NULL_ACTION_OWNER,
            inputArtifact,
            outputArtifact,
            "Test symlink action",
            useExecRootForSource);

    assertThat(action.getInputs().toList()).containsExactly(inputArtifact);
    assertThat(action.getOutputs()).containsExactly(outputArtifact);
    assertThat(action.getProgressMessage()).isEqualTo("Test symlink action");

    execute(action);
  }

  private void runSymlinkAction(Artifact inputArtifact, Artifact outputArtifact) throws Exception {
    var action =
        SymlinkAction.toArtifact(
            NULL_ACTION_OWNER,
            inputArtifact,
            outputArtifact,
            "Test symlink action",
            /* useExecRootForSource= */ false);

    assertThat(action.getInputs().toList()).containsExactly(inputArtifact);
    assertThat(action.getOutputs()).containsExactly(outputArtifact);
    assertThat(action.getProgressMessage()).isEqualTo("Test symlink action");

    execute(action);
  }

  private void runSymlinkAction(PathFragment absolutePath, Artifact outputArtifact)
      throws Exception {
    var action =
        SymlinkAction.toAbsolutePath(
            NULL_ACTION_OWNER, absolutePath, outputArtifact, "Test symlink action");

    assertThat(action.getInputs().toList()).isEmpty();
    assertThat(action.getOutputs()).containsExactly(outputArtifact);
    assertThat(action.getProgressMessage()).isEqualTo("Test symlink action");

    execute(action);
  }

  private void execute(SymlinkAction action) throws Exception {
    ActionResult actionResult =
        action.execute(
            new ActionExecutionContext(
                executor,
                createInputMetadataProvider(action.getInputs().toList()),
                ActionInputPrefetcher.NONE,
                actionKeyContext,
                /* outputMetadataStore= */ null,
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
  }

  private static InputMetadataProvider createInputMetadataProvider(Iterable<Artifact> inputs)
      throws IOException {
    ActionInputMap inputMap = new ActionInputMap(1);
    for (Artifact input : inputs) {
      if (input.isTreeArtifact()) {
        inputMap.putTreeArtifact(
            (Artifact.SpecialArtifact) input, TreeArtifactValue.empty(), /* depOwner= */ null);
      } else {
        inputMap.put(input, FileArtifactValue.createForTesting(input), /* depOwner= */ null);
      }
    }
    return inputMap;
  }
}
