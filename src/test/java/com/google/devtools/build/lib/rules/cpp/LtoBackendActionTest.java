// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.rules.cpp.LtoBackendAction}. */
@RunWith(JUnit4.class)
public class LtoBackendActionTest extends BuildViewTestCase {
  private Artifact bitcode1Artifact;
  private Artifact bitcode2Artifact;
  private Artifact index1Artifact;
  private Artifact index2Artifact;
  private Artifact imports1Artifact;
  private Artifact imports2Artifact;
  private Artifact destinationArtifact;
  private BitcodeFiles allBitcodeFiles;
  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;
  private Executor executor;
  private ActionExecutionContext context;

  @Before
  public final void createArtifacts() throws Exception {
    collectingAnalysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
    bitcode1Artifact = getSourceArtifact("bitcode1.o");
    bitcode2Artifact = getSourceArtifact("bitcode2.o");
    index1Artifact = getSourceArtifact("bitcode1.thinlto.bc");
    index2Artifact = getSourceArtifact("bitcode2.thinlto.bc");
    scratch.file("bitcode1.imports");
    scratch.file("bitcode2.imports", "bitcode1.o");
    imports1Artifact = getSourceArtifact("bitcode1.imports");
    imports2Artifact = getSourceArtifact("bitcode2.imports");
    destinationArtifact = getBinArtifactWithNoOwner("output");
    allBitcodeFiles =
        new BitcodeFiles(
            ImmutableMap.<PathFragment, Artifact>builder()
                .put(bitcode1Artifact.getExecPath(), bitcode1Artifact)
                .put(bitcode2Artifact.getExecPath(), bitcode2Artifact)
                .build());
  }

  @Before
  public final void createExecutorAndContext() throws Exception {
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    executor = new TestExecutorBuilder(fileSystem, directories, binTools).build();
    context =
        new ActionExecutionContext(
            executor,
            /*actionInputFileCache=*/ null,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            /*metadataHandler=*/ null,
            LostInputsCheck.NONE,
            new FileOutErr(),
            new StoredEventHandler(),
            /*clientEnv=*/ ImmutableMap.of(),
            /*topLevelFilesets=*/ ImmutableMap.of(),
            /*artifactExpander=*/ null,
            /*actionFileSystem=*/ null,
            /*skyframeDepsResult=*/ null);
  }

  @Test
  public void testEmptyImports() throws Exception {
    Action[] actions =
        new LtoBackendAction.Builder()
            .addImportsInfo(allBitcodeFiles, imports1Artifact)
            .addInput(bitcode1Artifact)
            .addInput(index1Artifact)
            .addOutput(destinationArtifact)
            .setExecutable(scratch.file("/bin/clang").asFragment())
            .setProgressMessage("Test")
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    LtoBackendAction action = (LtoBackendAction) actions[0];
    assertThat(action.getOwner().getLabel())
        .isEqualTo(ActionsTestUtil.NULL_ACTION_OWNER.getLabel());
    assertThat(action.getInputs()).containsExactly(bitcode1Artifact, index1Artifact);
    assertThat(action.getOutputs()).containsExactly(destinationArtifact);
    assertThat(action.getSpawn().getLocalResources())
        .isEqualTo(AbstractAction.DEFAULT_RESOURCE_SET);
    assertThat(action.getArguments()).containsExactly("/bin/clang");
    assertThat(action.getProgressMessage()).isEqualTo("Test");
    assertThat(action.inputsDiscovered()).isFalse();

    // Discover inputs, which should not add any inputs since bitcode1.imports is empty.
    action.discoverInputs(context);
    assertThat(action.inputsDiscovered()).isTrue();
    assertThat(action.getInputs()).containsExactly(bitcode1Artifact, index1Artifact);
  }

  @Test
  public void testNonEmptyImports() throws Exception {
    Action[] actions =
        new LtoBackendAction.Builder()
            .addImportsInfo(allBitcodeFiles, imports2Artifact)
            .addInput(bitcode2Artifact)
            .addInput(index2Artifact)
            .addOutput(destinationArtifact)
            .setExecutable(scratch.file("/bin/clang").asFragment())
            .setProgressMessage("Test")
            .build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(actions);
    LtoBackendAction action = (LtoBackendAction) actions[0];
    assertThat(action.getOwner().getLabel())
        .isEqualTo(ActionsTestUtil.NULL_ACTION_OWNER.getLabel());
    assertThat(action.getInputs()).containsExactly(bitcode2Artifact, index2Artifact);
    assertThat(action.getOutputs()).containsExactly(destinationArtifact);
    assertThat(action.getSpawn().getLocalResources())
        .isEqualTo(AbstractAction.DEFAULT_RESOURCE_SET);
    assertThat(action.getArguments()).containsExactly("/bin/clang");
    assertThat(action.getProgressMessage()).isEqualTo("Test");
    assertThat(action.inputsDiscovered()).isFalse();

    // Discover inputs, which should add bitcode1.o which is listed in bitcode2.imports.
    action.discoverInputs(context);
    assertThat(action.inputsDiscovered()).isTrue();
    assertThat(action.getInputs())
        .containsExactly(bitcode1Artifact, bitcode2Artifact, index2Artifact);
  }

  private enum KeyAttributes {
    EXECUTABLE,
    IMPORTS_INFO,
    MNEMONIC,
    RUNFILES_SUPPLIER,
    INPUT,
    FIXED_ENVIRONMENT,
    VARIABLE_ENVIRONMENT
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact artifactA = getSourceArtifact("a");
    final Artifact artifactB = getSourceArtifact("b");
    final Artifact artifactAimports = getSourceArtifact("a.imports");
    final Artifact artifactBimports = getSourceArtifact("b.imports");

    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            LtoBackendAction.Builder builder = new LtoBackendAction.Builder();
            builder.addOutput(destinationArtifact);

            PathFragment executable =
                attributesToFlip.contains(KeyAttributes.EXECUTABLE)
                    ? artifactA.getExecPath()
                    : artifactB.getExecPath();
            builder.setExecutable(executable);

            if (attributesToFlip.contains(KeyAttributes.IMPORTS_INFO)) {
              builder.addImportsInfo(new BitcodeFiles(ImmutableMap.of()), artifactAimports);
            } else {
              builder.addImportsInfo(new BitcodeFiles(ImmutableMap.of()), artifactBimports);
            }

            builder.setMnemonic(attributesToFlip.contains(KeyAttributes.MNEMONIC) ? "a" : "b");

            if (attributesToFlip.contains(KeyAttributes.RUNFILES_SUPPLIER)) {
              builder.addRunfilesSupplier(
                  new RunfilesSupplierImpl(
                      PathFragment.create("a"),
                      Runfiles.EMPTY,
                      artifactA,
                      /* buildRunfileLinks= */ false,
                      /* runfileLinksEnabled= */ false));
            } else {
              builder.addRunfilesSupplier(
                  new RunfilesSupplierImpl(
                      PathFragment.create("a"),
                      Runfiles.EMPTY,
                      artifactB,
                      /* buildRunfileLinks= */ false,
                      /* runfileLinksEnabled= */ false));
            }

            if (attributesToFlip.contains(KeyAttributes.INPUT)) {
              builder.addInput(artifactA);
            } else {
              builder.addInput(artifactB);
            }

            Map<String, String> env = new HashMap<>();
            if (attributesToFlip.contains(KeyAttributes.FIXED_ENVIRONMENT)) {
              env.put("foo", "bar");
            }
            builder.setEnvironment(env);
            if (attributesToFlip.contains(KeyAttributes.VARIABLE_ENVIRONMENT)) {
              builder.setInheritedEnvironment(Arrays.asList("baz"));
            }

            Action[] actions = builder.build(ActionsTestUtil.NULL_ACTION_OWNER, targetConfig);
            collectingAnalysisEnvironment.registerAction(actions);
            return actions[0];
          }
        },
        actionKeyContext);
  }
}
