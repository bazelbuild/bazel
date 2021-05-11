// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.actions.StarlarkAction;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;

/** Tests for {@link StarlarkAction} using the shadowed action parameter. */
@RunWith(JUnit4.class)
public final class StarlarkActionWithShadowedActionTest extends BuildViewTestCase {

  private ActionExecutionContext executionContext;
  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;
  private NestedSet<Artifact> starlarkActionInputs;
  private NestedSet<Artifact> shadowedActionInputs;
  private NestedSet<Artifact> discoveredInputs;
  private Map<String, String> starlarkActionEnvironment;
  private Map<String, String> shadowedActionEnvironment;

  private Artifact output;
  private PathFragment executable;

  @Before
  public final void createArtifacts() throws Exception {
    collectingAnalysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
    starlarkActionInputs =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            getSourceArtifact("pkg/shadowed_action_inp1"),
            getSourceArtifact("pkg/discovered_inp2"),
            getSourceArtifact("pkg/starlark_action_inp3"));
    shadowedActionInputs =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            getSourceArtifact("pkg/shadowed_action_inp1"),
            getSourceArtifact("pkg/shadowed_action_inp2"),
            getSourceArtifact("pkg/shadowed_action_inp3"));
    discoveredInputs =
        NestedSetBuilder.create(
            Order.STABLE_ORDER,
            getSourceArtifact("pkg/shadowed_action_inp1"),
            getSourceArtifact("pkg/discovered_inp2"),
            getSourceArtifact("pkg/discovered_inp3"));
    output = getBinArtifactWithNoOwner("output");
    executable = scratch.file("/bin/xxx").asFragment();
    starlarkActionEnvironment =
        ImmutableMap.of(
            "repeated_var", "starlark_val",
            "a_var", "a_val",
            "b_var", "b_val");
    shadowedActionEnvironment =
        ImmutableMap.of(
            "repeated_var", "shadowed_val",
            "c_var", "c_val",
            "d_var", "d_val");
  }

  @Before
  public final void createExecutorAndContext() throws Exception {
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    Executor executor = new TestExecutorBuilder(fileSystem, directories, binTools).build();
    executionContext =
        new ActionExecutionContext(
            executor,
            /*actionInputFileCache=*/ null,
            ActionInputPrefetcher.NONE,
            actionKeyContext,
            /*metadataHandler=*/ null,
            /*rewindingEnabled=*/ false,
            LostInputsCheck.NONE,
            /*fileOutErr=*/ null,
            /*eventHandler=*/ null,
            /*clientEnv=*/ ImmutableMap.of(),
            /*topLevelFilesets=*/ ImmutableMap.of(),
            /*artifactExpander=*/ null,
            /*actionFileSystem=*/ null,
            /*skyframeDepsResult=*/ null,
            NestedSetExpander.DEFAULT,
            UnixGlob.DEFAULT_SYSCALLS);
  }

  @Test
  public void testUsingOnlyShadowedActionInputs() throws Exception {
    // If both starlark action and the shadowed action do not have inputs, then getInputs of both of
    // them should return empty set
    Action shadowedAction =
        createShadowedAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER), /*discoversInputs=*/ false, null);
    StarlarkAction starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList()).isEmpty();
    assertThat(starlarkAction.discoversInputs()).isFalse();
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList()).isEmpty();

    // If the starlark action does not have any inputs, then it will use the shadowed action inputs
    shadowedAction = createShadowedAction(shadowedActionInputs, false, null);
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(shadowedActionInputs.toList());
    assertThat(starlarkAction.discoversInputs()).isFalse();
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList())
        .containsExactlyElementsIn(shadowedActionInputs.toList());
  }

  @Test
  public void testUsingOnlyShadowedActionWithDiscoveredInputs() throws Exception {
    // Test that the shadowed action's discovered inputs are passed to the starlark action
    Action shadowedAction =
        createShadowedAction(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /*discoversInputs=*/ true,
            discoveredInputs);
    StarlarkAction starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList()).isEmpty();
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList()).isEmpty();
    assertThat(starlarkAction.discoversInputs()).isTrue();
    assertThat(starlarkAction.discoverInputs(executionContext).toList())
        .containsExactlyElementsIn(discoveredInputs.toList());
    // after discovering inputs, the starlark action inputs should be updated
    assertThat(starlarkAction.inputsDiscovered()).isTrue();
    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(discoveredInputs.toList());

    // Test that both inputs and discovered inputs of the shadowed action are passed to the starlark
    // action
    shadowedAction = createShadowedAction(shadowedActionInputs, true, discoveredInputs);
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(shadowedActionInputs.toList());
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList())
        .containsExactlyElementsIn(shadowedActionInputs.toList());
    assertThat(starlarkAction.discoversInputs()).isTrue();
    assertThat(starlarkAction.discoverInputs(executionContext).toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>difference(discoveredInputs.toSet(), shadowedActionInputs.toSet())
                .toArray());
    // after discovering inputs, the starlark action inputs should be updated
    assertThat(starlarkAction.inputsDiscovered()).isTrue();
    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(shadowedActionInputs.toSet(), discoveredInputs.toSet()).toArray());
  }

  @Test
  public void testUsingShadowedActionWithStarlarkActionInputs() throws Exception {
    // Test using Starlark action's inputs without using a shadowed action
    StarlarkAction starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addInput(starlarkActionInputs.toList().get(1))
                .addInput(starlarkActionInputs.toList().get(2))
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(starlarkActionInputs.toList());
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList())
        .containsExactlyElementsIn(starlarkActionInputs.toList());
    assertThat(starlarkAction.discoversInputs()).isFalse();

    // Test using Starlark actions's inputs with shadowed action's inputs
    Action shadowedAction =
        createShadowedAction(
            shadowedActionInputs, /*discoversInputs=*/ false, /*discoveredInputs=*/ null);
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addInput(starlarkActionInputs.toList().get(1))
                .addInput(starlarkActionInputs.toList().get(2))
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(shadowedActionInputs.toSet(), starlarkActionInputs.toSet())
                .toArray());
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(shadowedActionInputs.toSet(), starlarkActionInputs.toSet())
                .toArray());
    assertThat(starlarkAction.discoversInputs()).isFalse();

    // Test using Starlark actions's inputs with shadowed action's inputs and discovered inputs
    shadowedAction = createShadowedAction(shadowedActionInputs, true, discoveredInputs);
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addInput(starlarkActionInputs.toList().get(1))
                .addInput(starlarkActionInputs.toList().get(2))
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(shadowedActionInputs.toSet(), starlarkActionInputs.toSet())
                .toArray());
    assertThat(starlarkAction.getUnusedInputsList()).isEmpty();
    assertThat(starlarkAction.getAllowedDerivedInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(shadowedActionInputs.toSet(), starlarkActionInputs.toSet())
                .toArray());
    assertThat(starlarkAction.discoversInputs()).isTrue();
    assertThat(starlarkAction.discoverInputs(executionContext).toList())
        .containsExactly(discoveredInputs.toList().get(2));
    // after discovering inputs, the starlark action inputs should be updated
    assertThat(starlarkAction.inputsDiscovered()).isTrue();
    assertThat(starlarkAction.getInputs().toList())
        .containsExactlyElementsIn(
            Sets.<Artifact>union(
                    NestedSetBuilder.wrap(
                            Order.STABLE_ORDER,
                            Sets.<Artifact>union(
                                shadowedActionInputs.toSet(), starlarkActionInputs.toSet()))
                        .toSet(),
                    discoveredInputs.toSet())
                .toArray());
  }

  @Test
  public void testPassingShadowedActionEnvironment() throws Exception {
    // Test using Starlark action's environment without using a shadowed action
    StarlarkAction starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addOutput(output)
                .setEnvironment(starlarkActionEnvironment)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getEffectiveEnvironment(ImmutableMap.of()))
        .containsExactlyEntriesIn(starlarkActionEnvironment);

    // Test using shadowed action's environment without Starlark actions's environment
    Action shadowedAction =
        createShadowedAction(
            shadowedActionInputs, /*discoversInputs=*/ false, /*discoveredInputs=*/ null);
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    assertThat(starlarkAction.getEffectiveEnvironment(ImmutableMap.of()))
        .containsExactlyEntriesIn(shadowedActionEnvironment);

    // Test using Starlark actions's environment with shadowed action's environment
    starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(shadowedAction))
                .setExecutable(executable)
                .addInput(starlarkActionInputs.toList().get(0))
                .addOutput(output)
                .setEnvironment(starlarkActionEnvironment)
                .build(NULL_ACTION_OWNER, targetConfig);
    collectingAnalysisEnvironment.registerAction(starlarkAction);

    LinkedHashMap<String, String> expectedEnvironment = new LinkedHashMap<>();
    expectedEnvironment.putAll(shadowedActionEnvironment);
    expectedEnvironment.putAll(starlarkActionEnvironment);

    ImmutableMap<String, String> actualEnvironment =
        starlarkAction.getEffectiveEnvironment(ImmutableMap.of());
    assertThat(actualEnvironment).hasSize(5);
    // Starlark action's env overwrites any repeated variable from the shadowed action env
    assertThat(actualEnvironment).containsEntry("repeated_var", "starlark_val");
    assertThat(actualEnvironment).containsExactlyEntriesIn(expectedEnvironment);
  }

  private Action createShadowedAction(
      NestedSet<Artifact> inputs, boolean discoversInputs, NestedSet<Artifact> discoveredInputs)
      throws Exception {
    Action shadowedAction = mock(Action.class);
    when(shadowedAction.discoversInputs()).thenReturn(discoversInputs);
    when(shadowedAction.getInputs()).thenReturn(inputs);
    when(shadowedAction.getAllowedDerivedInputs()).thenReturn(inputs);
    when(shadowedAction.getInputFilesForExtraAction(
            ArgumentMatchers.any(ActionExecutionContext.class)))
        .thenReturn(discoveredInputs);
    when(shadowedAction.inputsDiscovered()).thenReturn(true);
    when(shadowedAction.getOwner()).thenReturn(NULL_ACTION_OWNER);
    when(shadowedAction.getEffectiveEnvironment(ArgumentMatchers.anyMap()))
        .thenReturn(ImmutableMap.copyOf(shadowedActionEnvironment));

    return shadowedAction;
  }
}
