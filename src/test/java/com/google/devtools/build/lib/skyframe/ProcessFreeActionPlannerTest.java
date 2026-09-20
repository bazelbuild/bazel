// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionContext.ActionContextRegistry;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ProcessFreeActionContextRegistry;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Root;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProcessFreeActionPlannerTest {
  private final Scratch scratch = new Scratch();
  private final Map<Artifact, ActionAnalysisMetadata> generatingActions = new IdentityHashMap<>();
  private ArtifactRoot sourceRoot;
  private ArtifactRoot outputRoot;
  private int nextActionIndex;

  @Before
  public void setUp() throws Exception {
    sourceRoot = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/workspace")));
    outputRoot =
        ArtifactRoot.asDerivedRoot(scratch.dir("/execroot"), RootType.OUTPUT, "bazel-out");
  }

  @Test
  public void selectsOnlySourceRootedProcessFreeActions() {
    Artifact source = ActionsTestUtil.createArtifact(sourceRoot, "source.txt");
    DerivedArtifact generatedMetadata = output("generated-metadata");
    ActionLookupData metadataKey =
        addAction(new TestAction(ImmutableList.of(source), generatedMetadata, true));
    DerivedArtifact compiled = output("compiled");
    addAction(new TestAction(ImmutableList.of(generatedMetadata), compiled, false));
    DerivedArtifact downstreamMetadata = output("downstream-metadata");
    addAction(new TestAction(ImmutableList.of(compiled), downstreamMetadata, true));

    DerivedArtifact independentMetadata = output("independent-metadata");
    ActionLookupData independentKey =
        addAction(new TestAction(ImmutableList.of(source), independentMetadata, true));

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(downstreamMetadata, independentMetadata),
            generatingActions::get,
            /* collectInventory= */ true);

    assertThat(plan.actionKeys()).containsExactly(metadataKey, independentKey).inOrder();
    assertThat(plan.visitedActions()).isEqualTo(4);
    assertThat(plan.deferredActions()).isEqualTo(2);
    assertThat(
            plan.actionInventory().stream()
                .filter(action -> action.state().equals("selected"))
                .count())
        .isEqualTo(2);
    assertThat(
            plan.actionInventory().stream()
                .filter(action -> action.state().equals("blocked"))
                .count())
        .isEqualTo(1);
    assertThat(
            plan.actionInventory().stream()
                .filter(action -> action.state().equals("deferred"))
                .count())
        .isEqualTo(1);
  }

  @Test
  public void unknownDerivedInputDefersOtherwiseProcessFreeAction() {
    DerivedArtifact unknown = output("unknown");
    DerivedArtifact metadata = output("metadata");
    addAction(new TestAction(ImmutableList.of(unknown), metadata, true));

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(metadata), generatingActions::get, /* collectInventory= */ false);

    assertThat(plan.actionKeys()).isEmpty();
    assertThat(plan.visitedActions()).isEqualTo(1);
    assertThat(plan.deferredActions()).isEqualTo(1);
    assertThat(plan.unresolvedArtifacts()).isEqualTo(1);
    assertThat(plan.actionInventory()).isEmpty();
  }

  @Test
  public void sharedUnknownDerivedInputDefersEveryConsumer() {
    DerivedArtifact unknown = output("unknown");
    DerivedArtifact first = output("first");
    DerivedArtifact second = output("second");
    addAction(new TestAction(ImmutableList.of(unknown), first, true));
    addAction(new TestAction(ImmutableList.of(unknown), second, true));

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(first, second), generatingActions::get, /* collectInventory= */ false);

    assertThat(plan.actionKeys()).isEmpty();
    assertThat(plan.visitedActions()).isEqualTo(2);
    assertThat(plan.deferredActions()).isEqualTo(2);
  }

  @Test
  public void actionCycleFailsClosed() {
    DerivedArtifact first = output("first");
    DerivedArtifact second = output("second");
    addAction(new TestAction(ImmutableList.of(second), first, true));
    addAction(new TestAction(ImmutableList.of(first), second, true));

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(first), generatingActions::get, /* collectInventory= */ false);

    assertThat(plan.actionKeys()).isEmpty();
    assertThat(plan.visitedActions()).isEqualTo(2);
    assertThat(plan.deferredActions()).isEqualTo(2);
  }

  @Test
  public void deepActionChainDoesNotUseTheJavaCallStack() {
    Artifact input = ActionsTestUtil.createArtifact(sourceRoot, "source.txt");
    int actionCount = 20_000;
    for (int i = 0; i < actionCount; i++) {
      DerivedArtifact output = output("output-" + i);
      addAction(new TestAction(ImmutableList.of(input), output, true));
      input = output;
    }

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(input), generatingActions::get, /* collectInventory= */ false);

    assertThat(plan.actionKeys()).hasSize(actionCount);
    assertThat(plan.visitedActions()).isEqualTo(actionCount);
    assertThat(plan.deferredActions()).isEqualTo(0);
  }

  @Test
  public void processFreeInputDiscoveringActionRemainsDeferred() {
    Artifact source = ActionsTestUtil.createArtifact(sourceRoot, "source.txt");
    DerivedArtifact discovered = output("discovered");
    addAction(new InputDiscoveringAction(ImmutableList.of(source), discovered));

    ProcessFreeActionPlanner.Plan plan =
        ProcessFreeActionPlanner.create(
            ImmutableList.of(discovered), generatingActions::get, /* collectInventory= */ true);

    assertThat(plan.actionKeys()).isEmpty();
    assertThat(plan.visitedActions()).isEqualTo(1);
    assertThat(plan.deferredActions()).isEqualTo(1);
    assertThat(plan.actionInventory().getFirst().state()).isEqualTo("blocked");
  }

  @Test
  public void processFreeContextRegistryAllowsOnlyDeclaredContexts() {
    DerivedArtifact output = output("metadata");
    TestAction action =
        new TestAction(ImmutableList.of(), output, true, ImmutableSet.of(AllowedContext.class));
    AtomicInteger delegateLookups = new AtomicInteger();
    AllowedContext allowed = new AllowedContext();
    ActionContextRegistry delegate =
        new ActionContextRegistry() {
          @Override
          public <T extends ActionContext> T getContext(Class<T> identifyingType) {
            delegateLookups.incrementAndGet();
            return identifyingType.cast(allowed);
          }
        };
    ActionContextRegistry restricted = new ProcessFreeActionContextRegistry(action, delegate);

    assertThat(restricted.getContext(AllowedContext.class)).isSameInstanceAs(allowed);
    assertThat(delegateLookups.get()).isEqualTo(1);
  }

  @Test
  public void misclassifiedProcessFreeActionCannotReachExecutionContext() {
    DerivedArtifact output = output("metadata");
    TestAction action = new TestAction(ImmutableList.of(), output, true);
    AtomicInteger delegateLookups = new AtomicInteger();
    ActionContextRegistry delegate =
        new ActionContextRegistry() {
          @Override
          public <T extends ActionContext> T getContext(Class<T> identifyingType) {
            delegateLookups.incrementAndGet();
            return null;
          }
        };
    ActionContextRegistry restricted = new ProcessFreeActionContextRegistry(action, delegate);

    IllegalStateException thrown =
        org.junit.Assert.assertThrows(
            IllegalStateException.class, () -> restricted.getContext(ProcessContext.class));

    assertThat(thrown)
        .hasMessageThat()
        .contains("requested undeclared execution context");
    assertThat(delegateLookups.get()).isEqualTo(0);
  }

  @Test
  public void materializationEventReportsTypedResult() throws Exception {
    ProcessFreeMaterializationEvent event =
        new ProcessFreeMaterializationEvent(
            /* selectedActions= */ 5,
            /* materializedActions= */ 4,
            /* deferredActions= */ 3,
            /* visitedActions= */ 8,
            /* unresolvedArtifacts= */ 2,
            /* success= */ false);

    BuildEventStreamProtos.BuildEvent proto = event.asStreamProto(/* converters= */ null);

    assertThat(proto.getId()).isEqualTo(BuildEventIdUtil.processFreeMaterializationResultId());
    assertThat(proto.getProcessFreeMaterializationResult())
        .isEqualTo(
            BuildEventStreamProtos.ProcessFreeMaterializationResult.newBuilder()
                .setSelectedActionCount(5)
                .setMaterializedActionCount(4)
                .setDeferredActionCount(3)
                .setVisitedActionCount(8)
                .setUnresolvedArtifactCount(2)
                .setSuccess(false)
                .build());
  }

  private DerivedArtifact output(String path) {
    return (DerivedArtifact) ActionsTestUtil.createArtifact(outputRoot, path);
  }

  private ActionLookupData addAction(ActionAnalysisMetadata action) {
    ActionLookupData key =
        ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, nextActionIndex++);
    for (Artifact output : action.getOutputs()) {
      ((DerivedArtifact) output).setGeneratingActionKey(key);
      generatingActions.put(output, action);
    }
    return key;
  }

  private static final class TestAction extends ActionsTestUtil.NullAction {
    private final boolean processFree;
    private final ImmutableSet<Class<? extends ActionContext>> allowedContexts;

    TestAction(List<Artifact> inputs, Artifact output, boolean processFree) {
      this(inputs, output, processFree, ImmutableSet.of());
    }

    TestAction(
        List<Artifact> inputs,
        Artifact output,
        boolean processFree,
        ImmutableSet<Class<? extends ActionContext>> allowedContexts) {
      super(inputs, output);
      this.processFree = processFree;
      this.allowedContexts = allowedContexts;
    }

    @Override
    public boolean isProcessFree() {
      return processFree;
    }

    @Override
    public ImmutableSet<Class<? extends ActionContext>> getProcessFreeActionContexts() {
      return allowedContexts;
    }
  }

  private static final class AllowedContext implements ActionContext {}

  private static final class ProcessContext implements ActionContext {}

  private static final class InputDiscoveringAction extends ActionsTestUtil.NullAction {
    InputDiscoveringAction(List<Artifact> inputs, Artifact output) {
      super(inputs, output);
    }

    @Override
    public boolean discoversInputs() {
      return true;
    }

    @Override
    public boolean isProcessFree() {
      return true;
    }
  }
}
