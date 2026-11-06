// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that the inputs of an action that are also among its scheduling dependencies are not
 * requested from Skyframe individually, which would add an edge per action and input. See
 * https://github.com/bazelbuild/bazel/issues/32456.
 */
@RunWith(JUnit4.class)
public final class SchedulingDependencyInputsTest extends TimestampBuilderTestCase {

  /** The most recently recorded direct Skyframe dependencies of each action. */
  private final Map<SkyKey, ImmutableSet<SkyKey>> actionDeps = new ConcurrentHashMap<>();

  private final EvaluationProgressReceiver actionDepsRecorder =
      new EvaluationProgressReceiver() {
        @Override
        public void evaluated(
            SkyKey skyKey,
            EvaluationState state,
            @Nullable SkyValue newValue,
            @Nullable ErrorInfo newError,
            @Nullable GroupedDeps directDeps) {
          if (skyKey.functionName().equals(SkyFunctions.ACTION_EXECUTION) && directDeps != null) {
            actionDeps.put(skyKey, ImmutableSet.copyOf(directDeps.getAllElementsAsIterable()));
          }
        }
      };

  @Test
  public void discoveredSchedulingDependencies_notRequestedIndividually() throws Exception {
    Artifact first = createSourceFile("first.h");
    Artifact second = createSourceFile("second.h");
    NestedSet<Artifact> sharedHeaders = NestedSetBuilder.create(Order.STABLE_ORDER, first, second);
    Artifact oneHeader = createSourceFile("one.h");
    Artifact twoHeader = createSourceFile("two.h");
    Artifact one = createDerivedArtifact("one");
    Artifact two = createDerivedArtifact("two");
    AtomicInteger oneExecutions = new AtomicInteger();
    AtomicInteger twoExecutions = new AtomicInteger();
    registerAction(
        new SchedulingDependencyDiscoveringAction(
            oneExecutions::incrementAndGet, sharedHeaders, oneHeader, one));
    registerAction(
        new SchedulingDependencyDiscoveringAction(
            twoExecutions::incrementAndGet, sharedHeaders, twoHeader, two));

    BuilderWithResult builder =
        createBuilder(
            inMemoryCache, /* threadCount= */ 1, /* keepGoing= */ false, actionDepsRecorder);
    buildArtifacts(builder, one, two);

    assertThat(oneExecutions.get()).isEqualTo(1);
    assertThat(twoExecutions.get()).isEqualTo(1);
    for (Artifact output : ImmutableList.of(one, two)) {
      ImmutableSet<SkyKey> deps = actionDeps.get(Artifact.key(output));
      assertThat(deps).contains(ArtifactNestedSetKey.create(sharedHeaders));
      assertThat(deps).containsNoneOf(Artifact.key(first), Artifact.key(second));
    }
    // Discovered inputs that aren't scheduling dependencies are still requested individually.
    assertThat(actionDeps.get(Artifact.key(one))).contains(Artifact.key(oneHeader));
    assertThat(actionDeps.get(Artifact.key(two))).contains(Artifact.key(twoHeader));

    // Without the direct edges, a change to a discovered input that is a scheduling dependency
    // must still invalidate the actions in the same Skyframe graph.
    modifySourceFile(first);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(2);
    assertThat(twoExecutions.get()).isEqualTo(2);

    modifySourceFile(oneHeader);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(3);
    assertThat(twoExecutions.get()).isEqualTo(2);

    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(3);
    assertThat(twoExecutions.get()).isEqualTo(2);
  }

  @Test
  public void schedulingDependenciesFromActionCache_notRequestedIndividually() throws Exception {
    // The action cache records discovered derived inputs by exec path and resolves them via
    // Action#getAllowedDerivedInputs, so this test uses generated headers.
    artifactResolver = new ArtifactFactory(outputBase.getRelative("execroot"), "out");
    Artifact first = createDerivedArtifact("first.h");
    Artifact second = createDerivedArtifact("second.h");
    registerAction(
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(first)));
    registerAction(
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(second)));
    NestedSet<Artifact> sharedHeaders = NestedSetBuilder.create(Order.STABLE_ORDER, first, second);
    Artifact one = createDerivedArtifact("one");
    AtomicInteger executions = new AtomicInteger();
    SchedulingDependencyDiscoveringAction action =
        registerAction(
            new SchedulingDependencyDiscoveringAction(
                executions::incrementAndGet, sharedHeaders, /* otherDiscoveredInput= */ null, one));

    buildArtifacts(cachingBuilder(), one);
    assertThat(executions.get()).isEqualTo(1);

    // Simulate a server restart: the action is analyzed again and starts out with unknown inputs,
    // but the action cache still knows about the inputs it discovered.
    action.resetDiscoveredInputs();
    buildArtifacts(
        createBuilder(
            inMemoryCache, /* threadCount= */ 1, /* keepGoing= */ false, actionDepsRecorder),
        one);

    assertThat(executions.get()).isEqualTo(1); // action cache hit
    ImmutableSet<SkyKey> deps = actionDeps.get(Artifact.key(one));
    assertThat(deps).contains(ArtifactNestedSetKey.create(sharedHeaders));
    assertThat(deps).containsNoneOf(Artifact.key(first), Artifact.key(second));
  }

  private Artifact createSourceFile(String name) throws IOException {
    Artifact artifact = createSourceArtifact(name);
    artifact.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(artifact.getPath(), "initial");
    return artifact;
  }

  private void modifySourceFile(Artifact artifact) throws IOException {
    FileSystemUtils.writeContentAsLatin1(artifact.getPath(), "modified");
    differencer.invalidate(
        ImmutableList.of(
            FileStateValue.key(
                RootedPath.toRootedPath(
                    artifact.getRoot().getRoot(), artifact.getRootRelativePath()))));
  }

  /**
   * Like a C++ compilation without include scanning, discovers all its scheduling dependencies as
   * inputs, plus optionally an input that isn't a scheduling dependency.
   */
  private static final class SchedulingDependencyDiscoveringAction extends TestAction {
    private final NestedSet<Artifact> schedulingDependencies;
    private final NestedSet<Artifact> discoveredInputs;

    SchedulingDependencyDiscoveringAction(
        Runnable effect,
        NestedSet<Artifact> schedulingDependencies,
        @Nullable Artifact otherDiscoveredInput,
        Artifact output) {
      super(effect, NestedSetBuilder.emptySet(Order.STABLE_ORDER), ImmutableSet.of(output));
      this.schedulingDependencies = schedulingDependencies;
      NestedSetBuilder<Artifact> discoveredInputs =
          NestedSetBuilder.<Artifact>stableOrder().addTransitive(schedulingDependencies);
      if (otherDiscoveredInput != null) {
        discoveredInputs.add(otherDiscoveredInput);
      }
      this.discoveredInputs = discoveredInputs.build();
    }

    @Override
    public boolean discoversInputs() {
      return true;
    }

    @Override
    public NestedSet<Artifact> getSchedulingDependencies() {
      return schedulingDependencies;
    }

    @Override
    public NestedSet<Artifact> getAllowedDerivedInputs() {
      return discoveredInputs;
    }

    @Override
    public NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
      updateInputs(
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(getMandatoryInputs())
              .addTransitive(discoveredInputs)
              .build());
      return discoveredInputs;
    }
  }
}
