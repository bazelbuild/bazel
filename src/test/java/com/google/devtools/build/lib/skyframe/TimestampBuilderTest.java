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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.ArtifactNestedSetKey;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
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
 * Test suite for TimestampBuilder.
 *
 */
@RunWith(JUnit4.class)
public class TimestampBuilderTest extends TimestampBuilderTestCase {
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

  private static NestedSet<Artifact> asNestedSet(Artifact... artifacts) {
    return NestedSetBuilder.create(Order.STABLE_ORDER, artifacts);
  }

  @Test
  public void testAmnesiacBuilderAlwaysRebuilds() throws Exception {
    // [action] -> hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptyNestedSet, ImmutableSet.of(hello));

    button.pressed = false;
    buildArtifacts(amnesiacBuilder(), hello);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(amnesiacBuilder(), hello);
    assertThat(button.pressed).isTrue(); // rebuilt
  }

  // If we re-use the same builder (even an "amnesiac" builder), it remembers
  // which Actions it has already visited, and doesn't revisit them, even if
  // they would otherwise be rebuilt.
  //
  // That is, Builders conflate traversal and dependency analysis, and don't
  // revisit a node (traversal) even if it needs to be rebuilt (dependency
  // analysis).  We might want to separate these aspects.
  @Test
  public void testBuilderDoesntRevisitActions() throws Exception {
    // [action] -> hello
    Artifact hello = createDerivedArtifact("hello");
    Counter counter = createActionCounter(emptyNestedSet, ImmutableSet.of(hello));

    Builder amnesiacBuilder = amnesiacBuilder();

    counter.count = 0;
    buildArtifacts(amnesiacBuilder, hello, hello);
    assertThat(counter.count).isEqualTo(1); // built only once
  }

  @Test
  public void testBuildingExistingSourcefileSuceeds() throws Exception {
    Artifact hello = createSourceArtifact("hello");
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    buildArtifacts(cachingBuilder(), hello);
  }

  @Test
  public void testCachingBuilderCachesUntilReset() throws Exception {
    // [action] -> hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptyNestedSet, ImmutableSet.of(hello));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isFalse(); // not rebuilt

    inMemoryCache.reset();

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isTrue(); // rebuilt
  }

  @Test
  public void testUnneededInputs() throws Exception {
    Artifact hello = createSourceArtifact("hello");
    hello.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");
    Artifact optional = createSourceArtifact("hello.optional");
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(asNestedSet(hello, optional), ImmutableSet.of(goodbye));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt

    BlazeTestUtils.makeEmptyFile(optional.getPath());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content2");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt

    optional.getPath().delete();
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content3");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt
  }

  @Test
  public void testModifyingInputCausesActionReexecution() throws Exception {
    // hello -> [action] -> goodbye
    Artifact hello = createSourceArtifact("hello");
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(asNestedSet(hello), ImmutableSet.of(goodbye));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt

    hello.getPath().setWritable(true);
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "new content");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // rebuilt

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt
  }

  @Test
  public void testOnlyModifyingInputContentCausesReexecution() throws Exception {
    // hello -> [action] -> goodbye
    Artifact hello = createSourceArtifact("hello");
    // touch file to create the directory structure
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content1");

    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button = createActionButton(asNestedSet(hello), ImmutableSet.of(goodbye));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt

    FileSystemUtils.touchFile(hello.getPath());

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // still not rebuilt

    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "content2");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isTrue(); // rebuilt

    button.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button.pressed).isFalse(); // not rebuilt
  }

  @Test
  public void testModifyingOutputCausesActionReexecution() throws Exception {
    // [action] -> hello
    Artifact hello = createDerivedArtifact("hello");
    Button button = createActionButton(emptyNestedSet, ImmutableSet.of(hello));

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isTrue(); // built

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isFalse(); // not rebuilt

    // Changing the *output* file 'hello' causes 'action' to re-execute, to make things consistent
    // again.
    hello.getPath().setWritable(true);
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "new content");

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isTrue(); // rebuilt

    button.pressed = false;
    buildArtifacts(cachingBuilder(), hello);
    assertThat(button.pressed).isFalse(); // not rebuilt
  }

  @Test
  public void testBuildingTransitivePrerequisites() throws Exception {
    // hello -> [action1] -> wazuup -> [action2] -> goodbye
    Artifact hello = createSourceArtifact("hello");
    BlazeTestUtils.makeEmptyFile(hello.getPath());
    Artifact wazuup = createDerivedArtifact("wazuup");
    Button button1 = new Button();
    registerAction(new CopyingAction(button1, hello, wazuup));
    Artifact goodbye = createDerivedArtifact("goodbye");
    Button button2 = createActionButton(asNestedSet(wazuup), ImmutableSet.of(goodbye));

    button1.pressed = button2.pressed = false;
    buildArtifacts(cachingBuilder(), wazuup);
    assertThat(button1.pressed).isTrue(); // built wazuup
    assertThat(button2.pressed).isFalse(); // goodbye not built

    button1.pressed = button2.pressed = false;
    buildArtifacts(cachingBuilder(), wazuup);
    assertThat(button1.pressed).isFalse(); // wazuup not rebuilt
    assertThat(button2.pressed).isFalse(); // goodbye not built

    button1.pressed = button2.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button1.pressed).isFalse(); // wazuup not rebuilt
    assertThat(button2.pressed).isTrue(); // built goodbye

    button1.pressed = button2.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button1.pressed).isFalse(); // wazuup not rebuilt
    assertThat(button2.pressed).isFalse(); // goodbye not rebuilt

    hello.getPath().setWritable(true);
    FileSystemUtils.writeContentAsLatin1(hello.getPath(), "new content");

    button1.pressed = button2.pressed = false;
    buildArtifacts(cachingBuilder(), goodbye);
    assertThat(button1.pressed).isTrue(); // hello rebuilt
    assertThat(button2.pressed).isTrue(); // goodbye rebuilt
  }

  @Test
  public void testWillNotRebuildActionsWithEmptyListOfInputsSpuriously()
      throws Exception {

    Artifact anOutputFile = createDerivedArtifact("anOutputFile");
    Artifact anotherOutputFile = createDerivedArtifact("anotherOutputFile");

    Button aButton = createActionButton(emptyNestedSet, ImmutableSet.of(anOutputFile));
    Button anotherButton = createActionButton(emptyNestedSet, ImmutableSet.of(anotherOutputFile));

    buildArtifacts(cachingBuilder(), anOutputFile, anotherOutputFile);

    assertThat(aButton.pressed).isTrue();
    assertThat(anotherButton.pressed).isTrue();

    aButton.pressed = anotherButton.pressed = false;

    buildArtifacts(cachingBuilder(), anOutputFile, anotherOutputFile);

    assertThat(aButton.pressed).isFalse();
    assertThat(anotherButton.pressed).isFalse();
  }

  @Test
  public void testMissingSourceFileIsAnError() {
    // A missing input to an action must be treated as an error because there's
    // a risk that the action that consumes it will succeed, but with a
    // different behavior (imagine that it globs over the directory, for
    // example).  It's not ok to simply try the action and let the action
    // report "input file not found".
    //
    // (However, there are exceptions to this principle: C++ compilation
    // actions may depend on non-existent headers from stale .d files.  We need
    // to allow the action to proceed to execution in this case.)

    reporter.removeHandler(failFastHandler);
    // doesn't exist
    Artifact in =
        new Artifact.SourceArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fileSystem.getPath("/src"))),
            PathFragment.create("in/in"),
            () -> Label.parseCanonicalUnchecked("//in:in"));
    Artifact out = createDerivedArtifact("out");

    registerAction(new TestAction(TestAction.NO_EFFECT, asNestedSet(in), ImmutableSet.of(out)));

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildArtifacts(amnesiacBuilder(), out));
    assertThat(e).hasMessageThat().contains("1 input file(s) do not exist");
  }

  /**
   * Discovered inputs must be requested from Skyframe via the direct children of their nested set,
   * so that a nested set shared between actions doesn't add an edge per action and input. See
   * https://github.com/bazelbuild/bazel/issues/32456.
   */
  @Test
  public void discoveredInputs_requestedViaDirectChildren() throws Exception {
    // Like the scheduling dependencies of a C++ compilation, which contain the headers of the
    // target itself and the shared nested sets of headers of its dependencies.
    Artifact ownHeader = createSourceFile("own.h");
    Artifact depFirst = createSourceFile("dep_first.h");
    Artifact depSecond = createSourceFile("dep_second.h");
    NestedSet<Artifact> depHeaders =
        NestedSetBuilder.create(Order.STABLE_ORDER, depFirst, depSecond);
    NestedSet<Artifact> schedulingDependencies =
        NestedSetBuilder.<Artifact>stableOrder().add(ownHeader).addTransitive(depHeaders).build();
    Artifact oneHeader = createSourceFile("one.h");
    Artifact oneNestedFirst = createSourceFile("one_nested_first.h");
    Artifact oneNestedSecond = createSourceFile("one_nested_second.h");
    NestedSet<Artifact> oneNestedHeaders =
        NestedSetBuilder.create(Order.STABLE_ORDER, oneNestedFirst, oneNestedSecond);
    Artifact one = createDerivedArtifact("one");
    Artifact two = createDerivedArtifact("two");
    AtomicInteger oneExecutions = new AtomicInteger();
    AtomicInteger twoExecutions = new AtomicInteger();
    // The discovered inputs of "one" consist of its scheduling dependencies, another nested set and
    // a single artifact, while those of "two" are exactly its scheduling dependencies.
    registerAction(
        new SchedulingDependencyDiscoveringAction(
            oneExecutions::incrementAndGet,
            schedulingDependencies,
            NestedSetBuilder.<Artifact>stableOrder()
                .addTransitive(schedulingDependencies)
                .addTransitive(oneNestedHeaders)
                .add(oneHeader)
                .build(),
            one));
    registerAction(
        new SchedulingDependencyDiscoveringAction(
            twoExecutions::incrementAndGet, schedulingDependencies, schedulingDependencies, two));

    BuilderWithResult builder =
        createBuilder(
            inMemoryCache, /* threadCount= */ 1, /* keepGoing= */ false, actionDepsRecorder);
    buildArtifacts(builder, one, two);

    assertThat(oneExecutions.get()).isEqualTo(1);
    assertThat(twoExecutions.get()).isEqualTo(1);
    ImmutableSet<SkyKey> oneDeps = actionDeps.get(Artifact.key(one));
    assertThat(oneDeps)
        .containsAtLeast(
            ArtifactNestedSetKey.create(schedulingDependencies),
            ArtifactNestedSetKey.create(oneNestedHeaders),
            Artifact.key(oneHeader));
    assertThat(oneDeps)
        .containsNoneOf(
            Artifact.key(ownHeader),
            Artifact.key(depFirst),
            Artifact.key(depSecond),
            Artifact.key(oneNestedFirst),
            Artifact.key(oneNestedSecond));
    ImmutableSet<SkyKey> twoDeps = actionDeps.get(Artifact.key(two));
    assertThat(twoDeps).contains(ArtifactNestedSetKey.create(schedulingDependencies));
    assertThat(twoDeps)
        .containsNoneOf(
            Artifact.key(ownHeader),
            Artifact.key(depFirst),
            Artifact.key(depSecond),
            ArtifactNestedSetKey.create(depHeaders));

    // Without the direct edges, a change to a discovered input beneath a nested set must still
    // invalidate the actions in the same Skyframe graph.
    modifySourceFile(depFirst);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(2);
    assertThat(twoExecutions.get()).isEqualTo(2);

    modifySourceFile(ownHeader);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(3);
    assertThat(twoExecutions.get()).isEqualTo(3);

    modifySourceFile(oneHeader);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(4);
    assertThat(twoExecutions.get()).isEqualTo(3);

    modifySourceFile(oneNestedSecond);
    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(5);
    assertThat(twoExecutions.get()).isEqualTo(3);

    buildArtifacts(builder, one, two);
    assertThat(oneExecutions.get()).isEqualTo(5);
    assertThat(twoExecutions.get()).isEqualTo(3);
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
   * Like a C++ compilation without include scanning, discovers the given inputs, which contain its
   * scheduling dependencies.
   */
  private static final class SchedulingDependencyDiscoveringAction extends TestAction {
    private final NestedSet<Artifact> schedulingDependencies;
    private final NestedSet<Artifact> discoveredInputs;

    SchedulingDependencyDiscoveringAction(
        Runnable effect,
        NestedSet<Artifact> schedulingDependencies,
        NestedSet<Artifact> discoveredInputs,
        Artifact output) {
      super(effect, NestedSetBuilder.emptySet(Order.STABLE_ORDER), ImmutableSet.of(output));
      this.schedulingDependencies = schedulingDependencies;
      this.discoveredInputs = discoveredInputs;
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
