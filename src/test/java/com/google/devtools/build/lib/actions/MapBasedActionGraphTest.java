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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.UncheckedActionConflictException;
import com.google.devtools.build.lib.actions.util.TestAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link MapBasedActionGraph}.
 */
@RunWith(JUnit4.class)
public class MapBasedActionGraphTest {
  private final FileSystem fileSystem = new InMemoryFileSystem();
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  @Test
  public void testSmoke() throws Exception {
    MutableActionGraph actionGraph = new MapBasedActionGraph(actionKeyContext);
    Path execRoot = fileSystem.getPath("/");
    String outSegment = "root";
    Path root = execRoot.getChild(outSegment);
    Path path = root.getRelative("foo");
    Artifact output =
        ActionsTestUtil.createArtifact(ArtifactRoot.asDerivedRoot(execRoot, outSegment), path);
    Action action =
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(output));
    actionGraph.registerAction(action);
    actionGraph.unregisterAction(action);
    path = root.getRelative("bar");
    output = ActionsTestUtil.createArtifact(ArtifactRoot.asDerivedRoot(execRoot, outSegment), path);
    Action action2 =
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(output));
    actionGraph.registerAction(action);
    actionGraph.registerAction(action2);
    actionGraph.unregisterAction(action);
  }

  @Test
  public void testNoActionConflictWhenUnregisteringSharedAction() throws Exception {
    MutableActionGraph actionGraph = new MapBasedActionGraph(actionKeyContext);
    Path execRoot = fileSystem.getPath("/");
    Path root = fileSystem.getPath("/root");
    Path path = root.getRelative("foo");
    Artifact output =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asDerivedRoot(execRoot, root.relativeTo(execRoot).getPathString()), path);
    Action action =
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(output));
    actionGraph.registerAction(action);
    Action otherAction =
        new TestAction(
            TestAction.NO_EFFECT,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(output));
    actionGraph.registerAction(otherAction);
    actionGraph.unregisterAction(action);
  }

  private class ActionRegisterer extends AbstractQueueVisitor {
    private final MutableActionGraph graph = new MapBasedActionGraph(new ActionKeyContext());
    private final Artifact output;
    // Just to occasionally add actions that were already present.
    private final Set<Action> allActions = Sets.newConcurrentHashSet();
    private final AtomicInteger actionCount = new AtomicInteger(0);

    private ActionRegisterer() {
      super(
          200,
          1,
          TimeUnit.SECONDS,
          /*failFastOnException=*/ true,
          "action-graph-test",
          ErrorClassifier.DEFAULT);
      Path execRoot = fileSystem.getPath("/");
      String rootSegment = "root";
      Path root = execRoot.getChild(rootSegment);
      Path path = root.getChild("foo");
      output =
          ActionsTestUtil.createArtifact(ArtifactRoot.asDerivedRoot(execRoot, rootSegment), path);
      allActions.add(
          new TestAction(
              TestAction.NO_EFFECT,
              NestedSetBuilder.emptySet(Order.STABLE_ORDER),
              ImmutableSet.of(output)));
    }

    private void registerAction(final Action action) {
      execute(
          new Runnable() {
            @Override
            public void run() {
              try {
                graph.registerAction(action);
              } catch (ActionConflictException e) {
                throw new UncheckedActionConflictException(e);
              }
              doRandom();
            }
          });
    }

    private void unregisterAction(final Action action) {
      execute(
          new Runnable() {
            @Override
            public void run() {
              graph.unregisterAction(action);
              doRandom();
            }
          });
    }

    private void doRandom() {
      if (actionCount.incrementAndGet() > 10000) {
        return;
      }
      Action action = null;
      if (Math.random() < 0.5) {
        action = Iterables.getFirst(allActions, null);
      } else {
        action =
            new TestAction(
                TestAction.NO_EFFECT,
                NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                ImmutableSet.of(output));
        allActions.add(action);
      }
      if (Math.random() < 0.5) {
        registerAction(action);
      } else {
        unregisterAction(action);
      }
    }

    private void work() throws InterruptedException {
      awaitQuiescence(/*interruptWorkers=*/ true);
    }
  }

  @Test
  public void testSharedActionStressTest() throws Exception {
    ActionRegisterer actionRegisterer = new ActionRegisterer();
    actionRegisterer.doRandom();
    actionRegisterer.work();
  }
}
