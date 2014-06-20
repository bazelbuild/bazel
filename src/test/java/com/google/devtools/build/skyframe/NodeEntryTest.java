// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.truth0.Truth.ASSERT;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.skyframe.BuildingState.ContinueGroup;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Tests for {@link NodeEntry}.
 */
@RunWith(JUnit4.class)
public class NodeEntryTest {

  private static final NodeType NODE_TYPE = new NodeType("Type", false);
  private static final NestedSet<TaggedEvents> NO_EVENTS =
      NestedSetBuilder.<TaggedEvents>emptySet(Order.STABLE_ORDER);

  private static NodeKey key(String name) {
    return new NodeKey(NODE_TYPE, name);
  }

  @Test
  public void createEntry() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    assertFalse(entry.isDirty());
    assertFalse(entry.isChanged());
    ASSERT.that(entry.getTemporaryDirectDeps()).isEmpty();
  }

  @Test
  public void signalEntry() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep1 = key("dep1");
    entry.addTemporaryDirectDep(dep1, ContinueGroup.FALSE);
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep1);
    NodeKey dep2 = key("dep2");
    NodeKey dep3 = key("dep3");
    entry.addTemporaryDirectDep(dep2, ContinueGroup.FALSE);
    entry.addTemporaryDirectDep(dep3, ContinueGroup.FALSE);
    assertFalse(entry.isReady());
    assertFalse(entry.signalDep());
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    MoreAsserts.assertEmpty(setValue(entry, new Node() {},
        /*errorInfo=*/null, /*graphVersion=*/0L));
    assertTrue(entry.isDone());
    assertEquals(0L, entry.getVersion());
    MoreAsserts.assertContentsAnyOrder(entry.getDirectDeps(), dep1, dep2, dep3);
  }

  @Test
  public void reverseDeps() {
    NodeEntry entry = new NodeEntry();
    NodeKey mother = key("mother");
    NodeKey father = key("father");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(mother));
    assertEquals(DependencyState.ADDED_DEP, entry.addReverseDepAndCheckIfDone(null));
    assertEquals(DependencyState.ADDED_DEP, entry.addReverseDepAndCheckIfDone(father));
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new Node() {},
        /*errorInfo=*/null, /*graphVersion=*/0L),
        mother, father);
    MoreAsserts.assertContentsAnyOrder(entry.getReverseDeps(), mother, father);
    assertTrue(entry.isDone());
    entry.removeReverseDep(mother);
    assertFalse(Iterables.contains(entry.getReverseDeps(), mother));
  }

  @Test
  public void errorNode() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    GenericNodeBuilderException exception =
        new GenericNodeBuilderException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    MoreAsserts.assertEmpty(setValue(entry, /*node=*/null, errorInfo, /*graphVersion=*/0L));
    assertTrue(entry.isDone());
    assertNull(entry.getNode());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void errorAndValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    GenericNodeBuilderException exception =
        new GenericNodeBuilderException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    setValue(entry, new Node() {}, errorInfo, /*graphVersion=*/0L);
    assertTrue(entry.isDone());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void crashOnNullErrorAndValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    try {
      setValue(entry, /*node=*/null, /*errorInfo=*/null, /*graphVersion=*/0L);
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnTooManySignals() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    try {
      entry.signalDep();
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnDifferentValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    try {
      // Node() {} and Node() {} are not .equals().
      setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/1L);
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void dirtyLifecycle() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    ASSERT.that(entry.getTemporaryDirectDeps()).isEmpty();
    NodeKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep();
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    assertTrue(entry.isReady());
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new Node() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L), parent);
  }

  @Test
  public void changedLifecycle() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/true);
    assertTrue(entry.isDirty());
    assertTrue(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    NodeKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    assertTrue(entry.isReady());
    ASSERT.that(entry.getTemporaryDirectDeps()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new Node() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L), parent);
    assertEquals(1L, entry.getVersion());
  }

  @Test
  public void markDirtyThenChanged() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.addTemporaryDirectDep(key("dep"), ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    entry.markDirty(/*isChanged=*/true);
    assertTrue(entry.isDirty());
    assertTrue(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
  }


  @Test
  public void markChangedThenDirty() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.addTemporaryDirectDep(key("dep"), ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/true);
    assertTrue(entry.isDirty());
    assertTrue(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertTrue(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
  }

  @Test
  public void crashOnTwiceMarkedChanged() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/true);
    try {
      entry.markDirty(/*isChanged=*/true);
      fail("Cannot mark entry changed twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnTwiceMarkedDirty() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.addTemporaryDirectDep(key("dep"), ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    try {
      entry.markDirty(/*isChanged=*/false);
      fail("Cannot mark entry dirty twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnAddReverseDepTwice() {
    NodeEntry entry = new NodeEntry();
    NodeKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      fail("Cannot add same dep twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnAddReverseDepTwiceAfterDone() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    NodeKey parent = key("parent");
    assertEquals(DependencyState.DONE, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      // We only check for duplicates when we request all the reverse deps.
      entry.getReverseDeps();
      fail("Cannot add same dep twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnAddReverseDepBeforeAfterDone() {
    NodeEntry entry = new NodeEntry();
    NodeKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      // We only check for duplicates when we request all the reverse deps.
      entry.getReverseDeps();
      fail("Cannot add same dep twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnAddDirtyReverseDep() {
    NodeEntry entry = new NodeEntry();
    NodeKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      fail("Cannot add same dep twice in one build, even if dirty");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void pruneBeforeBuild() {
    NodeEntry entry = new NodeEntry();
    NodeKey dep = key("dep");
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    NodeKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep(/*version=*/0L);
    assertEquals(BuildingState.DirtyState.VERIFIED_CLEAN, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.markClean(), parent);
    assertTrue(entry.isDone());
    assertEquals(0L, entry.getVersion());
  }

  private static class IntegerNode implements Node {
    private final int value;

    IntegerNode(int value) {
      this.value = value;
    }

    @Override
    public boolean equals(Object that) {
      return (that instanceof IntegerNode) && (((IntegerNode) that).value == value);
    }

    @Override
    public int hashCode() {
      return value;
    }
  }

  @Test
  public void pruneAfterBuild() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep(/*version=*/1L);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals(0L, entry.getVersion());
  }


  @Test
  public void noPruneWhenDetailsChange() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    NodeKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep(/*version=*/1L);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    GenericNodeBuilderException exception =
        new GenericNodeBuilderException(key("cause"), new Exception());
    setValue(entry, new IntegerNode(5), new ErrorInfo(exception), /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when setValue changes", 1, entry.getVersion());
  }

  @Test
  public void pruneErrorNode() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    GenericNodeBuilderException exception =
        new GenericNodeBuilderException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep(/*version=*/1L);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals(0L, entry.getVersion());
  }

  @Test
  public void getDependencyGroup() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    NodeKey dep2 = key("dep2");
    NodeKey dep3 = key("dep3");
    entry.addTemporaryDirectDep(dep, ContinueGroup.TRUE);
    entry.addTemporaryDirectDep(dep2, ContinueGroup.FALSE);
    entry.addTemporaryDirectDep(dep3, ContinueGroup.FALSE);
    entry.signalDep();
    entry.signalDep();
    entry.signalDep();
    setValue(entry, /*value=*/new IntegerNode(5), null, 0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep, dep2);
    entry.signalDep(/*version=*/0L);
    entry.signalDep(/*version=*/0L);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep3);
  }

  @Test
  public void maintainDependencyGroupAfterRemoval() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    NodeKey dep2 = key("dep2");
    NodeKey dep3 = key("dep3");
    NodeKey dep4 = key("dep4");
    NodeKey dep5 = key("dep5");
    entry.addTemporaryDirectDep(dep, ContinueGroup.TRUE);
    entry.addTemporaryDirectDep(dep2, ContinueGroup.TRUE);
    entry.addTemporaryDirectDep(dep3, ContinueGroup.FALSE);
    entry.addTemporaryDirectDep(dep4, ContinueGroup.FALSE);
    entry.addTemporaryDirectDep(dep5, ContinueGroup.FALSE);
    entry.signalDep();
    entry.signalDep();
    // Oops! Evaluation terminated with an error, but we're going to set this entry's value anyway.
    entry.removeUnfinishedDeps(ImmutableSet.of(dep2, dep3, dep5));
    setValue(entry, null,
        new ErrorInfo(new GenericNodeBuilderException(key("key"), new Exception())), 0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    entry.signalDep(/*version=*/0L);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep4);
  }

  @Test
  public void noPruneWhenDepsChange() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    NodeKey dep = key("dep");
    entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
    entry.signalDep();
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    assertTrue(entry.signalDep(/*version=*/1L));
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    entry.addTemporaryDirectDep(key("dep2"), ContinueGroup.FALSE);
    assertTrue(entry.signalDep(/*version=*/1L));
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when deps change", 1, entry.getVersion());
  }

  @Test
  public void checkDepsOneByOne() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    List<NodeKey> deps = new ArrayList<>();
    for (int ii = 0; ii < 10; ii++) {
      NodeKey dep = key(Integer.toString(ii));
      deps.add(dep);
      entry.addTemporaryDirectDep(dep, ContinueGroup.FALSE);
      entry.signalDep();
    }
    setValue(entry, new IntegerNode(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start new evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    for (int ii = 0; ii < 10; ii++) {
      MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), deps.get(ii));
      assertTrue(entry.signalDep(/*graphVersion=*/0L));
      if (ii < 9) {
        assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
      } else {
        assertEquals(BuildingState.DirtyState.VERIFIED_CLEAN, entry.getDirtyState());
      }
    }
  }

  @Test
  public void signalOnlyNewParents() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(key("parent"));
    setValue(entry, new Node() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/true);
    NodeKey newParent = key("new parent");
    entry.addReverseDepAndCheckIfDone(newParent);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new Node() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L),
        newParent);
  }

  private static Set<NodeKey> setValue(NodeEntry entry, Node value, @Nullable ErrorInfo errorInfo,
      long graphVersion) {
    return entry.setValue(NodeWithMetadata.normal(value, errorInfo, NO_EVENTS),
        graphVersion);
  }
}
