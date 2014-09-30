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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
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

  private static final SkyFunctionName NODE_TYPE = new SkyFunctionName("Type", false);
  private static final NestedSet<TaggedEvents> NO_EVENTS =
      NestedSetBuilder.<TaggedEvents>emptySet(Order.STABLE_ORDER);

  private static SkyKey key(String name) {
    return new SkyKey(NODE_TYPE, name);
  }

  @Test
  public void createEntry() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    assertFalse(entry.isDirty());
    assertFalse(entry.isChanged());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
  }

  @Test
  public void signalEntry() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep1 = key("dep1");
    addTemporaryDirectDep(entry, dep1);
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep1);
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    addTemporaryDirectDep(entry, dep2);
    addTemporaryDirectDep(entry, dep3);
    assertFalse(entry.isReady());
    assertFalse(entry.signalDep());
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    MoreAsserts.assertEmpty(setValue(entry, new SkyValue() {},
        /*errorInfo=*/null, /*graphVersion=*/0L));
    assertTrue(entry.isDone());
    assertEquals(new IntVersion(0L), entry.getVersion());
    MoreAsserts.assertContentsAnyOrder(entry.getDirectDeps(), dep1, dep2, dep3);
  }

  @Test
  public void reverseDeps() {
    NodeEntry entry = new NodeEntry();
    SkyKey mother = key("mother");
    SkyKey father = key("father");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(mother));
    assertEquals(DependencyState.ADDED_DEP, entry.addReverseDepAndCheckIfDone(null));
    assertEquals(DependencyState.ADDED_DEP, entry.addReverseDepAndCheckIfDone(father));
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new SkyValue() {},
        /*errorInfo=*/null, /*graphVersion=*/0L),
        mother, father);
    MoreAsserts.assertContentsAnyOrder(entry.getReverseDeps(), mother, father);
    assertTrue(entry.isDone());
    entry.removeReverseDep(mother);
    assertFalse(Iterables.contains(entry.getReverseDeps(), mother));
  }

  @Test
  public void errorValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    GenericFunctionException exception =
        new GenericFunctionException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    MoreAsserts.assertEmpty(setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/0L));
    assertTrue(entry.isDone());
    assertNull(entry.getValue());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void errorAndValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    GenericFunctionException exception =
        new GenericFunctionException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    setValue(entry, new SkyValue() {}, errorInfo, /*graphVersion=*/0L);
    assertTrue(entry.isDone());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void crashOnNullErrorAndValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    try {
      setValue(entry, /*value=*/null, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    try {
      // Value() {} and Value() {} are not .equals().
      setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/1L);
      fail();
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void dirtyLifecycle() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    assertTrue(entry.isReady());
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L), parent);
  }

  @Test
  public void changedLifecycle() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/true);
    assertTrue(entry.isDirty());
    assertTrue(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    assertTrue(entry.isReady());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L), parent);
    assertEquals(new IntVersion(1L), entry.getVersion());
  }

  @Test
  public void markDirtyThenChanged() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    addTemporaryDirectDep(entry, key("dep"));
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    addTemporaryDirectDep(entry, key("dep"));
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    addTemporaryDirectDep(entry, key("dep"));
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    SkyKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      entry.getReverseDeps();
      fail("Cannot add same dep twice");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void crashOnAddReverseDepTwiceAfterDone() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    SkyKey parent = key("parent");
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
    SkyKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
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
    SkyKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      // We only check for duplicates when we request all the reverse deps.
      entry.getReverseDeps();
      fail("Cannot add same dep twice in one build, even if dirty");
    } catch (IllegalStateException e) {
      // Expected.
    }
  }

  @Test
  public void pruneBeforeBuild() {
    NodeEntry entry = new NodeEntry();
    SkyKey dep = key("dep");
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(new IntVersion(0L));
    assertEquals(BuildingState.DirtyState.VERIFIED_CLEAN, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.markClean(), parent);
    assertTrue(entry.isDone());
    assertEquals(new IntVersion(0L), entry.getVersion());
  }

  private static class IntegerValue implements SkyValue {
    private final int value;

    IntegerValue(int value) {
      this.value = value;
    }

    @Override
    public boolean equals(Object that) {
      return (that instanceof IntegerValue) && (((IntegerValue) that).value == value);
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
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(new IntVersion(1L));
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals(new IntVersion(0L), entry.getVersion());
  }


  @Test
  public void noPruneWhenDetailsChange() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(new IntVersion(1L));
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    GenericFunctionException exception =
        new GenericFunctionException(key("cause"), new Exception());
    setValue(entry, new IntegerValue(5), new ErrorInfo(exception), /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when setValue changes", new IntVersion(1), entry.getVersion());
  }

  @Test
  public void pruneErrorValue() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    GenericFunctionException exception =
        new GenericFunctionException(key("cause"), new Exception());
    ErrorInfo errorInfo = new ErrorInfo(exception);
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(new IntVersion(1L));
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals(new IntVersion(0L), entry.getVersion());
  }

  @Test
  public void getDependencyGroup() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    addTemporaryDirectDeps(entry, dep, dep2);
    addTemporaryDirectDep(entry, dep3);
    entry.signalDep();
    entry.signalDep();
    entry.signalDep();
    setValue(entry, /*value=*/new IntegerValue(5), null, 0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep, dep2);
    addTemporaryDirectDeps(entry, dep, dep2);
    entry.signalDep(new IntVersion(0L));
    entry.signalDep(new IntVersion(0L));
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep3);
  }

  @Test
  public void maintainDependencyGroupAfterRemoval() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    SkyKey dep4 = key("dep4");
    SkyKey dep5 = key("dep5");
    addTemporaryDirectDeps(entry, dep, dep2, dep3);
    addTemporaryDirectDep(entry, dep4);
    addTemporaryDirectDep(entry, dep5);
    entry.signalDep();
    entry.signalDep();
    // Oops! Evaluation terminated with an error, but we're going to set this entry's value anyway.
    entry.removeUnfinishedDeps(ImmutableSet.of(dep2, dep3, dep5));
    setValue(entry, null,
        new ErrorInfo(new GenericFunctionException(key("key"), new Exception())), 0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(new IntVersion(0L));
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep4);
  }

  @Test
  public void noPruneWhenDepsChange() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), dep);
    addTemporaryDirectDep(entry, dep);
    assertTrue(entry.signalDep(new IntVersion(1L)));
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(entry.getTemporaryDirectDeps(), dep);
    addTemporaryDirectDep(entry, key("dep2"));
    assertTrue(entry.signalDep(new IntVersion(1L)));
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when deps change", new IntVersion(1L), entry.getVersion());
  }

  @Test
  public void checkDepsOneByOne() {
    NodeEntry entry = new NodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    List<SkyKey> deps = new ArrayList<>();
    for (int ii = 0; ii < 10; ii++) {
      SkyKey dep = key(Integer.toString(ii));
      deps.add(dep);
      addTemporaryDirectDep(entry, dep);
      entry.signalDep();
    }
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start new evaluation.
    assertEquals(BuildingState.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    for (int ii = 0; ii < 10; ii++) {
      MoreAsserts.assertContentsInOrder(entry.getNextDirtyDirectDeps(), deps.get(ii));
      addTemporaryDirectDep(entry, deps.get(ii));
      assertTrue(entry.signalDep(new IntVersion(0L)));
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
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/true);
    SkyKey newParent = key("new parent");
    entry.addReverseDepAndCheckIfDone(newParent);
    assertEquals(BuildingState.DirtyState.REBUILDING, entry.getDirtyState());
    MoreAsserts.assertContentsAnyOrder(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L),
        newParent);
  }

  private static Set<SkyKey> setValue(NodeEntry entry, SkyValue value,
      @Nullable ErrorInfo errorInfo, long graphVersion) {
    return entry.setValue(ValueWithMetadata.normal(value, errorInfo, NO_EVENTS),
        new IntVersion(graphVersion));
  }

  private static void addTemporaryDirectDep(NodeEntry entry, SkyKey key) {
    GroupedListHelper<SkyKey> helper = new GroupedListHelper<>();
    helper.add(key);
    entry.addTemporaryDirectDeps(helper);
  }

  private static void addTemporaryDirectDeps(NodeEntry entry, SkyKey... keys) {
    GroupedListHelper<SkyKey> helper = new GroupedListHelper<>();
    helper.startGroup();
    for (SkyKey key : keys) {
      helper.add(key);
    }
    helper.endGroup();
    entry.addTemporaryDirectDeps(helper);
  }
}
