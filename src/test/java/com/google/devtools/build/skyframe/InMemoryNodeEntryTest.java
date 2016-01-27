// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Tests for {@link InMemoryNodeEntry}.
 */
@RunWith(JUnit4.class)
public class InMemoryNodeEntryTest {

  private static final SkyFunctionName NODE_TYPE = SkyFunctionName.create("Type");
  private static final NestedSet<TaggedEvents> NO_EVENTS =
      NestedSetBuilder.<TaggedEvents>emptySet(Order.STABLE_ORDER);

  private static SkyKey key(String name) {
    return new SkyKey(NODE_TYPE, name);
  }

  @Test
  public void createEntry() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    assertFalse(entry.isDirty());
    assertFalse(entry.isChanged());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
  }

  @Test
  public void signalEntry() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep1 = key("dep1");
    addTemporaryDirectDep(entry, dep1);
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep1);
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    addTemporaryDirectDep(entry, dep2);
    addTemporaryDirectDep(entry, dep3);
    assertFalse(entry.isReady());
    assertFalse(entry.signalDep());
    assertFalse(entry.isReady());
    assertTrue(entry.signalDep());
    assertTrue(entry.isReady());
    assertThat(setValue(entry, new SkyValue() {},
        /*errorInfo=*/null, /*graphVersion=*/0L)).isEmpty();
    assertTrue(entry.isDone());
    assertEquals(IntVersion.of(0L), entry.getVersion());
    assertThat(entry.getDirectDeps()).containsExactly(dep1, dep2, dep3);
  }

  @Test
  public void reverseDeps() {
    NodeEntry entry = new InMemoryNodeEntry();
    SkyKey mother = key("mother");
    SkyKey father = key("father");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(mother));
    assertEquals(DependencyState.ALREADY_EVALUATING, entry.addReverseDepAndCheckIfDone(null));
    assertEquals(DependencyState.ALREADY_EVALUATING, entry.addReverseDepAndCheckIfDone(father));
    assertThat(setValue(entry, new SkyValue() {},
        /*errorInfo=*/null, /*graphVersion=*/0L)).containsExactly(mother, father);
    assertThat(entry.getReverseDeps()).containsExactly(mother, father);
    assertTrue(entry.isDone());
    entry.removeReverseDep(mother);
    assertFalse(Iterables.contains(entry.getReverseDeps(), mother));
  }

  @Test
  public void errorValue() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    ReifiedSkyFunctionException exception = new ReifiedSkyFunctionException(
        new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT),
        key("cause"));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    assertThat(setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/0L)).isEmpty();
    assertTrue(entry.isDone());
    assertNull(entry.getValue());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void errorAndValue() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    ReifiedSkyFunctionException exception = new ReifiedSkyFunctionException(
        new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT),
        key("cause"));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, new SkyValue() {}, errorInfo, /*graphVersion=*/0L);
    assertTrue(entry.isDone());
    assertEquals(errorInfo, entry.getErrorInfo());
  }

  @Test
  public void crashOnNullErrorAndValue() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    assertTrue(entry.isReady());
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    assertThat(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L)).containsExactly(parent);
  }

  @Test
  public void changedLifecycle() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertTrue(entry.isReady());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).containsExactly(dep);
    assertThat(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L)).containsExactly(parent);
    assertEquals(IntVersion.of(1L), entry.getVersion());
  }

  @Test
  public void markDirtyThenChanged() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
    SkyKey parent = key("parent");
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(parent));
    try {
      entry.addReverseDepAndCheckIfDone(parent);
      assertThat(setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L))
          .containsExactly(parent);
      fail("Cannot add same dep twice");
    } catch (IllegalStateException e) {
      assertThat(e.getMessage()).contains("Duplicate reverse deps");
    }
  }

  @Test
  public void crashOnAddReverseDepTwiceAfterDone() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    NodeEntry entry = new InMemoryNodeEntry();
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
  public void pruneBeforeBuild() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(0L));
    assertEquals(NodeEntry.DirtyState.VERIFIED_CLEAN, entry.getDirtyState());
    assertThat(entry.markClean()).containsExactly(parent);
    assertTrue(entry.isDone());
    assertEquals(IntVersion.of(0L), entry.getVersion());
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
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(1L));
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals(IntVersion.of(0L), entry.getVersion());
  }

  @Test
  public void noPruneWhenDetailsChange() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(1L));
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    ReifiedSkyFunctionException exception = new ReifiedSkyFunctionException(
        new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT),
        key("cause"));
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    setValue(entry, new IntegerValue(5), ErrorInfo.fromException(exception, false),
        /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when setValue changes", IntVersion.of(1), entry.getVersion());
  }

  @Test
  public void pruneWhenDepGroupReordered() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    SkyKey dep1InGroup = key("dep1InGroup");
    SkyKey dep2InGroup = key("dep2InGroup");
    addTemporaryDirectDep(entry, dep);
    addTemporaryDirectDeps(entry, dep1InGroup, dep2InGroup);
    entry.signalDep();
    entry.signalDep();
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/ null, /*graphVersion=*/ 0L);
    assertFalse(entry.isDirty());
    assertTrue(entry.isDone());
    entry.markDirty(/*isChanged=*/ false);
    assertTrue(entry.isDirty());
    assertFalse(entry.isChanged());
    assertFalse(entry.isDone());
    assertTrue(entry.isReady());
    entry.addReverseDepAndCheckIfDone(null);
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(1L));
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps())
        .containsExactly(dep1InGroup, dep2InGroup);
    addTemporaryDirectDeps(entry, dep2InGroup, dep1InGroup);
    assertFalse(entry.signalDep());
    assertTrue(entry.signalDep());
    setValue(entry, new IntegerValue(5), /*errorInfo=*/ null, /*graphVersion=*/ 1L);
    assertTrue(entry.isDone());
    assertEquals(
        "Version does not change when dep group reordered", IntVersion.of(0), entry.getVersion());
  }

  @Test
  public void errorInfoCannotBePruned() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    ReifiedSkyFunctionException exception = new ReifiedSkyFunctionException(
        new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT),
        key("cause"));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(1L));
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    setValue(entry, /*value=*/null, errorInfo, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    // ErrorInfo is treated as a NotComparableSkyValue, so it is not pruned.
    assertEquals(IntVersion.of(1L), entry.getVersion());
  }

  @Test
  public void getDependencyGroup() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep, dep2);
    addTemporaryDirectDeps(entry, dep, dep2);
    entry.signalDep(IntVersion.of(0L));
    entry.signalDep(IntVersion.of(0L));
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep3);
  }

  @Test
  public void maintainDependencyGroupAfterRemoval() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    ReifiedSkyFunctionException exception = new ReifiedSkyFunctionException(
        new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT),
        key("key"));
    setValue(entry, null, ErrorInfo.fromException(exception, false), 0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    entry.signalDep(IntVersion.of(0L));
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep4);
  }

  @Test
  public void noPruneWhenDepsChange() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    SkyKey dep = key("dep");
    addTemporaryDirectDep(entry, dep);
    entry.signalDep();
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/false);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    addTemporaryDirectDep(entry, dep);
    assertTrue(entry.signalDep(IntVersion.of(1L)));
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(dep);
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    addTemporaryDirectDep(entry, key("dep2"));
    assertTrue(entry.signalDep(IntVersion.of(1L)));
    setValue(entry, new IntegerValue(5), /*errorInfo=*/null, /*graphVersion=*/1L);
    assertTrue(entry.isDone());
    assertEquals("Version increments when deps change", IntVersion.of(1L), entry.getVersion());
  }

  @Test
  public void checkDepsOneByOne() {
    NodeEntry entry = new InMemoryNodeEntry();
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
    assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
    for (int ii = 0; ii < 10; ii++) {
      assertThat(entry.getNextDirtyDirectDeps()).containsExactly(deps.get(ii));
      addTemporaryDirectDep(entry, deps.get(ii));
      assertTrue(entry.signalDep(IntVersion.of(0L)));
      if (ii < 9) {
        assertEquals(NodeEntry.DirtyState.CHECK_DEPENDENCIES, entry.getDirtyState());
      } else {
        assertEquals(NodeEntry.DirtyState.VERIFIED_CLEAN, entry.getDirtyState());
      }
    }
  }

  @Test
  public void signalOnlyNewParents() {
    NodeEntry entry = new InMemoryNodeEntry();
    entry.addReverseDepAndCheckIfDone(key("parent"));
    setValue(entry, new SkyValue() {}, /*errorInfo=*/null, /*graphVersion=*/0L);
    entry.markDirty(/*isChanged=*/true);
    SkyKey newParent = key("new parent");
    entry.addReverseDepAndCheckIfDone(newParent);
    assertEquals(NodeEntry.DirtyState.NEEDS_REBUILDING, entry.getDirtyState());
    assertThat(entry.markRebuildingAndGetAllRemainingDirtyDirectDeps()).isEmpty();
    assertThat(setValue(entry, new SkyValue() {}, /*errorInfo=*/null,
        /*graphVersion=*/1L)).containsExactly(newParent);
  }

  @Test
  public void testClone() {
    InMemoryNodeEntry entry = new InMemoryNodeEntry();
    IntVersion version = IntVersion.of(0);
    IntegerValue originalValue = new IntegerValue(42);
    SkyKey originalChild = key("child");
    addTemporaryDirectDep(entry, originalChild);
    entry.signalDep();
    entry.setValue(originalValue, version);
    entry.addReverseDepAndCheckIfDone(key("parent1"));
    InMemoryNodeEntry clone1 = entry.cloneNodeEntry();
    entry.addReverseDepAndCheckIfDone(key("parent2"));
    InMemoryNodeEntry clone2 = entry.cloneNodeEntry();
    entry.removeReverseDep(key("parent1"));
    entry.removeReverseDep(key("parent2"));
    IntegerValue updatedValue = new IntegerValue(52);
    clone2.markDirty(true);
    clone2.addReverseDepAndCheckIfDone(null);
    SkyKey newChild = key("newchild");
    addTemporaryDirectDep(clone2, newChild);
    clone2.signalDep();
    assertThat(clone2.markRebuildingAndGetAllRemainingDirtyDirectDeps())
        .containsExactly(originalChild);
    clone2.setValue(updatedValue, version.next());

    assertThat(entry.getVersion()).isEqualTo(version);
    assertThat(clone1.getVersion()).isEqualTo(version);
    assertThat(clone2.getVersion()).isEqualTo(version.next());

    assertThat(entry.getValue()).isEqualTo(originalValue);
    assertThat(clone1.getValue()).isEqualTo(originalValue);
    assertThat(clone2.getValue()).isEqualTo(updatedValue);

    assertThat(entry.getDirectDeps()).containsExactly(originalChild);
    assertThat(clone1.getDirectDeps()).containsExactly(originalChild);
    assertThat(clone2.getDirectDeps()).containsExactly(newChild);

    assertThat(entry.getReverseDeps()).hasSize(0);
    assertThat(clone1.getReverseDeps()).containsExactly(key("parent1"));
    assertThat(clone2.getReverseDeps()).containsExactly(key("parent1"), key("parent2"));
  }

  @Test
  public void getGroupedDirectDeps() {
    InMemoryNodeEntry entry = new InMemoryNodeEntry();
    ImmutableList<ImmutableSet<SkyKey>> groupedDirectDeps = ImmutableList.of(
        ImmutableSet.of(key("1A")),
        ImmutableSet.of(key("2A"), key("2B")),
        ImmutableSet.of(key("3A"), key("3B"), key("3C")),
        ImmutableSet.of(key("4A"), key("4B"), key("4C"), key("4D")));
    for (Set<SkyKey> depGroup : groupedDirectDeps) {
      entry.addTemporaryDirectDeps(GroupedListHelper.create(depGroup));
      for (int i = 0; i < depGroup.size(); i++) {
        entry.signalDep();
      }
    }
    entry.setValue(new IntegerValue(42), IntVersion.of(42L));
    int i = 0;
    GroupedList<SkyKey> entryGroupedDirectDeps = entry.getGroupedDirectDeps();
    assertThat(Iterables.size(entryGroupedDirectDeps)).isEqualTo(groupedDirectDeps.size());
    for (Iterable<SkyKey> depGroup : entryGroupedDirectDeps) {
      assertThat(depGroup).containsExactlyElementsIn(groupedDirectDeps.get(i++));
    }
  }

  private static Set<SkyKey> setValue(NodeEntry entry, SkyValue value,
      @Nullable ErrorInfo errorInfo, long graphVersion) {
    return entry.setValue(
        ValueWithMetadata.normal(value, errorInfo, NO_EVENTS), IntVersion.of(graphVersion));
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
