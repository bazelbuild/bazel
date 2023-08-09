// Copyright 2023 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.skyframe.NodeEntrySubjectFactory.assertThatNodeEntry;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link IncrementalInMemoryNodeEntry}. */
@RunWith(TestParameterInjector.class)
public final class IncrementalInMemoryNodeEntryTest extends InMemoryNodeEntryTest {

  @Override
  boolean keepEdges() {
    return true;
  }

  @Test
  public void dirtyLifecycle() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ONE_VERSION, dep);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    entry.markRebuilding();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 1L))
        .containsExactly(parent);
  }

  @Test
  public void changedLifecycle() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.CHANGE);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    entry.markRebuilding();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 1L))
        .containsExactly(parent);
    assertThat(entry.getVersion()).isEqualTo(ONE_VERSION);
  }

  @Test
  public void markDirtyThenChanged() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    entry.markDirty(DirtyType.CHANGE);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
  }

  @Test
  public void markChangedThenDirty() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.CHANGE);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
  }

  @Test
  public void crashOnTwiceMarkedChanged() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.CHANGE);
    assertThrows(
        "Cannot mark entry changed twice",
        IllegalStateException.class,
        () -> entry.markDirty(DirtyType.CHANGE));
  }

  @Test
  public void crashOnTwiceMarkedDirty() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.DIRTY);
    assertThrows(
        "Cannot mark entry dirty twice",
        IllegalStateException.class,
        () -> entry.markDirty(DirtyType.DIRTY));
  }

  @Test
  public void crashOnAddReverseDepTwiceAfterDone() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    SkyKey parent = key("parent");
    assertThat(entry.addReverseDepAndCheckIfDone(parent)).isEqualTo(DependencyState.DONE);
    entry.addReverseDepAndCheckIfDone(parent);
    assertThrows(
        "Cannot add same dep twice",
        IllegalStateException.class,
        // We only check for duplicates when we request all the reverse deps.
        entry::getReverseDepsForDoneEntry);
  }

  @Test
  public void crashOnAddReverseDepBeforeAfterDone() throws InterruptedException {
    NodeEntry entry = createEntry();
    SkyKey parent = key("parent");
    assertThat(entry.addReverseDepAndCheckIfDone(parent))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.addReverseDepAndCheckIfDone(parent);
    assertThrows(
        "Cannot add same dep twice",
        IllegalStateException.class,
        // We only check for duplicates when we request all the reverse deps.
        entry::getReverseDepsForDoneEntry);
  }

  @Test
  public void pruneBeforeBuild() throws InterruptedException {
    NodeEntry entry = createEntry();
    SkyKey dep = key("dep");
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.VERIFIED_CLEAN);
    assertThat(entry.markClean().getRdepsToSignal()).containsExactly(parent);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getVersion()).isEqualTo(ZERO_VERSION);
  }

  @Test
  public void pruneAfterBuild() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ONE_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 1L);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getVersion()).isEqualTo(ZERO_VERSION);
  }

  @Test
  public void noPruneWhenDetailsChange() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ONE_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    entry.markRebuilding();
    setValue(
        entry,
        new IntegerValue(5),
        ErrorInfo.fromException(exception, false),
        /* graphVersion= */ 1L);
    assertThat(entry.isDone()).isTrue();
    assertWithMessage("Version increments when setValue changes")
        .that(entry.getVersion())
        .isEqualTo(IntVersion.of(1));
  }

  @Test
  public void pruneWhenDepGroupReordered() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    SkyKey dep1InGroup = key("dep1InGroup");
    SkyKey dep2InGroup = key("dep2InGroup");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep1InGroup, dep2InGroup));
    entry.signalDep(ZERO_VERSION, dep);
    entry.signalDep(ZERO_VERSION, dep1InGroup);
    entry.signalDep(ZERO_VERSION, dep2InGroup);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    entry.addReverseDepAndCheckIfDone(null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ONE_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep2InGroup, dep1InGroup));
    assertThat(entry.signalDep(ONE_VERSION, dep2InGroup)).isFalse();
    assertThat(entry.signalDep(ONE_VERSION, dep1InGroup)).isTrue();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 1L);
    assertThat(entry.isDone()).isTrue();
    assertWithMessage("Version does not change when dep group reordered")
        .that(entry.getVersion())
        .isEqualTo(IntVersion.of(0));
  }

  @Test
  public void errorInfoCannotBePruned() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, /* value= */ null, errorInfo, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ONE_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    setValue(entry, /* value= */ null, errorInfo, /* graphVersion= */ 1L);
    assertThat(entry.isDone()).isTrue();
    // ErrorInfo is treated as a NotComparableSkyValue, so it is not pruned.
    assertThat(entry.getVersion()).isEqualTo(ONE_VERSION);
  }

  @Test
  public void getDependencyGroup() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep, dep2));
    entry.addSingletonTemporaryDirectDep(dep3);
    entry.signalDep(ZERO_VERSION, dep);
    entry.signalDep(ZERO_VERSION, dep2);
    entry.signalDep(ZERO_VERSION, dep3);
    setValue(entry, /* value= */ new IntegerValue(5), null, 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep, dep2);
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep, dep2));
    entry.signalDep(ZERO_VERSION, /* childForDebugging= */ null);
    entry.signalDep(ZERO_VERSION, /* childForDebugging= */ null);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep3);
  }

  @Test
  public void maintainDependencyGroupAfterRemoval() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    SkyKey dep4 = key("dep4");
    SkyKey dep5 = key("dep5");
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep, dep2, dep3));
    entry.addSingletonTemporaryDirectDep(dep4);
    entry.addSingletonTemporaryDirectDep(dep5);
    entry.signalDep(ZERO_VERSION, dep4);
    entry.signalDep(ZERO_VERSION, dep);
    // Oops! Evaluation terminated with an error, but we're going to set this entry's value anyway.
    entry.removeUnfinishedDeps(ImmutableSet.of(dep2, dep3, dep5));
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    setValue(entry, null, ErrorInfo.fromException(exception, false), 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep4);
  }

  @Test
  public void pruneWhenDepsChange() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    assertThat(entry.signalDep(ONE_VERSION, /* childForDebugging= */ null)).isTrue();
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(key("dep2"));
    assertThat(entry.signalDep(ONE_VERSION, /* childForDebugging= */ null)).isTrue();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 1L);
    assertThat(entry.isDone()).isTrue();
    assertThatNodeEntry(entry).hasVersionThat().isEqualTo(ZERO_VERSION);
  }

  @Test
  public void checkDepsOneByOne() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    List<SkyKey> deps = new ArrayList<>();
    for (int ii = 0; ii < 10; ii++) {
      SkyKey dep = key(Integer.toString(ii));
      deps.add(dep);
      entry.addSingletonTemporaryDirectDep(dep);
      entry.signalDep(ZERO_VERSION, dep);
    }
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start new evaluation.
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
    for (int ii = 0; ii < 10; ii++) {
      assertThat(entry.getNextDirtyDirectDeps()).containsExactly(deps.get(ii));
      entry.addSingletonTemporaryDirectDep(deps.get(ii));
      assertThat(entry.signalDep(ZERO_VERSION, /* childForDebugging= */ null)).isTrue();
      if (ii < 9) {
        assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.CHECK_DEPENDENCIES);
      } else {
        assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.VERIFIED_CLEAN);
      }
    }
  }

  @Test
  public void signalOnlyNewParents() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(key("parent"));
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    entry.markDirty(DirtyType.CHANGE);
    SkyKey newParent = key("new parent");
    entry.addReverseDepAndCheckIfDone(newParent);
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.NEEDS_REBUILDING);
    entry.markRebuilding();
    assertThat(entry.getDirtyState()).isEqualTo(NodeEntry.DirtyState.REBUILDING);
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 1L))
        .containsExactly(newParent);
  }

  @Test
  public void getCompressedDirectDepsForDoneEntry() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    ImmutableList<ImmutableList<SkyKey>> groupedDirectDeps =
        ImmutableList.of(
            ImmutableList.of(key("1A")),
            ImmutableList.of(key("2A"), key("2B")),
            ImmutableList.of(key("3A"), key("3B"), key("3C")),
            ImmutableList.of(key("4A"), key("4B"), key("4C"), key("4D")));
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    for (ImmutableList<SkyKey> depGroup : groupedDirectDeps) {
      entry.addTemporaryDirectDepGroup(depGroup);
      for (SkyKey dep : depGroup) {
        entry.signalDep(ZERO_VERSION, dep);
      }
    }
    entry.setValue(new IntegerValue(42), IntVersion.of(42L), null);
    assertThat(GroupedDeps.decompress(entry.getCompressedDirectDepsForDoneEntry()))
        .containsExactlyElementsIn(groupedDirectDeps)
        .inOrder();
  }

  @Test
  public void hasAtLeastOneDep_true() throws Exception {
    SkyKey dep = key("dep");
    InMemoryNodeEntry entry = createEntry();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    entry.setValue(new IntegerValue(1), ZERO_VERSION, null);
    assertThat(entry.hasAtLeastOneDep()).isTrue();
  }

  @Test
  public void hasAtLeastOneDep_false() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    entry.addTemporaryDirectDepGroup(ImmutableList.of());
    entry.setValue(new IntegerValue(1), ZERO_VERSION, null);
    assertThat(entry.hasAtLeastOneDep()).isFalse();
  }
}
