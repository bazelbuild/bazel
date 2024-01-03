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
import com.google.devtools.build.skyframe.NodeEntry.LifecycleState;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link IncrementalInMemoryNodeEntry}. */
@RunWith(TestParameterInjector.class)
public class IncrementalInMemoryNodeEntryTest extends InMemoryNodeEntryTest<IntVersion> {

  protected final IntVersion incrementalVersion = initialVersion.next();

  @Override
  protected IncrementalInMemoryNodeEntry createEntry(SkyKey key) {
    return new IncrementalInMemoryNodeEntry(key);
  }

  @Override
  final IntVersion getInitialVersion() {
    return IntVersion.of(0);
  }

  @Test
  public void dirtyLifecycle() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    SkyValue oldValue = new IntegerValue(1);
    setValue(entry, oldValue, /* errorInfo= */ null, initialVersion);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.toValue()).isEqualTo(oldValue);

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
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);

    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(incrementalVersion, dep);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();

    entry.markRebuilding();
    assertThat(setValue(entry, new IntegerValue(2), /* errorInfo= */ null, incrementalVersion))
        .containsExactly(parent);
    assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
  }

  @Test
  public void changedLifecycle() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    SkyValue oldValue = new IntegerValue(1);
    setValue(entry, oldValue, /* errorInfo= */ null, initialVersion);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    entry.markDirty(DirtyType.CHANGE);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.toValue()).isEqualTo(oldValue);

    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();

    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();

    entry.markRebuilding();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, incrementalVersion))
        .containsExactly(parent);
    assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
  }

  @Test
  public void markDirtyThenChanged() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
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
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
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
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
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
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    assertThrows(
        "Cannot mark entry dirty twice",
        IllegalStateException.class,
        () -> entry.markDirty(DirtyType.DIRTY));
  }

  @Test
  public void forceRebuildAfterTransientError() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    entry.addSingletonTemporaryDirectDep(ErrorTransienceValue.KEY);
    entry.signalDep(initialVersion, ErrorTransienceValue.KEY);

    setValue(
        entry,
        /* value= */ null,
        ErrorInfo.fromException(
            new ReifiedSkyFunctionException(
                new GenericFunctionException(
                    new SomeErrorException("transient error"), Transience.TRANSIENT)),
            /* isTransitivelyTransient= */ true),
        initialVersion);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    entry.markDirty(DirtyType.DIRTY);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isFalse();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);

    SkyKey parent = key("parent");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(parent)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();

    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    assertThat(entry.signalDep(initialVersion, dep)).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);

    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(ErrorTransienceValue.KEY);
    entry.forceRebuild();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);

    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, incrementalVersion))
        .containsExactly(parent);
    assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
    assertThat(entry.getDirectDeps()).containsExactly(dep); // No more dep on error transience.
  }

  @Test
  public void crashOnAddReverseDepTwiceAfterDone() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
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
    InMemoryNodeEntry entry = createEntry();
    SkyKey parent = key("parent");
    assertThat(entry.addReverseDepAndCheckIfDone(parent))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
    entry.addReverseDepAndCheckIfDone(parent);
    assertThrows(
        "Cannot add same dep twice",
        IllegalStateException.class,
        // We only check for duplicates when we request all the reverse deps.
        entry::getReverseDepsForDoneEntry);
  }

  @Test
  public void pruneBeforeBuild() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    SkyKey dep = key("dep");
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
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
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.VERIFIED_CLEAN);
    assertThat(entry.markClean().getRdepsToSignal()).containsExactly(parent);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getVersion()).isEqualTo(initialVersion);
  }

  @Test
  public void pruneAfterBuild() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(incrementalVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, incrementalVersion);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getVersion()).isEqualTo(initialVersion);
  }

  @Test
  public void noPruneWhenDetailsChange() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, initialVersion);
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
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(incrementalVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    entry.markRebuilding();
    setValue(
        entry, new IntegerValue(5), ErrorInfo.fromException(exception, false), incrementalVersion);
    assertThat(entry.isDone()).isTrue();
    assertWithMessage("Version increments when setValue changes")
        .that(entry.getVersion())
        .isEqualTo(IntVersion.of(1));
  }

  @Test
  public void pruneWhenDepGroupReordered() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    SkyKey dep1InGroup = key("dep1InGroup");
    SkyKey dep2InGroup = key("dep2InGroup");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep1InGroup, dep2InGroup));
    entry.signalDep(initialVersion, dep);
    entry.signalDep(initialVersion, dep1InGroup);
    entry.signalDep(initialVersion, dep2InGroup);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, initialVersion);
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
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(incrementalVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep2InGroup, dep1InGroup));
    assertThat(entry.signalDep(incrementalVersion, dep2InGroup)).isFalse();
    assertThat(entry.signalDep(incrementalVersion, dep1InGroup)).isTrue();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, incrementalVersion);
    assertThat(entry.isDone()).isTrue();
    assertWithMessage("Version does not change when dep group reordered")
        .that(entry.getVersion())
        .isEqualTo(IntVersion.of(0));
  }

  @Test
  public void errorInfoCannotBePruned() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, /* value= */ null, errorInfo, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(incrementalVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    setValue(entry, /* value= */ null, errorInfo, incrementalVersion);
    assertThat(entry.isDone()).isTrue();
    // ErrorInfo is treated as a NotComparableSkyValue, so it is not pruned.
    assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
  }

  @Test
  public void getDependencyGroup() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep, dep2));
    entry.addSingletonTemporaryDirectDep(dep3);
    entry.signalDep(initialVersion, dep);
    entry.signalDep(initialVersion, dep2);
    entry.signalDep(initialVersion, dep3);
    setValue(entry, /* value= */ new IntegerValue(5), null, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep, dep2);
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep, dep2));
    entry.signalDep(initialVersion, /* childForDebugging= */ null);
    entry.signalDep(initialVersion, /* childForDebugging= */ null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep3);
  }

  @Test
  public void maintainDependencyGroupAfterRemoval() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
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
    entry.signalDep(initialVersion, dep4);
    entry.signalDep(initialVersion, dep);
    // Oops! Evaluation terminated with an error, but we're going to set this entry's value anyway.
    entry.removeUnfinishedDeps(ImmutableSet.of(dep2, dep3, dep5));
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    setValue(entry, null, ErrorInfo.fromException(exception, false), initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Restart evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep4);
  }

  @Test
  public void pruneWhenDepsChange() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(dep);
    entry.addSingletonTemporaryDirectDep(dep);
    assertThat(entry.signalDep(incrementalVersion, /* childForDebugging= */ null)).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep);
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(key("dep2"));
    assertThat(entry.signalDep(incrementalVersion, /* childForDebugging= */ null)).isTrue();
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, incrementalVersion);
    assertThat(entry.isDone()).isTrue();
    assertThatNodeEntry(entry).hasVersionThat().isEqualTo(initialVersion);
  }

  @Test
  public void checkDepsOneByOne() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    List<SkyKey> deps = new ArrayList<>();
    for (int ii = 0; ii < 10; ii++) {
      SkyKey dep = key(Integer.toString(ii));
      deps.add(dep);
      entry.addSingletonTemporaryDirectDep(dep);
      entry.signalDep(initialVersion, dep);
    }
    setValue(entry, new IntegerValue(5), /* errorInfo= */ null, initialVersion);
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null); // Start new evaluation.
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
    for (int ii = 0; ii < 10; ii++) {
      assertThat(entry.getNextDirtyDirectDeps()).containsExactly(deps.get(ii));
      entry.addSingletonTemporaryDirectDep(deps.get(ii));
      assertThat(entry.signalDep(initialVersion, /* childForDebugging= */ null)).isTrue();
      if (ii < 9) {
        assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);
      } else {
        assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.VERIFIED_CLEAN);
      }
    }
  }

  @Test
  public void signalOnlyNewParents() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(key("parent"));
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
    entry.markDirty(DirtyType.CHANGE);
    SkyKey newParent = key("new parent");
    entry.addReverseDepAndCheckIfDone(newParent);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    entry.markRebuilding();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, incrementalVersion))
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
        entry.signalDep(initialVersion, dep);
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
    entry.signalDep(initialVersion, dep);
    entry.setValue(new IntegerValue(1), initialVersion, null);
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
    entry.setValue(new IntegerValue(1), initialVersion, null);
    assertThat(entry.hasAtLeastOneDep()).isFalse();
  }

  @Test
  public void getAllDirectDepsForIncompleteNode_notYetEvaluating() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    assertThat(entry.getAllDirectDepsForIncompleteNode()).isEmpty();
  }

  @Test
  public void getAllDirectDepsForIncompleteNode_initialBuild() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
    entry.markRebuilding();

    SkyKey dep1 = key("dep1");
    SkyKey dep2 = key("dep2");
    entry.addTemporaryDirectDepGroup(ImmutableList.of(dep1, dep2));

    assertThat(entry.getAllDirectDepsForIncompleteNode()).containsExactly(dep1, dep2);
  }

  @Test
  public void getAllDirectDepsForIncompleteNode_incrementalBuild() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
    entry.markRebuilding();

    // Dep added on initial build that stays on incremental build.
    SkyKey oldAndNewDep = key("oldAndNewDep");
    entry.addSingletonTemporaryDirectDep(oldAndNewDep);
    entry.signalDep(initialVersion, oldAndNewDep);

    // Dep added on initial build that is removed on incremental build.
    SkyKey oldDep = key("oldDep");
    entry.addSingletonTemporaryDirectDep(oldDep);
    entry.signalDep(initialVersion, oldDep);

    // Initial build completes.
    setValue(entry, new IntegerValue(1), /* errorInfo= */ null, initialVersion);

    // Start of incremental build.
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null);

    // First dep changed, causes rebuild.
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(oldAndNewDep);
    entry.addSingletonTemporaryDirectDep(oldAndNewDep);
    entry.signalDep(incrementalVersion, oldAndNewDep);
    entry.markRebuilding();

    // New dep requested.
    SkyKey newDep = key("newDep");
    entry.addSingletonTemporaryDirectDep(newDep);

    assertThat(entry.getAllDirectDepsForIncompleteNode())
        .containsExactly(oldDep, oldAndNewDep, newDep);
  }

  @Test
  public void getAllDirectDepsForIncompleteNode_afterReset() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
    entry.markRebuilding();

    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    entry.resetEvaluationFromScratch();

    assertThat(entry.getAllDirectDepsForIncompleteNode()).containsExactly(dep);
  }

  @Test
  public void resetOnDirtyNode(@TestParameter boolean valueChanges) throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();

    // Rdep added on initial build that stays on incremental build.
    SkyKey oldAndNewParent = key("oldAndNewParent");
    entry.addReverseDepAndCheckIfDone(oldAndNewParent);

    // Rdep added on initial build that is removed on incremental build.
    SkyKey oldParent = key("oldParent");
    entry.addReverseDepAndCheckIfDone(oldParent);

    // Dep added on initial build that stays on incremental build.
    SkyKey oldAndNewDep = key("oldAndNewDep");
    entry.addSingletonTemporaryDirectDep(oldAndNewDep);
    entry.signalDep(initialVersion, oldAndNewDep);

    // Dep added on initial build that is removed on incremental build.
    SkyKey oldDep = key("oldDep");
    entry.addSingletonTemporaryDirectDep(oldDep);
    entry.signalDep(initialVersion, oldDep);

    // Initial build completes.
    SkyValue oldValue = new IntegerValue(1);
    setValue(entry, oldValue, /* errorInfo= */ null, initialVersion);

    // Start of incremental build.
    entry.markDirty(DirtyType.DIRTY);

    // One rdep added, one rdep stays, one rdep removed.
    SkyKey newParent = key("newParent");
    entry.addReverseDepAndCheckIfDone(newParent);
    entry.checkIfDoneForDirtyReverseDep(oldAndNewParent);
    entry.removeReverseDep(oldParent);

    // Old dep added before reset, triggers rebuild.
    assertThat(entry.getNextDirtyDirectDeps()).containsExactly(oldAndNewDep);
    entry.addSingletonTemporaryDirectDep(oldAndNewDep);
    entry.signalDep(incrementalVersion, oldAndNewDep);
    entry.markRebuilding();
    assertThat(entry.getResetDirectDeps()).isEmpty();

    // Reset clears temporary direct deps, but preserves the dirty value.
    entry.resetEvaluationFromScratch();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Add back same dep.
    entry.addSingletonTemporaryDirectDep(oldAndNewDep);
    assertThat(entry.signalDep(incrementalVersion, oldAndNewDep)).isTrue();
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(oldAndNewDep));

    // New dep added after reset.
    SkyKey newDep = key("newDep");
    entry.addSingletonTemporaryDirectDep(newDep);
    assertThat(entry.signalDep(incrementalVersion, newDep)).isTrue();
    assertThat(entry.getTemporaryDirectDeps())
        .containsExactly(ImmutableList.of(oldAndNewDep), ImmutableList.of(newDep));

    // Check dep accounting.
    assertThat(entry.getResetDirectDeps()).containsExactly(oldAndNewDep);
    assertThat(entry.getAllRemainingDirtyDirectDeps()).containsExactly(oldDep);

    // Set value and check that new parents will be signaled.
    SkyValue newValue = valueChanges ? new IntegerValue(2) : oldValue;
    assertThat(setValue(entry, newValue, /* errorInfo= */ null, incrementalVersion))
        .containsExactly(oldAndNewParent, newParent);

    if (valueChanges) {
      assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
    } else {
      assertThat(entry.getVersion()).isEqualTo(initialVersion); // Change pruning.
    }
  }

  @Test
  public void rewindOnIncrementalBuild(@TestParameter boolean valueChanges)
      throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();

    // Initial build.
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyValue oldValue = new IntegerValue(1);
    setValue(entry, oldValue, /* errorInfo= */ null, initialVersion);

    // Rdeps that exhibit various behavior and timing on the incremental build.
    SkyKey earlyOldParent = key("earlyOldParent");
    SkyKey lateOldParent = key("lateOldParent");
    SkyKey cleanParent = key("cleanParent");
    SkyKey resetDirtyParent = key("resetDirtyParent");
    SkyKey earlyDirtyParent = key("earlyDirtyParent");
    SkyKey lateDirtyParent = key("lateDirtyParent");
    entry.addReverseDepAndCheckIfDone(earlyOldParent);
    entry.addReverseDepAndCheckIfDone(lateOldParent);
    entry.addReverseDepAndCheckIfDone(cleanParent);
    entry.addReverseDepAndCheckIfDone(resetDirtyParent);
    entry.addReverseDepAndCheckIfDone(earlyDirtyParent);
    entry.addReverseDepAndCheckIfDone(lateDirtyParent);

    // Start of incremental build.

    // Rdep removed before rewinding.
    entry.removeReverseDep(earlyOldParent);

    // Dirty rdep registered before rewinding.
    assertThat(entry.checkIfDoneForDirtyReverseDep(earlyDirtyParent))
        .isEqualTo(DependencyState.DONE);

    // New rdep registered before rewinding.
    SkyKey earlyNewParent = key("earlyNewParent");
    assertThat(entry.addReverseDepAndCheckIfDone(earlyNewParent)).isEqualTo(DependencyState.DONE);

    // Rewinding initiated.
    assertThat(entry.checkIfDoneForDirtyReverseDep(resetDirtyParent))
        .isEqualTo(DependencyState.DONE);
    assertThat(entry.markDirty(DirtyType.REWIND)).isNotNull();
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Parent declares dep again after resetting.
    assertThat(entry.checkIfDoneForDirtyReverseDep(resetDirtyParent))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    entry.markRebuilding();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);

    // Rdep removed after rewinding was initiated.
    entry.removeReverseDep(lateOldParent);

    // Dirty rdep registered after rewinding was initiated.
    assertThat(entry.checkIfDoneForDirtyReverseDep(lateDirtyParent))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);

    // New rdep registered after rewinding was initiated.
    SkyKey lateNewParent = key("lateNewParent");
    assertThat(entry.addReverseDepAndCheckIfDone(lateNewParent))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);

    // Rewound evaluation completes. Only parents that are waiting on the node (registered after
    // rewinding was initiated) are signalled.
    SkyValue newValue = valueChanges ? new IntegerValue(2) : oldValue;
    assertThat(setValue(entry, newValue, /* errorInfo= */ null, incrementalVersion))
        .containsExactly(resetDirtyParent, lateDirtyParent, lateNewParent);
    assertThat(entry.getValue()).isEqualTo(newValue);
    if (valueChanges) {
      assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
    } else {
      assertThat(entry.getVersion()).isEqualTo(initialVersion); // Change pruning.
    }

    // Check rdep accounting.
    assertThat(entry.getReverseDepsForDoneEntry())
        .containsExactly(
            cleanParent,
            earlyDirtyParent,
            earlyNewParent,
            resetDirtyParent,
            lateDirtyParent,
            lateNewParent);
  }

  @Test
  public void rewindOnDirtyNodeIgnored(@TestParameter boolean valueChanges)
      throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();

    // Dep added on initial build that stays on incremental build.
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);

    // Initial build completes.
    SkyValue oldValue = new IntegerValue(1);
    setValue(entry, oldValue, /* errorInfo= */ null, initialVersion);

    // Start of incremental build.
    entry.markDirty(DirtyType.DIRTY);
    entry.addReverseDepAndCheckIfDone(null);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);

    // Attempt to rewind node while it is dirty. Its lifecycle state does not change.
    entry.markDirty(DirtyType.REWIND);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.CHECK_DEPENDENCIES);

    // Add back same dep.
    entry.addSingletonTemporaryDirectDep(dep);
    assertThat(entry.signalDep(incrementalVersion, dep)).isTrue();
    entry.markRebuilding();
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(dep));

    // Set value and check version.
    SkyValue newValue = valueChanges ? new IntegerValue(2) : oldValue;
    setValue(entry, newValue, /* errorInfo= */ null, incrementalVersion);

    if (valueChanges) {
      assertThat(entry.getVersion()).isEqualTo(incrementalVersion);
    } else {
      assertThat(entry.getVersion()).isEqualTo(initialVersion); // Change pruning.
    }
  }
}
