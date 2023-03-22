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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.skyframe.NodeEntrySubjectFactory.assertThatNodeEntry;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link InMemoryNodeEntry}. */
@RunWith(TestParameterInjector.class)
public final class InMemoryNodeEntryTest {

  @TestParameter public boolean isPartialReevaluation;

  private static final SkyKey REGULAR_KEY = GraphTester.toSkyKey("regular");
  private static final SkyKey PARTIAL_REEVALUATION_KEY =
      new SkyKey() {
        @Override
        public SkyFunctionName functionName() {
          return SkyFunctionName.FOR_TESTING;
        }

        @Override
        public boolean supportsPartialReevaluation() {
          return true;
        }
      };

  private static final NestedSet<Reportable> NO_EVENTS =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  private static SkyKey key(String name) {
    return GraphTester.toSkyKey(name);
  }

  private InMemoryNodeEntry createEntry() {
    return new InMemoryNodeEntry(getEntryKey());
  }

  private SkyKey getEntryKey() {
    return isPartialReevaluation ? PARTIAL_REEVALUATION_KEY : REGULAR_KEY;
  }

  @Test
  public void entryAtStartOfEvaluation() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
  }

  private static final IntVersion ZERO_VERSION = IntVersion.of(0L);
  private static final IntVersion ONE_VERSION = IntVersion.of(1L);

  @Test
  public void signalEntry() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep1 = key("dep1");
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(ZERO_VERSION, dep1)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep1);
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    entry.addSingletonTemporaryDirectDep(dep2);
    entry.addSingletonTemporaryDirectDep(dep3);
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(ZERO_VERSION, dep2)).isFalse();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(ZERO_VERSION, dep3)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L))
        .isEmpty();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getVersion()).isEqualTo(ZERO_VERSION);
    assertThat(entry.getDirectDeps()).containsExactly(dep1, dep2, dep3);
  }

  @Test
  public void signalExternalDep() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    entry.addExternalDep();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(ZERO_VERSION, null)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    entry.addExternalDep();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(ZERO_VERSION, null)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly();
  }

  @Test
  public void reverseDeps() throws InterruptedException {
    NodeEntry entry = createEntry();
    SkyKey mother = key("mother");
    SkyKey father = key("father");
    assertThat(entry.addReverseDepAndCheckIfDone(mother))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.addReverseDepAndCheckIfDone(null))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    assertThat(entry.addReverseDepAndCheckIfDone(father))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    entry.markRebuilding();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L))
        .containsExactly(mother, father);
    assertThat(entry.getReverseDepsForDoneEntry()).containsExactly(mother, father);
    assertThat(entry.isDone()).isTrue();
    entry.removeReverseDep(mother);
    assertThat(entry.getReverseDepsForDoneEntry()).doesNotContain(mother);
  }

  @Test
  public void errorValue() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    assertThat(setValue(entry, /* value= */ null, errorInfo, /* graphVersion= */ 0L)).isEmpty();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getValue()).isNull();
    assertThat(entry.getErrorInfo()).isEqualTo(errorInfo);
  }

  @Test
  public void errorAndValue() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, new SkyValue() {}, errorInfo, /* graphVersion= */ 0L);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getErrorInfo()).isEqualTo(errorInfo);
  }

  @Test
  public void crashOnNullErrorAndValue() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    assertThrows(
        IllegalStateException.class,
        () -> setValue(entry, /* value= */ null, /* errorInfo= */ null, /* graphVersion= */ 0L));
  }

  @Test
  public void crashOnTooManySignals() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    assertThrows(IllegalStateException.class, () -> entry.signalDep(ZERO_VERSION, null));
  }

  @Test
  public void crashOnSetValueWhenDone() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDone()).isTrue();
    assertThrows(
        IllegalStateException.class,
        () -> setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 1L));
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
  public void forceRebuildLifecycle() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(ZERO_VERSION, dep);
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    entry.markDirty(DirtyType.FORCE_REBUILD);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);

    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();

    SkyKey parent = key("parent");
    entry.addReverseDepAndCheckIfDone(parent);
    assertThat(entry.getDirtyState()).isEqualTo(DirtyState.NEEDS_FORCED_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();

    // A force-rebuilt node tolerates evaluating to different values within the same version.
    entry.forceRebuild();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L))
        .containsExactly(parent);

    assertThat(entry.getVersion()).isEqualTo(ZERO_VERSION);
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
  public void allowTwiceMarkedForceRebuild() throws InterruptedException {
    NodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    entry.markDirty(DirtyType.FORCE_REBUILD);
    entry.markDirty(DirtyType.FORCE_REBUILD);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
  }

  @Test
  public void crashOnAddReverseDepTwice() throws InterruptedException {
    NodeEntry entry = createEntry();
    SkyKey parent = key("parent");
    assertThat(entry.addReverseDepAndCheckIfDone(parent))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.addReverseDepAndCheckIfDone(parent);
    entry.markRebuilding();
    IllegalStateException e =
        assertThrows(
            "Cannot add same dep twice",
            IllegalStateException.class,
            () ->
                setValue(entry, new SkyValue() {}, /* errorInfo= */ null, /* graphVersion= */ 0L));
    assertThat(e).hasMessageThat().contains("Duplicate new reverse deps");
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
  public void testClone() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    IntVersion version = IntVersion.of(0);
    IntegerValue originalValue = new IntegerValue(42);
    SkyKey originalChild = key("child");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(null)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();
    entry.addSingletonTemporaryDirectDep(originalChild);
    entry.signalDep(ZERO_VERSION, originalChild);
    entry.setValue(originalValue, version, null);
    entry.addReverseDepAndCheckIfDone(key("parent1"));
    InMemoryNodeEntry clone1 = entry.cloneNodeEntry();
    entry.addReverseDepAndCheckIfDone(key("parent2"));
    InMemoryNodeEntry clone2 = entry.cloneNodeEntry();
    entry.removeReverseDep(key("parent1"));
    entry.removeReverseDep(key("parent2"));
    IntegerValue updatedValue = new IntegerValue(52);
    clone2.markDirty(DirtyType.CHANGE);
    clone2.addReverseDepAndCheckIfDone(null);
    SkyKey newChild = key("newchild");
    clone2.addSingletonTemporaryDirectDep(newChild);
    clone2.signalDep(ONE_VERSION, newChild);
    clone2.markRebuilding();
    clone2.setValue(updatedValue, version.next(), null);

    assertThat(entry.getVersion()).isEqualTo(version);
    assertThat(clone1.getVersion()).isEqualTo(version);
    assertThat(clone2.getVersion()).isEqualTo(version.next());

    assertThat(entry.getValue()).isEqualTo(originalValue);
    assertThat(clone1.getValue()).isEqualTo(originalValue);
    assertThat(clone2.getValue()).isEqualTo(updatedValue);

    assertThat(entry.getDirectDeps()).containsExactly(originalChild);
    assertThat(clone1.getDirectDeps()).containsExactly(originalChild);
    assertThat(clone2.getDirectDeps()).containsExactly(newChild);

    assertThat(entry.getReverseDepsForDoneEntry()).isEmpty();
    assertThat(clone1.getReverseDepsForDoneEntry()).containsExactly(key("parent1"));
    assertThat(clone2.getReverseDepsForDoneEntry()).containsExactly(key("parent1"), key("parent2"));
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

  @Test
  public void addTemporaryDirectDepsInGroups() {
    InMemoryNodeEntry entry = createEntry();
    entry.addTemporaryDirectDepsInGroups(
        ImmutableSet.of(
            key("1A"), key("2A"), key("2B"), key("3A"), key("3B"), key("3C"), key("4A"), key("4B"),
            key("4C"), key("4D")),
        ImmutableList.of(1, 2, 3, 4));
    assertThat(entry.getTemporaryDirectDeps())
        .containsExactly(
            ImmutableList.of(key("1A")),
            ImmutableList.of(key("2A"), key("2B")),
            ImmutableList.of(key("3A"), key("3B"), key("3C")),
            ImmutableList.of(key("4A"), key("4B"), key("4C"), key("4D")))
        .inOrder();
  }

  @Test
  public void addTemporaryDirectDepsInGroups_toleratesEmpty() {
    InMemoryNodeEntry entry = createEntry();
    entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(), ImmutableList.of());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
  }

  @Test
  public void addTemporaryDirectDepsInGroups_toleratesGroupSizeOfZero() {
    InMemoryNodeEntry entry = createEntry();
    entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(key("dep")), ImmutableList.of(0, 1, 0));
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(key("dep")));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_notEnoughGroups_throws() {
    InMemoryNodeEntry entry = createEntry();
    assertThrows(
        RuntimeException.class,
        () ->
            entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(key("dep")), ImmutableList.of()));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_tooManyGroups_throws() {
    InMemoryNodeEntry entry = createEntry();
    assertThrows(
        RuntimeException.class,
        () -> entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(), ImmutableList.of(1)));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_depsLeftOver_throws() {
    InMemoryNodeEntry entry = createEntry();
    assertThrows(
        RuntimeException.class,
        () ->
            entry.addTemporaryDirectDepsInGroups(
                ImmutableSet.of(key("1"), key("2"), key("3")), ImmutableList.of(1, 1)));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_depsExhausted_throws() {
    InMemoryNodeEntry entry = createEntry();
    assertThrows(
        RuntimeException.class,
        () ->
            entry.addTemporaryDirectDepsInGroups(
                ImmutableSet.of(key("1"), key("2"), key("3")), ImmutableList.of(1, 1, 2)));
  }

  private static Set<SkyKey> setValue(
      NodeEntry entry, SkyValue value, @Nullable ErrorInfo errorInfo, long graphVersion)
      throws InterruptedException {
    return entry.setValue(
        ValueWithMetadata.normal(value, errorInfo, NO_EVENTS), IntVersion.of(graphVersion), null);
  }
}
