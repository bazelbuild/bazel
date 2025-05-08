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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.NodeEntrySubjectFactory.assertThatNodeEntry;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.NodeEntry.LifecycleState;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import com.google.testing.junit.testparameterinjector.TestParameter;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Test;

/**
 * Tests for {@link InMemoryNodeEntry} implementations.
 *
 * <p>Contains test cases that are relevant to both {@link IncrementalInMemoryNodeEntry} and {@link
 * NonIncrementalInMemoryNodeEntry}. Test cases that are only partially relevant to one or the other
 * may branch on {@link InMemoryNodeEntry#keepsEdges} and return early.
 *
 * @param <V> The type of {@link Version} used by the {@link InMemoryNodeEntry} class under test
 */
abstract class InMemoryNodeEntryTest<V extends Version> {
  private static final SkyKey REGULAR_KEY = GraphTester.skyKey("regular");
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

  @TestParameter boolean isPartialReevaluation;
  protected final V initialVersion = getInitialVersion();

  static SkyKey key(String name) {
    return GraphTester.skyKey(name);
  }

  final InMemoryNodeEntry createEntry() {
    return createEntry(isPartialReevaluation ? PARTIAL_REEVALUATION_KEY : REGULAR_KEY);
  }

  @ForOverride
  protected abstract InMemoryNodeEntry createEntry(SkyKey key);

  @ForOverride
  abstract V getInitialVersion();

  @Test
  public void entryAtStartOfEvaluation() {
    InMemoryNodeEntry entry = createEntry();
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NOT_YET_EVALUATING);
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
  }

  @Test
  public void signalEntry() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    SkyKey dep1 = key("dep1");
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly(dep1);
    SkyKey dep2 = key("dep2");
    SkyKey dep3 = key("dep3");
    entry.addSingletonTemporaryDirectDep(dep2);
    entry.addSingletonTemporaryDirectDep(dep3);
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(initialVersion, dep2)).isFalse();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(initialVersion, dep3)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion)).isEmpty();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.DONE);
    assertThat(entry.getVersion()).isEqualTo(initialVersion);
    if (!entry.keepsEdges()) {
      return;
    }
    assertThat(entry.getDirectDeps()).containsExactly(dep1, dep2, dep3);
  }

  @Test
  public void signalExternalDep() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    entry.addExternalDep();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(initialVersion, null)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    entry.addExternalDep();
    assertThat(entry.isReadyToEvaluate()).isEqualTo(isPartialReevaluation);
    assertThat(entry.hasUnsignaledDeps()).isTrue();
    assertThat(entry.signalDep(initialVersion, null)).isTrue();
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThatNodeEntry(entry).hasTemporaryDirectDepsThat().containsExactly();
  }

  @Test
  public void reverseDeps() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    SkyKey mother = key("mother");
    SkyKey father = key("father");
    assertThat(entry.addReverseDepAndCheckIfDone(mother))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.addReverseDepAndCheckIfDone(null))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    assertThat(entry.addReverseDepAndCheckIfDone(father))
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    entry.markRebuilding();
    assertThat(setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion))
        .containsExactly(mother, father);
    if (!entry.keepsEdges()) {
      return;
    }
    assertThat(entry.getReverseDepsForDoneEntry()).containsExactly(mother, father);
    assertThat(entry.isDone()).isTrue();
    entry.removeReverseDep(mother);
    assertThat(entry.getReverseDepsForDoneEntry()).doesNotContain(mother);
  }

  @Test
  public void errorValue() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    assertThat(setValue(entry, /* value= */ null, errorInfo, initialVersion)).isEmpty();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.DONE);
    assertThat(entry.getValue()).isNull();
    assertThat(entry.toValue()).isNull();
    assertThat(entry.getErrorInfo()).isEqualTo(errorInfo);
  }

  @Test
  public void errorAndValue() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), Transience.PERSISTENT));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, false);
    setValue(entry, new SkyValue() {}, errorInfo, initialVersion);
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.DONE);
    assertThat(entry.getErrorInfo()).isEqualTo(errorInfo);
  }

  @Test
  public void crashOnNullErrorAndValue() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    assertThrows(
        IllegalStateException.class,
        () -> setValue(entry, /* value= */ null, /* errorInfo= */ null, initialVersion));
  }

  @Test
  public void crashOnTooManySignals() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    assertThrows(IllegalStateException.class, () -> entry.signalDep(initialVersion, null));
  }

  @Test
  public void crashOnSetValueWhenDone() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
    assertThat(entry.isDone()).isTrue();
    assertThrows(
        IllegalStateException.class,
        () -> setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion));
  }

  @Test
  public void crashOnAddReverseDepTwice() {
    InMemoryNodeEntry entry = createEntry();
    SkyKey parent = key("parent");
    assertThat(entry.addReverseDepAndCheckIfDone(parent))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.addReverseDepAndCheckIfDone(parent);
    entry.markRebuilding();
    IllegalStateException e =
        assertThrows(
            "Cannot add same dep twice",
            IllegalStateException.class,
            () -> setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion));
    assertThat(e).hasMessageThat().containsMatch("[Dd]uplicate( new)? reverse deps");
  }

  static final class IntegerValue implements SkyValue {
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

    @Override
    public String toString() {
      return "IntegerValue{" + value + "}";
    }
  }

  @Test
  public void addTemporaryDirectDepsInGroups() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
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
    entry.addReverseDepAndCheckIfDone(null);
    entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(), ImmutableList.of());
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
  }

  @Test
  public void addTemporaryDirectDepsInGroups_toleratesGroupSizeOfZero() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
    entry.addTemporaryDirectDepsInGroups(ImmutableSet.of(key("dep")), ImmutableList.of(0, 1, 0));
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(key("dep")));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_notEnoughGroups_throws() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
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
    entry.addReverseDepAndCheckIfDone(null);
    assertThrows(
        RuntimeException.class,
        () ->
            entry.addTemporaryDirectDepsInGroups(
                ImmutableSet.of(key("1"), key("2"), key("3")), ImmutableList.of(1, 1)));
  }

  @Test
  public void addTemporaryDirectDepsInGroups_depsExhausted_throws() {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null);
    assertThrows(
        RuntimeException.class,
        () ->
            entry.addTemporaryDirectDepsInGroups(
                ImmutableSet.of(key("1"), key("2"), key("3")), ImmutableList.of(1, 1, 2)));
  }

  @Test
  public void resetLifecycle() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    // Rdep added before reset.
    SkyKey parent1 = key("parent1");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(parent1)
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    // Dep added before reset.
    SkyKey dep1 = key("dep1");
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    assertThat(entry.getResetDirectDeps()).isEmpty();
    // Reset clears temporary direct deps.
    entry.resetEvaluationFromScratch();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
    // Rdep added after reset.
    SkyKey parent2 = key("parent2");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(parent2)
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    // Add back same dep.
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(dep1));
    // Dep added after reset.
    SkyKey dep2 = key("dep2");
    entry.addSingletonTemporaryDirectDep(dep2);
    assertThat(entry.signalDep(initialVersion, dep2)).isTrue();
    assertThat(entry.getTemporaryDirectDeps())
        .containsExactly(ImmutableList.of(dep1), ImmutableList.of(dep2));
    // Deps registered before the reset must be tracked if keeping edges.
    if (entry.keepsEdges()) {
      assertThat(entry.getResetDirectDeps()).containsExactly(dep1);
    }
    // Set value and check that both parents will be signaled.
    assertThat(setValue(entry, new IntegerValue(1), /* errorInfo= */ null, initialVersion))
        .containsExactly(parent1, parent2);
  }

  @Test
  public void resetTwice_moreDepsRequestedBeforeFirstReset() throws Exception {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    // Rdep added before any reset.
    SkyKey parent = key("parent");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(parent)
        .isEqualTo(DependencyState.ALREADY_EVALUATING);
    // Two deps added before first reset.
    SkyKey dep1 = key("dep1");
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    SkyKey dep2 = key("dep2");
    entry.addSingletonTemporaryDirectDep(dep2);
    assertThat(entry.signalDep(initialVersion, dep2)).isTrue();
    // First reset.
    entry.resetEvaluationFromScratch();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    // Add back only one dep.
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    assertThat(entry.getTemporaryDirectDeps()).containsExactly(ImmutableList.of(dep1));
    // Second reset.
    entry.resetEvaluationFromScratch();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    // Both deps added back.
    entry.addSingletonTemporaryDirectDep(dep1);
    assertThat(entry.signalDep(initialVersion, dep1)).isTrue();
    entry.addSingletonTemporaryDirectDep(dep2);
    assertThat(entry.signalDep(initialVersion, dep2)).isTrue();
    assertThat(entry.getTemporaryDirectDeps())
        .containsExactly(ImmutableList.of(dep1), ImmutableList.of(dep2));
    // If tracking of reset deps is required, make sure both deps are reported even though only dep1
    // was registered during the most recent evaluation attempt.
    if (entry.keepsEdges()) {
      assertThat(entry.getResetDirectDeps()).containsExactly(dep1, dep2);
    }
    // Set value and check that parent will be signaled.
    assertThat(setValue(entry, new IntegerValue(1), /* errorInfo= */ null, initialVersion))
        .containsExactly(parent);
  }

  @Test
  public void rewindingLifecycle() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    // Rdep that will eventually rewind the entry.
    SkyKey resetParent = key("resetParent");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(resetParent)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();

    // Node completes.
    SkyValue oldValue = new IntegerValue(1);
    assertThat(setValue(entry, oldValue, /* errorInfo= */ null, initialVersion))
        .containsExactly(resetParent);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    // Rewinding initiated.
    entry.markDirty(DirtyType.REWIND);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Parent declares dep again after resetting.
    var dependencyState =
        entry.keepsEdges()
            ? entry.checkIfDoneForDirtyReverseDep(resetParent)
            : entry.addReverseDepAndCheckIfDone(resetParent);
    assertThat(dependencyState).isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    entry.markRebuilding();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);

    // Rewound evaluation completes. The parent that initiated rewinding is signalled.
    SkyValue newValue = new IntegerValue(2);
    assertThat(setValue(entry, newValue, /* errorInfo= */ null, initialVersion))
        .containsExactly(resetParent);
    assertThat(entry.getValue()).isEqualTo(newValue);
    assertThat(entry.toValue()).isEqualTo(newValue);
    assertThat(entry.getVersion()).isEqualTo(initialVersion);
  }

  @Test
  public void resetAfterRewind() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    // Rdep that will eventually rewind the entry.
    SkyKey resetParent = key("resetParent");
    assertThatNodeEntry(entry)
        .addReverseDepAndCheckIfDone(resetParent)
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    entry.markRebuilding();

    // One dep declared.
    SkyKey dep = key("dep");
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);

    // Node completes.
    SkyValue oldValue = new IntegerValue(1);
    assertThat(setValue(entry, oldValue, /* errorInfo= */ null, initialVersion))
        .containsExactly(resetParent);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    // Rewinding initiated.
    entry.markDirty(DirtyType.REWIND);
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getTemporaryDirectDeps() instanceof GroupedDeps.WithHashSet)
        .isEqualTo(isPartialReevaluation);
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Parent declares dep again after resetting.
    var dependencyState =
        entry.keepsEdges()
            ? entry.checkIfDoneForDirtyReverseDep(resetParent)
            : entry.addReverseDepAndCheckIfDone(resetParent);
    assertThat(dependencyState).isEqualTo(DependencyState.NEEDS_SCHEDULING);
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();
    entry.markRebuilding();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.REBUILDING);

    // Dep declared again, then there's a reset.
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    entry.resetEvaluationFromScratch();
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Dep declared again post-reset.
    entry.addSingletonTemporaryDirectDep(dep);
    entry.signalDep(initialVersion, dep);
    assertThat(entry.toValue()).isEqualTo(oldValue);

    // Rewound evaluation completes. The parent that initiated rewinding is signalled.
    SkyValue newValue = new IntegerValue(2);
    assertThat(setValue(entry, newValue, /* errorInfo= */ null, initialVersion))
        .containsExactly(resetParent);
    assertThat(entry.getValue()).isEqualTo(newValue);
    assertThat(entry.toValue()).isEqualTo(newValue);
    assertThat(entry.getVersion()).isEqualTo(initialVersion);
  }

  @Test
  public void concurrentRewindingAllowed() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new SkyValue() {}, /* errorInfo= */ null, initialVersion);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.markDirty(DirtyType.REWIND)).isNotNull();
    assertThat(entry.markDirty(DirtyType.REWIND)).isNull();
    assertThat(entry.isDirty()).isTrue();
    assertThat(entry.isChanged()).isTrue();
    assertThat(entry.isDone()).isFalse();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.NEEDS_REBUILDING);
  }

  @Test
  public void rewindErrorfulNode_toleratedButNoOp(@TestParameter Transience transience)
      throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();

    ReifiedSkyFunctionException exception =
        new ReifiedSkyFunctionException(
            new GenericFunctionException(new SomeErrorException("oops"), transience));
    ErrorInfo errorInfo = ErrorInfo.fromException(exception, transience == Transience.TRANSIENT);
    assertThat(setValue(entry, /* value= */ null, errorInfo, initialVersion)).isEmpty();

    assertThat(entry.markDirty(DirtyType.REWIND)).isNull();
    assertThat(entry.isDone()).isTrue();
    assertThat(entry.getLifecycleState()).isEqualTo(LifecycleState.DONE);
    assertThat(entry.getValue()).isNull();
    assertThat(entry.toValue()).isNull();
    assertThat(entry.getErrorInfo()).isEqualTo(errorInfo);
  }

  @Test
  public void skipsBatchPrefetch_testTemporaryDepsContainsHashSet() {
    InMemoryNodeEntry entry = createEntry(GraphTester.skipBatchPrefetchKey("dropBatchPrefetch"));
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation
    assertThat(entry.getTemporaryDirectDeps()).isInstanceOf(GroupedDeps.WithHashSet.class);
  }

  @CanIgnoreReturnValue
  static Set<SkyKey> setValue(
      NodeEntry entry, SkyValue value, @Nullable ErrorInfo errorInfo, Version graphVersion)
      throws InterruptedException {
    return entry.setValue(
        ValueWithMetadata.normal(value, errorInfo, NO_EVENTS),
        checkNotNull(graphVersion),
        /* maxTransitiveSourceVersion= */ null);
  }
}
