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
import static com.google.devtools.build.skyframe.NodeEntrySubjectFactory.assertThatNodeEntry;

import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.Version.ConstantVersion;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link NonIncrementalInMemoryNodeEntry}. */
@RunWith(TestParameterInjector.class)
public class NonIncrementalInMemoryNodeEntryTest extends InMemoryNodeEntryTest<ConstantVersion> {

  @Override
  protected NonIncrementalInMemoryNodeEntry createEntry(SkyKey key) {
    return new NonIncrementalInMemoryNodeEntry(key);
  }

  @Override
  final ConstantVersion getInitialVersion() {
    return Version.constant();
  }

  @Test
  public void rewindingLifecycle() throws InterruptedException {
    InMemoryNodeEntry entry = createEntry();
    entry.addReverseDepAndCheckIfDone(null); // Start evaluation.
    entry.markRebuilding();
    setValue(entry, new IntegerValue(1), /* errorInfo= */ null, initialVersion);
    assertThat(entry.isDirty()).isFalse();
    assertThat(entry.isDone()).isTrue();

    entry.markDirty(DirtyType.REWIND);
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
    assertThat(entry.getDirtyState()).isEqualTo(DirtyState.NEEDS_REBUILDING);
    assertThat(entry.isReadyToEvaluate()).isTrue();
    assertThat(entry.hasUnsignaledDeps()).isFalse();
    assertThat(entry.getTemporaryDirectDeps()).isEmpty();

    assertThat(setValue(entry, new IntegerValue(2), /* errorInfo= */ null, initialVersion))
        .containsExactly(parent);
    assertThat(entry.getValue()).isEqualTo(new IntegerValue(2));
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
  }
}
