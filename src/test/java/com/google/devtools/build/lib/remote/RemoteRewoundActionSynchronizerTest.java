// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteRewoundActionSynchronizer}. */
@RunWith(JUnit4.class)
public final class RemoteRewoundActionSynchronizerTest {
  private RemoteActionInputFetcher actionInputFetcher;
  private RemoteRewoundActionSynchronizer synchronizer;

  @Before
  public void setUp() {
    actionInputFetcher = mock(RemoteActionInputFetcher.class);
    synchronizer = new RemoteRewoundActionSynchronizer(actionInputFetcher);
  }

  @Test
  public void rewind_cancelsAllRegisteredTasksOnce() throws Exception {
    Action action = newAction();
    var first = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var second = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var unusedFirst = synchronizer.registerOutputUploadTask(action, first);
    var unusedSecond = synchronizer.registerOutputUploadTask(action, second);

    rewind(action);
    rewind(action);

    verify(first, times(1)).cancel();
    verify(second, times(1)).cancel();
  }

  @Test
  public void unregisterHandle_preventsCancellation() throws Exception {
    Action action = newAction();
    var task = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    Runnable unregister = synchronizer.registerOutputUploadTask(action, task);

    unregister.run();
    rewind(action);

    verify(task, never()).cancel();
    assertThat(synchronizer.hasRegisteredOutputUploadTasks(action)).isFalse();
  }

  @Test
  public void predecessorUnregisterHandle_doesNotRemoveReplacementTask() throws Exception {
    Action action = newAction();
    var predecessor = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    Runnable unregisterPredecessor = synchronizer.registerOutputUploadTask(action, predecessor);
    rewind(action);
    var replacement = mock(RemoteRewoundActionSynchronizer.Cancellable.class);
    var unused = synchronizer.registerOutputUploadTask(action, replacement);

    unregisterPredecessor.run();
    rewind(action);

    verify(predecessor, times(1)).cancel();
    verify(replacement, times(1)).cancel();
  }

  @Test
  public void unregisterHandle_usesIdentityComparison() throws Exception {
    Action action = newAction();
    var first = new EqualCancellable();
    var second = new EqualCancellable();
    Runnable unregisterFirst = synchronizer.registerOutputUploadTask(action, first);
    var unused = synchronizer.registerOutputUploadTask(action, second);

    unregisterFirst.run();
    rewind(action);

    assertThat(first.cancellations.get()).isEqualTo(0);
    assertThat(second.cancellations.get()).isEqualTo(1);
  }

  private void rewind(Action action) throws Exception {
    try (SilentCloseable ignored = synchronizer.enterActionPreparation(action, true)) {
      // Cancellation happens while entering preparation.
    }
  }

  private static Action newAction() {
    Action action = mock(Action.class);
    DerivedArtifact output = mock(DerivedArtifact.class);
    when(output.getGeneratingActionKey()).thenReturn(mock(ActionLookupData.class));
    when(action.getPrimaryOutput()).thenReturn(output);
    when(action.getOutputs()).thenReturn(ImmutableList.of(output));
    return action;
  }

  private static final class EqualCancellable
      implements RemoteRewoundActionSynchronizer.Cancellable {
    private final AtomicInteger cancellations = new AtomicInteger();

    @Override
    public void cancel() {
      cancellations.incrementAndGet();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof EqualCancellable;
    }

    @Override
    public int hashCode() {
      return 1;
    }
  }
}
