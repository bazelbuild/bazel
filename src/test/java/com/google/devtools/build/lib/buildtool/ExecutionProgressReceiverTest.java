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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog.InactivityMonitor;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.ManualSleeper;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionProgressReceiver}. */
@RunWith(JUnit4.class)
public final class ExecutionProgressReceiverTest {
  private final ExecutionProgressReceiver receiver =
      new ExecutionProgressReceiver(/* exclusiveTestsCount= */ 0, new EventBus());
  private final ManualClock clock = new ManualClock();
  private final ManualSleeper sleeper = new ManualSleeper(clock);
  private final InactivityMonitor inactivityMonitor =
      receiver.createInactivityMonitor(ActionExecutionStatusReporter.create(event -> {}), sleeper);

  @Test
  public void completedAction_notInFlight() {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;

    receiver.enqueueing(action);
    assertThat(receiver.hasActionsInFlight()).isTrue();
    assertThat(receiver.getProgressString()).isEqualTo("[0 / 1]");

    receiver.actionCompleted(action);
    assertThat(receiver.hasActionsInFlight()).isFalse();
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
  }

  @Test
  public void rewoundAction_inFlightUntilCompletedAgain() {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.enqueueing(action);
    receiver.actionCompleted(action);

    receiver.dirtied(action, DirtyType.REWIND);
    assertThat(receiver.hasActionsInFlight()).isTrue();
    assertThat(receiver.getProgressString()).isEqualTo("[0 / 1]");

    receiver.actionCompleted(action);
    assertThat(receiver.hasActionsInFlight()).isFalse();
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
  }

  @Test
  public void otherDirtyTypes_ignored() {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.enqueueing(action);
    receiver.actionCompleted(action);

    receiver.dirtied(action, DirtyType.DIRTY);
    receiver.dirtied(action, DirtyType.CHANGE);

    assertThat(receiver.hasActionsInFlight()).isFalse();
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
  }

  @Test
  public void inactivityMonitor_completionNotHiddenByRewinding() throws Exception {
    ActionLookupData rewoundAction = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    ActionLookupData otherAction = ActionsTestUtil.YET_ANOTHER_NULL_ACTION_LOOKUP_DATA;
    receiver.actionCompleted(rewoundAction);
    receiver.enqueueing(otherAction);
    sleeper.scheduleRunnable(
        () -> {
          receiver.dirtied(rewoundAction, DirtyType.REWIND);
          receiver.actionCompleted(otherAction);
        },
        1000);

    assertThat(inactivityMonitor.waitForNextCompletion(5)).isEqualTo(1);
    assertThat(clock.currentTimeMillis()).isEqualTo(1000);
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 2]");
    assertThat(receiver.hasActionsInFlight()).isTrue();
  }

  @Test
  public void inactivityMonitor_repeatedReexecutionsBetweenPollsCounted() throws Exception {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.actionCompleted(action);
    sleeper.scheduleRunnable(
        () -> {
          receiver.dirtied(action, DirtyType.REWIND);
          receiver.actionCompleted(action);
          receiver.dirtied(action, DirtyType.REWIND);
          receiver.actionCompleted(action);
        },
        1000);

    assertThat(inactivityMonitor.waitForNextCompletion(5)).isEqualTo(2);
    assertThat(clock.currentTimeMillis()).isEqualTo(1000);
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
    assertThat(receiver.hasActionsInFlight()).isFalse();
  }

  @Test
  public void inactivityMonitor_rewindingWithoutCompletionTimesOut() throws Exception {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.actionCompleted(action);
    sleeper.scheduleRunnable(() -> receiver.dirtied(action, DirtyType.REWIND), 1000);

    assertThat(inactivityMonitor.waitForNextCompletion(3)).isEqualTo(0);
    assertThat(clock.currentTimeMillis()).isEqualTo(3000);
    assertThat(receiver.getProgressString()).isEqualTo("[0 / 1]");
    assertThat(receiver.hasActionsInFlight()).isTrue();
  }

  @Test
  public void inactivityMonitor_executionAndEvaluationCountedOnce() throws Exception {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.enqueueing(action);
    sleeper.scheduleRunnable(() -> receiver.actionCompleted(action), 1000);

    assertThat(inactivityMonitor.waitForNextCompletion(5)).isEqualTo(1);
    sleeper.scheduleRunnable(
        () ->
            receiver.evaluated(
                action,
                EvaluationState.SUCCESS_VERSION_CHANGED,
                /* newValue= */ null,
                /* newError= */ null,
                /* directDeps= */ null),
        1000);

    assertThat(inactivityMonitor.waitForNextCompletion(3)).isEqualTo(0);
    assertThat(clock.currentTimeMillis()).isEqualTo(4000);
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
    assertThat(receiver.hasActionsInFlight()).isFalse();
  }

  @Test
  public void inactivityMonitor_cachedActionEvaluationCounted() throws Exception {
    ActionLookupData action = ActionsTestUtil.NULL_ACTION_LOOKUP_DATA;
    receiver.enqueueing(action);
    sleeper.scheduleRunnable(
        () ->
            receiver.evaluated(
                action,
                EvaluationState.SUCCESS_VERSION_UNCHANGED,
                /* newValue= */ null,
                /* newError= */ null,
                /* directDeps= */ null),
        1000);

    assertThat(inactivityMonitor.waitForNextCompletion(5)).isEqualTo(1);
    assertThat(clock.currentTimeMillis()).isEqualTo(1000);
    assertThat(receiver.getProgressString()).isEqualTo("[1 / 1]");
    assertThat(receiver.hasActionsInFlight()).isFalse();
  }
}
