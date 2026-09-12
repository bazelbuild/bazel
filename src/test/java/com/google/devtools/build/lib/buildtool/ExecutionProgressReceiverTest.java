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
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ExecutionProgressReceiver}. */
@RunWith(JUnit4.class)
public final class ExecutionProgressReceiverTest {
  private final ExecutionProgressReceiver receiver =
      new ExecutionProgressReceiver(/* exclusiveTestsCount= */ 0, new EventBus());

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
}
