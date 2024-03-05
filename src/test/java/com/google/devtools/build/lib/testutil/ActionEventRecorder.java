// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionResultReceivedEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewindingStats;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewoundEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

/** Records various action-related events for tests. */
public final class ActionEventRecorder {

  private final List<ActionStartedEvent> actionStartedEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionCompletionEvent> actionCompletionEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionExecutedEvent> actionExecutedEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionResultReceivedEvent> actionResultReceivedEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionMiddlemanEvent> actionMiddlemanEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<CachedActionEvent> cachedActionEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionRewoundEvent> actionRewoundEvents =
      Collections.synchronizedList(new ArrayList<>());
  private final List<ActionRewindingStats> actionRewindingStatsPosts =
      Collections.synchronizedList(new ArrayList<>());

  private Consumer<ActionRewoundEvent> actionRewoundEventSubscriber = e -> {};

  public void setActionRewoundEventSubscriber(Consumer<ActionRewoundEvent> subscriber) {
    actionRewoundEventSubscriber = subscriber;
  }

  public List<ActionStartedEvent> getActionStartedEvents() {
    return actionStartedEvents;
  }

  public List<ActionCompletionEvent> getActionCompletionEvents() {
    return actionCompletionEvents;
  }

  public List<ActionExecutedEvent> getActionExecutedEvents() {
    return actionExecutedEvents;
  }

  public List<ActionResultReceivedEvent> getActionResultReceivedEvents() {
    return actionResultReceivedEvents;
  }

  public List<ActionRewoundEvent> getActionRewoundEvents() {
    return actionRewoundEvents;
  }

  public List<ActionRewindingStats> getActionRewindingStatsPosts() {
    return actionRewindingStatsPosts;
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionStarted(ActionStartedEvent event) {
    actionStartedEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionCompleted(ActionCompletionEvent event) {
    actionCompletionEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionExecuted(ActionExecutedEvent event) {
    actionExecutedEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionResultReceived(ActionResultReceivedEvent event) {
    actionResultReceivedEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionMiddleman(ActionMiddlemanEvent event) {
    actionMiddlemanEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void cachedAction(CachedActionEvent event) {
    cachedActionEvents.add(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionRewound(ActionRewoundEvent event) {
    actionRewoundEvents.add(event);
    actionRewoundEventSubscriber.accept(event);
  }

  @SuppressWarnings("unused")
  @Subscribe
  @AllowConcurrentEvents
  void actionRewindingStats(ActionRewindingStats actionRewindingStats) {
    actionRewindingStatsPosts.add(actionRewindingStats);
  }

  public void clear() {
    actionStartedEvents.clear();
    actionCompletionEvents.clear();
    actionExecutedEvents.clear();
    actionResultReceivedEvents.clear();
    actionMiddlemanEvents.clear();
    cachedActionEvents.clear();
    actionRewoundEvents.clear();
    actionRewindingStatsPosts.clear();
  }

  /**
   * Check how many of each type of event was emitted during a successful build.
   *
   * @param runOnce Actions which ran and are not rewound
   * @param completedRewound Actions which ran and then are rewound by a later failed action
   * @param failedRewound Actions which fail because of lost inputs and which rewind themselves and
   *     the actions that generate those lost inputs
   * @param exactlyOneMiddlemanEventChecks A list of predicates which should be satisfied exactly
   *     once by the sequence of middleman events emitted
   */
  public void assertEvents(
      ImmutableList<String> runOnce,
      ImmutableList<String> completedRewound,
      ImmutableList<String> failedRewound,
      ImmutableList<Predicate<ActionMiddlemanEvent>> exactlyOneMiddlemanEventChecks,
      ImmutableList<Integer> actionRewindingPostLostInputCounts) {
    assertEvents(
        runOnce,
        completedRewound,
        failedRewound,
        exactlyOneMiddlemanEventChecks,
        /* expectResultReceivedForFailedRewound= */ true,
        actionRewindingPostLostInputCounts);
  }

  /**
   * Like {@link #assertEvents(ImmutableList, ImmutableList, ImmutableList, ImmutableList,
   * ImmutableList)}. The {@code expectResultReceivedForFailedRewound} should be true iff the failed
   * rewound actions ever successfully complete.
   *
   * @param expectResultReceivedForFailedRewound whether the failed rewound actions ever
   *     successfully complete, because no {@link ActionResultReceivedEvent} is emitted for a failed
   *     action
   */
  public void assertEvents(
      ImmutableList<String> runOnce,
      ImmutableList<String> completedRewound,
      ImmutableList<String> failedRewound,
      ImmutableList<Predicate<ActionMiddlemanEvent>> exactlyOneMiddlemanEventChecks,
      boolean expectResultReceivedForFailedRewound,
      ImmutableList<Integer> actionRewindingPostLostInputCounts) {
    EventCountAsserter eventCountAsserter =
        new EventCountAsserter(runOnce, completedRewound, failedRewound);

    eventCountAsserter.assertEventCounts(
        /* events= */ actionStartedEvents,
        /* eventsName= */ "actionStartedEvents",
        /* converter= */ e -> progressMessageOrPrettyPrint(e.getAction()),
        /* expectedRunOnceEventCount= */ 1,
        /* expectedCompletedRewoundEventCount= */ 1,
        /* expectedFailedRewoundEventCount= */ 2);

    eventCountAsserter.assertEventCounts(
        /*events=*/ actionCompletionEvents,
        /*eventsName=*/ "actionCompletionEvents",
        /*converter=*/ e -> progressMessageOrPrettyPrint(e.getAction()),
        /*expectedRunOnceEventCount=*/ 1,
        /*expectedCompletedRewoundEventCount=*/ 1,
        /*expectedFailedRewoundEventCount=*/ 1);

    eventCountAsserter.assertEventCounts(
        /*events=*/ actionExecutedEvents,
        /*eventsName=*/ "actionExecutedEvents",
        /*converter=*/ e -> progressMessageOrPrettyPrint(e.getAction()),
        /*expectedRunOnceEventCount=*/ 1,
        /*expectedCompletedRewoundEventCount=*/ 1,
        /*expectedFailedRewoundEventCount=*/ 1);

    eventCountAsserter.assertEventCounts(
        /*events=*/ actionResultReceivedEvents,
        /*eventsName=*/ "actionResultReceivedEvents",
        /*converter=*/ e -> progressMessageOrPrettyPrint(e.getAction()),
        /*expectedRunOnceEventCount=*/ 1,
        /*expectedCompletedRewoundEventCount=*/ 1,
        /*expectedFailedRewoundEventCount=*/ expectResultReceivedForFailedRewound ? 1 : 0);

    eventCountAsserter.assertEventCounts(
        /*events=*/ actionRewoundEvents,
        /*eventsName=*/ "actionRewoundEvents",
        /*converter=*/ e -> progressMessageOrPrettyPrint(e.getFailedRewoundAction()),
        /*expectedRunOnceEventCount=*/ 0,
        /*expectedCompletedRewoundEventCount=*/ 0,
        /*expectedFailedRewoundEventCount=*/ 1);

    assertTotalLostInputCountsFromStats(actionRewindingPostLostInputCounts);

    for (Predicate<ActionMiddlemanEvent> check : exactlyOneMiddlemanEventChecks) {
      assertThat(actionMiddlemanEvents.stream().filter(check)).hasSize(1);
    }
    assertThat(cachedActionEvents).isEmpty();
  }

  /**
   * Asserts that the total lost input counts from posted {@link ActionRewindingStats} matches
   * expected results.
   *
   * @param totalLostInputCounts - The list of the counts of all lost inputs logged with each {@link
   *     ActionRewindingStats} post.
   */
  public void assertTotalLostInputCountsFromStats(ImmutableList<Integer> totalLostInputCounts) {
    assertThat(actionRewindingStatsPosts).hasSize(totalLostInputCounts.size());
    for (int postIndex = 0; postIndex < totalLostInputCounts.size(); postIndex++) {
      ActionRewindingStats actionRewindingStats = actionRewindingStatsPosts.get(postIndex);
      assertThat(actionRewindingStats.lostInputsCount())
          .isEqualTo(totalLostInputCounts.get(postIndex));
    }
  }

  /**
   * Asserts that the total lost output counts from posted {@link ActionRewindingStats} matches
   * expected results.
   *
   * @param totalLostOutputCounts - The list of the counts of all lost outputs logged with each
   *     {@link ActionRewindingStats} post.
   */
  public void assertTotalLostOutputCountsFromStats(ImmutableList<Integer> totalLostOutputCounts) {
    assertThat(actionRewindingStatsPosts).hasSize(totalLostOutputCounts.size());
    for (int postIndex = 0; postIndex < totalLostOutputCounts.size(); postIndex++) {
      ActionRewindingStats actionRewindingStats = actionRewindingStatsPosts.get(postIndex);
      assertThat(actionRewindingStats.lostOutputsCount())
          .isEqualTo(totalLostOutputCounts.get(postIndex));
    }
  }

  private static final class EventCountAsserter {
    private final ImmutableList<String> runOnce;
    private final ImmutableList<String> completedRewound;
    private final ImmutableList<String> failedRewound;

    private EventCountAsserter(
        ImmutableList<String> runOnce,
        ImmutableList<String> completedRewound,
        ImmutableList<String> failedRewound) {
      this.runOnce = runOnce;
      this.completedRewound = completedRewound;
      this.failedRewound = failedRewound;
    }

    private <T> void assertEventCounts(
        List<T> events,
        String eventsName,
        Function<T, String> converter,
        int expectedRunOnceEventCount,
        int expectedCompletedRewoundEventCount,
        int expectedFailedRewoundEventCount) {
      ImmutableList<String> eventDescriptions =
          events.stream().map(converter).collect(toImmutableList());
      for (String runOnceAction : runOnce) {
        assertWithMessage("Run-once action \"%s\" in %s", runOnceAction, eventsName)
            .that(eventDescriptions.stream().filter(d -> d.equals(runOnceAction)).count())
            .isEqualTo(expectedRunOnceEventCount);
      }
      for (String rewoundAction : completedRewound) {
        assertWithMessage("Completed rewound action \"%s\" in %s", rewoundAction, eventsName)
            .that(eventDescriptions.stream().filter(d -> d.equals(rewoundAction)).count())
            .isEqualTo(expectedCompletedRewoundEventCount);
      }
      for (String failedRewoundAction : failedRewound) {
        assertWithMessage("Failed rewound action \"%s\" in %s", failedRewoundAction, eventsName)
            .that(eventDescriptions.stream().filter(d -> d.equals(failedRewoundAction)).count())
            .isEqualTo(expectedFailedRewoundEventCount);
      }
    }
  }

  public static String progressMessageOrPrettyPrint(ActionExecutionMetadata action) {
    String progressMessage = action.getProgressMessage();
    return progressMessage != null ? progressMessage : action.prettyPrint();
  }
}
