// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BuildEventStreamer}. */
@RunWith(JUnit4.class)
public class BuildEventStreamerTest {

  private static class RecordingBuildEventTransport implements BuildEventTransport {
    private final List<BuildEvent> events;

    RecordingBuildEventTransport() {
      events = new ArrayList<>();
    }

    @Override
    public void sendBuildEvent(BuildEvent event) {
      events.add(event);
    }

    @Override
    public void close() {}

    List<BuildEvent> getEvents() {
      return events;
    }
  }

  private static class GenericOrderEvent implements BuildEventWithOrderConstraint {
    private final BuildEventId id;
    private final Collection<BuildEventId> children;
    private final Collection<BuildEventId> after;

    GenericOrderEvent(
        BuildEventId id, Collection<BuildEventId> children, Collection<BuildEventId> after) {
      this.id = id;
      this.children = children;
      this.after = after;
    }

    GenericOrderEvent(BuildEventId id, Collection<BuildEventId> children) {
      this(id, children, children);
    }

    @Override
    public BuildEventId getEventId() {
      return id;
    }

    @Override
    public Collection<BuildEventId> getChildrenEvents() {
      return children;
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(PathConverter converter) {
      return GenericBuildEvent.protoChaining(this).build();
    }

    @Override
    public Collection<BuildEventId> postedAfter() {
      return after;
    }
  }

  private static BuildEventId testId(String opaque) {
    return BuildEventId.unknownBuildEventId(opaque);
  }

  @Test
  public void testSimpleStream() {
    // Verify that a well-formed event is passed through and that completion of the
    // build clears the pending progress-update event.

    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"), ImmutableSet.of(ProgressEvent.INITIAL_PROGRESS_UPDATE,
            BuildEventId.buildFinished()));

    streamer.buildEvent(startEvent);

    List<BuildEvent> afterFirstEvent = transport.getEvents();
    assertThat(afterFirstEvent).hasSize(1);
    assertEquals(startEvent.getEventId(), afterFirstEvent.get(0).getEventId());

    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    List<BuildEvent> finalStream = transport.getEvents();
    assertThat(finalStream).hasSize(3);
    assertEquals(BuildEventId.buildFinished(), finalStream.get(1).getEventId());
    assertEquals(ProgressEvent.INITIAL_PROGRESS_UPDATE, finalStream.get(2).getEventId());
  }

  @Test
  public void testChaining() {
    // Verify that unannounced events are linked in with progress update events, assuming
    // a correctly formed initial event.

    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"), ImmutableSet.of(ProgressEvent.INITIAL_PROGRESS_UPDATE));
    BuildEvent unexpectedEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(startEvent);
    streamer.buildEvent(unexpectedEvent);

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(3);
    assertEquals(startEvent.getEventId(), eventsSeen.get(0).getEventId());
    assertEquals(unexpectedEvent.getEventId(), eventsSeen.get(2).getEventId());
    BuildEvent linkEvent = eventsSeen.get(1);
    assertEquals(ProgressEvent.INITIAL_PROGRESS_UPDATE, linkEvent.getEventId());
    assertTrue(
        "Unexpected events should be linked",
        linkEvent.getChildrenEvents().contains(unexpectedEvent.getEventId()));
  }

  @Test
  public void testBadInitialEvent() {
    // Verify that, if the initial event does not announce the initial progress update event,
    // the initial progress event is used instead to chain that event; in this way, new
    // progress updates can always be chained in.

    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEvent unexpectedStartEvent =
        new GenericBuildEvent(testId("unexpected start"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(unexpectedStartEvent);

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(2);
    assertEquals(unexpectedStartEvent.getEventId(), eventsSeen.get(1).getEventId());
    BuildEvent initial = eventsSeen.get(0);
    assertEquals(ProgressEvent.INITIAL_PROGRESS_UPDATE, initial.getEventId());
    assertTrue(
        "Event should be linked",
        initial.getChildrenEvents().contains(unexpectedStartEvent.getEventId()));

    // The initial event should also announce a new progress event; we test this
    // by streaming another unannounced event.

    BuildEvent unexpectedEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(unexpectedEvent);
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(4);
    assertEquals(unexpectedEvent.getEventId(), allEventsSeen.get(3).getEventId());
    BuildEvent secondLinkEvent = allEventsSeen.get(2);
    assertTrue(
        "Progress should have been announced",
        initial.getChildrenEvents().contains(secondLinkEvent.getEventId()));
    assertTrue(
        "Second event should be linked",
        secondLinkEvent.getChildrenEvents().contains(unexpectedEvent.getEventId()));
  }

  @Test
  public void testReferPastEvent() {
    // Verify that, if an event is refers to a previously done event, that duplicated
    // late-referenced event is not expected again.
    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventId.buildFinished()));
    BuildEvent earlyEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());
    BuildEvent lateReference =
        new GenericBuildEvent(testId("late reference"), ImmutableSet.of(earlyEvent.getEventId()));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(earlyEvent);
    streamer.buildEvent(lateReference);
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    List<BuildEvent> eventsSeen = transport.getEvents();
    int earlyEventCount = 0;
    for (BuildEvent event : eventsSeen) {
      if (event.getEventId().equals(earlyEvent.getEventId())) {
        earlyEventCount++;
      }
    }
    // The early event should be reported precisely once.
    assertEquals(1, earlyEventCount);
  }

  @Test
  public void testReodering() {
    // Verify that an event requiring to be posted after another one is indeed.

    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEventId expectedId = testId("the target");
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE, expectedId));
    BuildEvent rootCause =
        new GenericBuildEvent(testId("failure event"), ImmutableSet.<BuildEventId>of());
    BuildEvent failedTarget =
        new GenericOrderEvent(expectedId, ImmutableSet.<BuildEventId>of(rootCause.getEventId()));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(failedTarget);
    streamer.buildEvent(rootCause);

    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(4);
    assertEquals(startEvent.getEventId(), allEventsSeen.get(0).getEventId());
    BuildEvent linkEvent = allEventsSeen.get(1);
    assertEquals(ProgressEvent.INITIAL_PROGRESS_UPDATE, linkEvent.getEventId());
    assertEquals(rootCause.getEventId(), allEventsSeen.get(2).getEventId());
    assertEquals(failedTarget.getEventId(), allEventsSeen.get(3).getEventId());
  }

  @Test
  public void testMissingPrerequisits() {
    // Verify that an event where the prerequisite is never coming till the end of
    // the build still gets posted, with the prerequisite aborted.

    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEventId expectedId = testId("the target");
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE, expectedId,
                BuildEventId.buildFinished()));
    BuildEventId rootCauseId = testId("failure event");
    BuildEvent failedTarget =
        new GenericOrderEvent(expectedId, ImmutableSet.<BuildEventId>of(rootCauseId));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(failedTarget);
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(6);
    assertEquals(startEvent.getEventId(), allEventsSeen.get(0).getEventId());
    assertEquals(BuildEventId.buildFinished(), allEventsSeen.get(1).getEventId());
    BuildEvent linkEvent = allEventsSeen.get(2);
    assertEquals(ProgressEvent.INITIAL_PROGRESS_UPDATE, linkEvent.getEventId());
    assertEquals(rootCauseId, allEventsSeen.get(3).getEventId());
    assertEquals(failedTarget.getEventId(), allEventsSeen.get(4).getEventId());
  }

  @Test
  public void testVeryFirstEventNeedsToWait() {
    // Verify that we can handle an first event waiting for another event.
    RecordingBuildEventTransport transport = new RecordingBuildEventTransport();
    BuildEventStreamer streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport));

    BuildEventId initialId = testId("Initial");
    BuildEventId waitId = testId("Waiting for initial event");
    BuildEvent startEvent =
        new GenericBuildEvent(
            initialId,
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE, waitId));
    BuildEvent waitingForStart =
        new GenericOrderEvent(waitId, ImmutableSet.<BuildEventId>of(), ImmutableSet.of(initialId));

    streamer.buildEvent(waitingForStart);
    streamer.buildEvent(startEvent);

    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(2);
    assertEquals(startEvent.getEventId(), allEventsSeen.get(0).getEventId());
    assertEquals(waitingForStart.getEventId(), allEventsSeen.get(1).getEventId());
  }
}
