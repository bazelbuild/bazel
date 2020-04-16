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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.ActionExecutedEvent.ErrorTiming;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.actions.SpawnResult.MetadataLog;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import javax.annotation.Nullable;
import org.apache.commons.lang.time.StopWatch;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link BuildEventStreamer}. */
@RunWith(JUnit4.class)
public class BuildEventStreamerTest extends FoundationTestCase {
  private CountingArtifactGroupNamer artifactGroupNamer;
  private RecordingBuildEventTransport transport;
  private BuildEventStreamer streamer;

  @Before
  public void setUp() {
    artifactGroupNamer = new CountingArtifactGroupNamer();
    transport = new RecordingBuildEventTransport(artifactGroupNamer, true);
    streamer =
        new BuildEventStreamer(ImmutableSet.<BuildEventTransport>of(transport), artifactGroupNamer);
  }

  @After
  public void tearDown() {
    artifactGroupNamer = null;
    transport = null;
    streamer = null;
  }

  private static BuildEventContext getTestBuildEventContext(ArtifactGroupNamer artifactGroupNamer) {
    return new BuildEventContext() {
      @Override
      public ArtifactGroupNamer artifactGroupNamer() {
        return artifactGroupNamer;
      }

      @Override
      public PathConverter pathConverter() {
        return new PathConverter() {
          @Override
          public String apply(Path path) {
            return path.toString();
          }
        };
      }

      @Override
      public BuildEventProtocolOptions getOptions() {
        return Options.getDefaults(BuildEventProtocolOptions.class);
      }
    };
  }

  private static final ActionExecutedEvent SUCCESSFUL_ACTION_EXECUTED_EVENT =
      new ActionExecutedEvent(
          ActionsTestUtil.DUMMY_ARTIFACT.getExecPath(),
          new ActionsTestUtil.NullAction(),
          /* exception= */ null,
          ActionsTestUtil.DUMMY_ARTIFACT.getPath(),
          /* stdout= */ null,
          /* stderr= */ null,
          /* actionMetadataLogs= */ ImmutableList.of(),
          ErrorTiming.NO_ERROR,
          /* isInMemoryFs= */ false);

  private static class RecordingBuildEventTransport implements BuildEventTransport {
    private final List<BuildEvent> events = new ArrayList<>();
    private final List<BuildEventStreamProtos.BuildEvent> eventsAsProtos = new ArrayList<>();
    private ArtifactGroupNamer artifactGroupNamer;

    RecordingBuildEventTransport(ArtifactGroupNamer namer, boolean recordEvents) {
      this.artifactGroupNamer = namer;
    }

    @Override
    public String name() {
      return this.getClass().getSimpleName();
    }

    @Override
    public boolean mayBeSlow() {
      return false;
    }

    @Override
    public synchronized void sendBuildEvent(BuildEvent event) {
      events.add(event);
      eventsAsProtos.add(event.asStreamProto(getTestBuildEventContext(this.artifactGroupNamer)));
    }

    @Override
    public ListenableFuture<Void> close() {
      return Futures.immediateFuture(null);
    }

    @Override
    public BuildEventArtifactUploader getUploader() {
      throw new IllegalStateException();
    }

    List<BuildEvent> getEvents() {
      return events;
    }

    List<BuildEventStreamProtos.BuildEvent> getEventProtos() {
      return eventsAsProtos;
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
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
      return GenericBuildEvent.protoChaining(this).build();
    }

    @Override
    public Collection<BuildEventId> postedAfter() {
      return after;
    }
  }

  private static class GenericArtifactReportingEvent implements EventReportingArtifacts {
    private final BuildEventId id;
    private final Collection<BuildEventId> children;
    private final Collection<NestedSet<Artifact>> artifacts;

    GenericArtifactReportingEvent(
        BuildEventId id,
        Collection<BuildEventId> children,
        Collection<NestedSet<Artifact>> artifacts) {
      this.id = id;
      this.children = children;
      this.artifacts = artifacts;
    }

    GenericArtifactReportingEvent(BuildEventId id, Collection<NestedSet<Artifact>> artifacts) {
      this(id, ImmutableSet.<BuildEventId>of(), artifacts);
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
    public ReportedArtifacts reportedArtifacts() {
      return new ReportedArtifacts(artifacts, CompletionContext.FAILED_COMPLETION_CTX);
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
      BuildEventStreamProtos.NamedSetOfFiles.Builder builder =
          BuildEventStreamProtos.NamedSetOfFiles.newBuilder();
      for (NestedSet<Artifact> artifactset : artifacts) {
        builder.addFileSets(
            converters
                .artifactGroupNamer()
                .apply((new NestedSetView<Artifact>(artifactset)).identifier()));
      }
      return GenericBuildEvent.protoChaining(this).setNamedSetOfFiles(builder.build()).build();
    }
  }

  private static class GenericConfigurationEvent implements BuildEventWithConfiguration {
    private final BuildEventId id;
    private final Collection<BuildEventId> children;
    private final Collection<BuildEvent> configurations;

    GenericConfigurationEvent(
        BuildEventId id, Collection<BuildEventId> children, Collection<BuildEvent> configurations) {
      this.id = id;
      this.children = children;
      this.configurations = configurations;
    }

    GenericConfigurationEvent(BuildEventId id, BuildEvent configuration) {
      this(id, ImmutableSet.<BuildEventId>of(), ImmutableSet.of(configuration));
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
    public Collection<BuildEvent> getConfigurations() {
      return configurations;
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
      return GenericBuildEvent.protoChaining(this).build();
    }
  }

  private static BuildEventId testId(String opaque) {
    return BuildEventIdUtil.unknownBuildEventId(opaque);
  }

  private static class EventBusHandler {

    Set<BuildEventTransport> transportSet;

    @Subscribe
    void transportsAnnounced(AnnounceBuildEventTransportsEvent evt) {
      transportSet = Collections.synchronizedSet(new HashSet<>(evt.transports()));
    }

    @Subscribe
    void transportClosed(BuildEventTransportClosedEvent evt) {
      transportSet.remove(evt.transport());
    }
  }

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
  }

  @Test(timeout = 5000)
  public void testSimpleStream() {
    // Verify that a well-formed event is passed through and that completion of the
    // build clears the pending progress-update event. However, there is no guarantee
    // on the order of the flushed events.
    // Additionally, assert that the actual last event has the last_message flag set.

    EventBusHandler handler = new EventBusHandler();
    eventBus.register(handler);
    assertThat(handler.transportSet).isNull();

    eventBus.post(new AnnounceBuildEventTransportsEvent(ImmutableSet.of(transport)));

    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));

    streamer.buildEvent(startEvent);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> afterFirstEvent = transport.getEvents();
    assertThat(afterFirstEvent).hasSize(1);
    assertThat(afterFirstEvent.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(handler.transportSet).hasSize(1);

    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    assertThat(streamer.isClosed()).isTrue();
    eventBus.post(new BuildEventTransportClosedEvent(transport));

    List<BuildEvent> finalStream = transport.getEvents();
    assertThat(finalStream).hasSize(3);
    assertThat(ImmutableSet.of(finalStream.get(1).getEventId(), finalStream.get(2).getEventId()))
        .isEqualTo(
            ImmutableSet.of(
                BuildEventIdUtil.buildFinished(), ProgressEvent.INITIAL_PROGRESS_UPDATE));

    // verify the "last_message" flag.
    assertThat(transport.getEventProtos().get(0).getLastMessage()).isFalse();
    assertThat(transport.getEventProtos().get(1).getLastMessage()).isFalse();
    assertThat(transport.getEventProtos().get(2).getLastMessage()).isTrue();

    while (!handler.transportSet.isEmpty()) {
      LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(100));
    }
  }

  @Test
  public void testChaining() {
    // Verify that unannounced events are linked in with progress update events, assuming
    // a correctly formed initial event.
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"), ImmutableSet.of(ProgressEvent.INITIAL_PROGRESS_UPDATE));
    BuildEvent unexpectedEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(startEvent);
    streamer.buildEvent(unexpectedEvent);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(3);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(eventsSeen.get(2).getEventId()).isEqualTo(unexpectedEvent.getEventId());
    BuildEvent linkEvent = eventsSeen.get(1);
    assertThat(linkEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertWithMessage("Unexpected events should be linked")
        .that(linkEvent.getChildrenEvents().contains(unexpectedEvent.getEventId()))
        .isTrue();
  }

  @Test
  public void testBadInitialEvent() {
    // Verify that, if the initial event does not announce the initial progress update event,
    // the initial progress event is used instead to chain that event; in this way, new
    // progress updates can always be chained in.
    BuildEvent unexpectedStartEvent =
        new GenericBuildEvent(testId("unexpected start"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(unexpectedStartEvent);

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(2);
    assertThat(eventsSeen.get(1).getEventId()).isEqualTo(unexpectedStartEvent.getEventId());
    BuildEvent initial = eventsSeen.get(0);
    assertThat(initial.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertWithMessage("Event should be linked")
        .that(initial.getChildrenEvents().contains(unexpectedStartEvent.getEventId()))
        .isTrue();

    // The initial event should also announce a new progress event; we test this
    // by streaming another unannounced event.

    BuildEvent unexpectedEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());

    streamer.buildEvent(unexpectedEvent);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(4);
    assertThat(allEventsSeen.get(3).getEventId()).isEqualTo(unexpectedEvent.getEventId());
    BuildEvent secondLinkEvent = allEventsSeen.get(2);
    assertWithMessage("Progress should have been announced")
        .that(initial.getChildrenEvents().contains(secondLinkEvent.getEventId()))
        .isTrue();
    assertWithMessage("Second event should be linked")
        .that(secondLinkEvent.getChildrenEvents().contains(unexpectedEvent.getEventId()))
        .isTrue();
  }

  @Test
  public void testReferPastEvent() {
    // Verify that, if an event is refers to a previously done event, that duplicated
    // late-referenced event is not expected again.
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));
    BuildEvent earlyEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());
    BuildEvent lateReference =
        new GenericBuildEvent(testId("late reference"), ImmutableSet.of(earlyEvent.getEventId()));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(earlyEvent);
    streamer.buildEvent(lateReference);
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    assertThat(streamer.isClosed()).isTrue();
    List<BuildEvent> eventsSeen = transport.getEvents();
    int earlyEventCount = 0;
    for (BuildEvent event : eventsSeen) {
      if (event.getEventId().equals(earlyEvent.getEventId())) {
        earlyEventCount++;
      }
    }
    // The early event should be reported precisely once.
    assertThat(earlyEventCount).isEqualTo(1);
  }

  @Test
  public void testReordering() {
    // Verify that an event requiring to be posted after another one is indeed.
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

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(4);
    assertThat(allEventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    BuildEvent linkEvent = allEventsSeen.get(1);
    assertThat(linkEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertThat(allEventsSeen.get(2).getEventId()).isEqualTo(rootCause.getEventId());
    assertThat(allEventsSeen.get(3).getEventId()).isEqualTo(failedTarget.getEventId());
  }

  private static BuildEvent indexOrderedBuildEvent(int index, int afterIndex) {
    return new GenericOrderEvent(
        testId("Concurrent-" + index),
        ImmutableList.of(),
        afterIndex == -1
            ? ImmutableList.of()
            : ImmutableList.of(testId("Concurrent-" + afterIndex)));
  }

  @Test
  public void testConcurrency() throws Exception {
    // Verify that we can blast the BuildEventStreamer with many build events in parallel without
    // violating internal consistency. The thread-safety under test is primarily sensitive to the
    // pendingEvents field constructed when there are ordering constraints, so we make sure to
    // include such ordering constraints in this test.
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));
    streamer.buildEvent(startEvent);

    int numThreads = 12;
    int numEventsPerThread = 10_000;
    int totalEvents = numThreads * numEventsPerThread;
    AtomicInteger idIndex = new AtomicInteger();
    ThreadPoolExecutor pool =
        new ThreadPoolExecutor(
            numThreads,
            numThreads,
            /* keepAliveTime= */ 0,
            TimeUnit.SECONDS,
            /* workQueue= */ new LinkedBlockingQueue<>());

    for (int i = 0; i < numThreads; i++) {
      pool.execute(
          () -> {
            for (int j = 0; j < numEventsPerThread; j++) {
              int index = idIndex.getAndIncrement();
              // Arrange for half of the events to have an ordering constraint on the subsequent
              // event. The ordering graph must avoid cycles.
              int afterIndex = (index % 2 == 0) ? (index + 1) % totalEvents : -1;
              streamer.buildEvent(indexOrderedBuildEvent(index, afterIndex));
            }
          });
    }

    pool.shutdown();
    pool.awaitTermination(1, TimeUnit.DAYS);

    BuildEventId lateId = testId("late event");
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0), ImmutableList.of(lateId)));
    assertThat(streamer.isClosed()).isFalse();
    streamer.buildEvent(new GenericBuildEvent(lateId, ImmutableSet.of()));
    assertThat(streamer.isClosed()).isTrue();

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(eventsSeen).hasSize(4 + totalEvents * 2);
  }

  // Re-enable this "test" for ad-hoc benchmarking of many concurrent build events.
  @Ignore
  public void concurrencyBenchmark() throws Exception {
    long time = 0;
    for (int iteration = 0; iteration < 3; iteration++) {
      StopWatch watch = new StopWatch();
      watch.start();

      transport = new RecordingBuildEventTransport(artifactGroupNamer, /*recordEvents=*/ false);
      streamer =
          new BuildEventStreamer(
              ImmutableSet.<BuildEventTransport>of(transport), artifactGroupNamer);
      BuildEvent startEvent =
          new GenericBuildEvent(
              testId("Initial"),
              ImmutableSet.of(
                  ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));
      streamer.buildEvent(startEvent);

      int numThreads = 12;
      int numEventsPerThread = 100_000;
      int totalEvents = numThreads * numEventsPerThread;
      AtomicInteger idIndex = new AtomicInteger();
      ThreadPoolExecutor pool =
          new ThreadPoolExecutor(
              numThreads, numThreads, 0, TimeUnit.SECONDS, new LinkedBlockingQueue<>());

      for (int i = 0; i < numThreads; i++) {
        pool.execute(
            () -> {
              for (int j = 0; j < numEventsPerThread; j++) {
                int index = idIndex.getAndIncrement();
                // Arrange for half of the events to have an ordering constraint on the subsequent
                // event. The ordering graph must avoid cycles.
                int afterIndex = (index % 2 == 0) ? (index + 1) % totalEvents : -1;
                streamer.buildEvent(indexOrderedBuildEvent(index, afterIndex));
              }
            });
      }

      pool.shutdown();
      pool.awaitTermination(1, TimeUnit.DAYS);
      watch.stop();

      time += watch.getTime();

      BuildEventId lateId = testId("late event");
      streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0), ImmutableList.of(lateId)));
      assertThat(streamer.isClosed()).isFalse();
      streamer.buildEvent(new GenericBuildEvent(lateId, ImmutableSet.of()));
      assertThat(streamer.isClosed()).isTrue();
    }

    System.err.println();
    System.err.println("=============================================================");
    System.err.println("Concurrent performance of BEP build event processing: " + time + "ms");
    System.err.println("=============================================================");
  }

  @Test
  public void testMissingPrerequisites() {
    // Verify that an event where the prerequisite is never coming till the end of
    // the build still gets posted, with the prerequisite aborted.
    BuildEventId expectedId = testId("the target");
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                expectedId,
                BuildEventIdUtil.buildFinished()));
    BuildEventId rootCauseId = testId("failure event");
    BuildEvent failedTarget =
        new GenericOrderEvent(expectedId, ImmutableSet.<BuildEventId>of(rootCauseId));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(failedTarget);
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    assertThat(streamer.isClosed()).isTrue();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(6);
    assertThat(allEventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(allEventsSeen.get(1).getEventId()).isEqualTo(BuildEventIdUtil.buildFinished());
    BuildEvent linkEvent = allEventsSeen.get(2);
    assertThat(linkEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertThat(allEventsSeen.get(3).getEventId()).isEqualTo(rootCauseId);
    assertThat(allEventsSeen.get(4).getEventId()).isEqualTo(failedTarget.getEventId());
  }

  @Test
  public void testVeryFirstEventNeedsToWait() {
    // Verify that we can handle an first event waiting for another event.
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

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(2);
    assertThat(allEventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(allEventsSeen.get(1).getEventId()).isEqualTo(waitingForStart.getEventId());
  }

  private Artifact makeArtifact(String pathString) {
    Path path = outputBase.getRelative(PathFragment.create(pathString));
    return ActionsTestUtil.createArtifact(
        ArtifactRoot.asSourceRoot(Root.fromPath(outputBase)), path);
  }

  @Test
  public void testReportedArtifacts() {
    // Verify that reported artifacts are correctly unfolded into the stream
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));

    Artifact a = makeArtifact("path/a");
    Artifact b = makeArtifact("path/b");
    Artifact c = makeArtifact("path/c");
    NestedSet<Artifact> innerGroup = NestedSetBuilder.<Artifact>stableOrder().add(a).add(b).build();
    NestedSet<Artifact> group =
        NestedSetBuilder.<Artifact>stableOrder().addTransitive(innerGroup).add(c).build();
    BuildEvent reportingArtifacts =
        new GenericArtifactReportingEvent(testId("reporting"), ImmutableSet.of(group));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(reportingArtifacts);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    List<BuildEventStreamProtos.BuildEvent> eventProtos = transport.getEventProtos();
    assertThat(allEventsSeen).hasSize(7);
    assertThat(allEventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(allEventsSeen.get(1).getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    List<BuildEventStreamProtos.File> firstSetDirects =
        eventProtos.get(2).getNamedSetOfFiles().getFilesList();
    assertThat(firstSetDirects).hasSize(2);
    assertThat(ImmutableSet.of(firstSetDirects.get(0).getUri(), firstSetDirects.get(1).getUri()))
        .isEqualTo(ImmutableSet.of(a.getPath().toString(), b.getPath().toString()));
    List<NamedSetOfFilesId> secondSetTransitives =
        eventProtos.get(4).getNamedSetOfFiles().getFileSetsList();
    assertThat(secondSetTransitives).hasSize(1);
    assertThat(secondSetTransitives.get(0)).isEqualTo(eventProtos.get(2).getId().getNamedSet());
    List<NamedSetOfFilesId> reportedArtifactSets =
        eventProtos.get(6).getNamedSetOfFiles().getFileSetsList();
    assertThat(reportedArtifactSets).hasSize(1);
    assertThat(reportedArtifactSets.get(0)).isEqualTo(eventProtos.get(4).getId().getNamedSet());
  }

  @Test
  public void testStdoutReported() {
    // Verify that stdout and stderr are reported in the build-event stream on progress
    // events.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String stdoutMsg = "Some text that was written to stdout.";
    String stderrMsg = "The UI text that bazel wrote to stderr.";
    when(outErr.getOut()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stdoutMsg)));
    when(outErr.getErr()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stderrMsg)));
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));
    BuildEvent unexpectedEvent =
        new GenericBuildEvent(testId("unexpected"), ImmutableSet.<BuildEventId>of());

    streamer.registerOutErrProvider(outErr);
    streamer.buildEvent(startEvent);
    streamer.buildEvent(unexpectedEvent);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(3);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(eventsSeen.get(2).getEventId()).isEqualTo(unexpectedEvent.getEventId());
    BuildEvent linkEvent = eventsSeen.get(1);
    BuildEventStreamProtos.BuildEvent linkEventProto = transport.getEventProtos().get(1);
    assertThat(linkEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertWithMessage("Unexpected events should be linked")
        .that(linkEvent.getChildrenEvents().contains(unexpectedEvent.getEventId()))
        .isTrue();
    assertThat(linkEventProto.getProgress().getStdout()).isEqualTo(stdoutMsg);
    assertThat(linkEventProto.getProgress().getStderr()).isEqualTo(stderrMsg);

    // As there is only one progress event, the OutErrProvider should be queried
    // only once for stdout and stderr.
    verify(outErr, times(1)).getOut();
    verify(outErr, times(1)).getErr();
  }

  @Test
  public void testStdoutReportedAfterCrash() {
    // Verify that stdout and stderr are reported in the build-event stream on progress
    // events.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String stdoutMsg = "Some text that was written to stdout.";
    String stderrMsg = "The UI text that bazel wrote to stderr.";
    when(outErr.getOut()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stdoutMsg)));
    when(outErr.getErr()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stderrMsg)));
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));

    streamer.registerOutErrProvider(outErr);
    streamer.buildEvent(startEvent);
    // Simulate a crash with an abrupt call to #close().
    streamer.close();
    assertThat(streamer.isClosed()).isTrue();

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(2);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    BuildEvent linkEvent = eventsSeen.get(1);
    BuildEventStreamProtos.BuildEvent linkEventProto = transport.getEventProtos().get(1);
    assertThat(linkEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertThat(linkEventProto.getProgress().getStdout()).isEqualTo(stdoutMsg);
    assertThat(linkEventProto.getProgress().getStderr()).isEqualTo(stderrMsg);

    // As there is only one progress event, the OutErrProvider should be queried
    // only once for stdout and stderr.
    verify(outErr, times(1)).getOut();
    verify(outErr, times(1)).getErr();
  }

  // private static ImmutableList<ImmutableList<Pair<String, String>>> consumeToLists(
  //     Iterable<String> left, Iterable<String> right) {
  //   ImmutableList.Builder<Pair<String, String>> consumerBuilder = ImmutableList.builder();
  //   ImmutableList.Builder<Pair<String, String>> lastConsumerBuilder = ImmutableList.builder();
  //
  //   BuildEventStreamer.consumeAsPairs(
  //       left,
  //       right,
  //       (t1, t2) -> consumerBuilder.add(Pair.of(t1, t2)),
  //       (t1, t2) -> lastConsumerBuilder.add(Pair.of(t1, t2)));
  //
  //   return ImmutableList.of(consumerBuilder.build(), lastConsumerBuilder.build());
  // }
  //
  // @Test
  // public void testConsumeAsPairs() {
  //   assertThat(consumeToLists(ImmutableList.of("1", "2", "3"), ImmutableList.of("4", "5", "6")))
  //       .containsExactly(
  //           ImmutableList.of(
  //               Pair.of("1", null), Pair.of("2", null), Pair.of("3", "4"), Pair.of(null, "5")),
  //           ImmutableList.of(Pair.of(null, "6")))
  //       .inOrder();
  //
  //   assertThat(consumeToLists(ImmutableList.of(), ImmutableList.of()))
  //       .containsExactly(ImmutableList.of(), ImmutableList.of(Pair.of(null, null)))
  //       .inOrder();
  //
  //   assertThat(consumeToLists(ImmutableList.of("1"), ImmutableList.of("2")))
  //       .containsExactly(ImmutableList.of(), ImmutableList.of(Pair.of("1", "2")))
  //       .inOrder();
  //
  //   assertThat(consumeToLists(ImmutableList.of("1"), ImmutableList.of("2", "3")))
  //       .containsExactly(ImmutableList.of(Pair.of("1", "2")), ImmutableList.of(Pair.of(null,
  // "3")))
  //       .inOrder();
  //
  //   assertThat(consumeToLists(ImmutableList.of("1", "2"), ImmutableList.of()))
  //       .containsExactly(ImmutableList.of(Pair.of("1", null)), ImmutableList.of(Pair.of("2",
  // null)))
  //       .inOrder();
  //
  //   assertThat(consumeToLists(ImmutableList.of(), ImmutableList.of("1")))
  //       .containsExactly(ImmutableList.of(), ImmutableList.of(Pair.of(null, "1")))
  //       .inOrder();
  // }

  @Test
  public void testReportedConfigurations() throws Exception {
    // Verify that configuration events are posted, but only once.
    BuildOptions defaultBuildOptions =
        BuildOptions.of(ImmutableList.<Class<? extends FragmentOptions>>of(CoreOptions.class));
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));
    BuildConfiguration configuration =
        new BuildConfiguration(
            new BlazeDirectories(
                new ServerDirectories(outputBase, outputBase, outputBase),
                rootDirectory,
                /* defaultSystemJavabase= */ null,
                "productName"),
            /* fragmentsMap= */ ImmutableMap.<Class<? extends Fragment>, Fragment>of(),
            defaultBuildOptions,
            BuildOptions.diffForReconstruction(defaultBuildOptions, defaultBuildOptions),
            /* reservedActionMnemonics= */ ImmutableSet.of(),
            ActionEnvironment.EMPTY,
            "workspace");
    BuildEvent firstWithConfiguration =
        new GenericConfigurationEvent(testId("first"), configuration.toBuildEvent());
    BuildEvent secondWithConfiguration =
        new GenericConfigurationEvent(testId("second"), configuration.toBuildEvent());

    streamer.buildEvent(startEvent);
    streamer.buildEvent(firstWithConfiguration);
    streamer.buildEvent(secondWithConfiguration);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> allEventsSeen = transport.getEvents();
    assertThat(allEventsSeen).hasSize(7);
    assertThat(allEventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(allEventsSeen.get(1).getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    assertThat(allEventsSeen.get(2)).isEqualTo(configuration.toBuildEvent());
    assertThat(allEventsSeen.get(3).getEventId()).isEqualTo(BuildEventIdUtil.progressId(1));
    assertThat(allEventsSeen.get(4)).isEqualTo(firstWithConfiguration);
    assertThat(allEventsSeen.get(5).getEventId()).isEqualTo(BuildEventIdUtil.progressId(2));
    assertThat(allEventsSeen.get(6)).isEqualTo(secondWithConfiguration);
  }

  @Test
  public void testEarlyFlush() throws Exception {
    // Verify that the streamer can handle early calls to flush() and still correctly
    // reports stdout and stderr in the build-event stream.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String firstStdoutMsg = "Some text that was written to stdout.";
    String firstStderrMsg = "The UI text that bazel wrote to stderr.";
    String secondStdoutMsg = "More text that was written to stdout, still before the start event.";
    String secondStderrMsg = "More text written to stderr, still before the start event.";
    when(outErr.getOut())
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(firstStdoutMsg)))
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(secondStdoutMsg)));
    when(outErr.getErr())
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(firstStderrMsg)))
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(secondStderrMsg)));
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));

    streamer.registerOutErrProvider(outErr);
    streamer.flush();
    streamer.flush();
    streamer.buildEvent(startEvent);

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(3);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    BuildEvent progressEvent = eventsSeen.get(1);
    assertThat(progressEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    BuildEventStreamProtos.BuildEvent progressEventProto = transport.getEventProtos().get(1);
    assertThat(progressEventProto.getProgress().getStdout()).isEqualTo(firstStdoutMsg);
    assertThat(progressEventProto.getProgress().getStderr()).isEqualTo(firstStderrMsg);
    BuildEventStreamProtos.BuildEvent secondProgressEventProto = transport.getEventProtos().get(2);
    assertThat(secondProgressEventProto.getProgress().getStdout()).isEqualTo(secondStdoutMsg);
    assertThat(secondProgressEventProto.getProgress().getStderr()).isEqualTo(secondStderrMsg);

    // As there is only one progress event, the OutErrProvider should be queried
    // only once per flush() for stdout and stderr.
    verify(outErr, times(2)).getOut();
    verify(outErr, times(2)).getErr();
  }

  @Test
  public void testChunkedFlush() throws Exception {
    // Verify that the streamer calls to flush() that return multiple chunked buffers.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String firstStdoutMsg = "Some text that was written to stdout.";
    String firstStderrMsg = "The UI text that bazel wrote to stderr.";
    String secondStdoutMsg = "More text that was written to stdout, still before the start event.";
    String secondStderrMsg = "More text written to stderr, still before the start event.";
    when(outErr.getOut())
        .thenReturn(
            ImmutableList.of(
                ByteString.copyFromUtf8(firstStdoutMsg), ByteString.copyFromUtf8(secondStdoutMsg)));
    when(outErr.getErr())
        .thenReturn(
            ImmutableList.of(
                ByteString.copyFromUtf8(firstStderrMsg), ByteString.copyFromUtf8(secondStderrMsg)));
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));

    streamer.registerOutErrProvider(outErr);
    streamer.buildEvent(startEvent);
    streamer.flush();

    assertThat(streamer.isClosed()).isFalse();
    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(4);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());

    // Expect to find 3 progress messages: (firstStdout, ""), (secondStdout, firstStderr),
    // ("", secondStdErr). Assuming UIs display stdout first, this maintains ordering.
    BuildEvent progressEvent = eventsSeen.get(1);
    assertThat(progressEvent.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    BuildEventStreamProtos.BuildEvent progressEventProto = transport.getEventProtos().get(1);
    assertThat(progressEventProto.getProgress().getStdout()).isEqualTo(firstStdoutMsg);
    assertThat(progressEventProto.getProgress().getStderr()).isEmpty();

    BuildEventStreamProtos.BuildEvent secondProgressEventProto = transport.getEventProtos().get(2);
    assertThat(secondProgressEventProto.getProgress().getStdout()).isEqualTo(secondStdoutMsg);
    assertThat(secondProgressEventProto.getProgress().getStderr()).isEqualTo(firstStderrMsg);

    BuildEventStreamProtos.BuildEvent thirdProgressEventProto = transport.getEventProtos().get(3);
    assertThat(thirdProgressEventProto.getProgress().getStdout()).isEmpty();
    assertThat(thirdProgressEventProto.getProgress().getStderr()).isEqualTo(secondStderrMsg);

    // The OutErrProvider should be queried only once per flush().
    verify(outErr, times(1)).getOut();
    verify(outErr, times(1)).getErr();
  }

  @Test
  public void testNoopFlush() throws Exception {
    // Verify that the streamer ignores a flush, if neither stream produces any output.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String stdoutMsg = "Some text that was written to stdout.";
    String stderrMsg = "The UI text that bazel wrote to stderr.";
    when(outErr.getOut())
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stdoutMsg)))
        .thenReturn(ImmutableList.of());
    when(outErr.getErr())
        .thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stderrMsg)))
        .thenReturn(ImmutableList.of());
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.<BuildEventId>of(ProgressEvent.INITIAL_PROGRESS_UPDATE));

    streamer.registerOutErrProvider(outErr);
    streamer.buildEvent(startEvent);
    assertThat(transport.getEvents()).hasSize(1);
    streamer.flush(); // Output, so a new progress event has to be added
    assertThat(transport.getEvents()).hasSize(2);
    streamer.flush(); // No further output, so no additional event should be generated.
    assertThat(transport.getEvents()).hasSize(2);

    assertThat(transport.getEvents().get(0)).isEqualTo(startEvent);
    assertThat(transport.getEventProtos().get(1).getProgress().getStdout()).isEqualTo(stdoutMsg);
    assertThat(transport.getEventProtos().get(1).getProgress().getStderr()).isEqualTo(stderrMsg);
  }

  @Test
  public void testEarlyFlushBadInitialEvent() throws Exception {
    // Verify that an early flush works correctly with an unusual start event.
    // In this case, we expect 3 events in the stream, in that order:
    // - an artificial progress event as initial event, to properly link in
    //   all events
    // - the unusal first event we have seen, and
    // - a progress event reporting the flushed messages.
    BuildEventStreamer.OutErrProvider outErr =
        Mockito.mock(BuildEventStreamer.OutErrProvider.class);
    String stdoutMsg = "Some text that was written to stdout.";
    String stderrMsg = "The UI text that bazel wrote to stderr.";
    when(outErr.getOut()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stdoutMsg)));
    when(outErr.getErr()).thenReturn(ImmutableList.of(ByteString.copyFromUtf8(stderrMsg)));

    BuildEvent unexpectedStartEvent =
        new GenericBuildEvent(testId("unexpected start"), ImmutableSet.<BuildEventId>of());

    streamer.registerOutErrProvider(outErr);
    streamer.flush();
    streamer.buildEvent(unexpectedStartEvent);

    assertThat(streamer.isClosed()).isFalse();

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(3);

    BuildEvent initial = eventsSeen.get(0);
    assertThat(initial.getEventId()).isEqualTo(ProgressEvent.INITIAL_PROGRESS_UPDATE);
    BuildEventStreamProtos.BuildEvent initialProto = transport.getEventProtos().get(0);
    assertThat(initialProto.getProgress().getStdout()).isEmpty();
    assertThat(initialProto.getProgress().getStderr()).isEmpty();

    assertThat(eventsSeen.get(1).getEventId()).isEqualTo(unexpectedStartEvent.getEventId());
    assertWithMessage("Unexpected event should be linked")
        .that(initial.getChildrenEvents().contains(unexpectedStartEvent.getEventId()))
        .isTrue();

    BuildEventStreamProtos.BuildEvent progressProto = transport.getEventProtos().get(2);
    assertThat(progressProto.getProgress().getStdout()).isEqualTo(stdoutMsg);
    assertThat(progressProto.getProgress().getStderr()).isEqualTo(stderrMsg);
    assertWithMessage("flushed progress should be linked")
        .that(initial.getChildrenEvents().contains(eventsSeen.get(2).getEventId()))
        .isTrue();

    verify(outErr, times(1)).getOut();
    verify(outErr, times(1)).getErr();
  }

  @Test
  public void testEarlyAbort() throws Exception {
    // For a build that is aborted before a build-started event is generated,
    // we still expect that, if a build-started event is forced by some order
    // constraint (e.g., CommandLine wants to come after build started), then
    // that gets sorted to the beginning.
    BuildEvent orderEvent =
        new GenericOrderEvent(
            testId("event depending on start"),
            ImmutableList.of(),
            ImmutableList.of(BuildEventIdUtil.buildStartedId()));

    streamer.buildEvent(orderEvent);
    streamer.buildEvent(new BuildCompleteEvent(new BuildResult(0)));

    assertThat(streamer.isClosed()).isTrue();
    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(4);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(BuildEventIdUtil.buildStartedId());
    assertThat(eventsSeen.get(1).getEventId()).isEqualTo(orderEvent.getEventId());
    assertThat(ImmutableSet.of(eventsSeen.get(2).getEventId(), eventsSeen.get(3).getEventId()))
        .isEqualTo(
            ImmutableSet.of(
                BuildEventIdUtil.buildFinished(), ProgressEvent.INITIAL_PROGRESS_UPDATE));
    assertThat(transport.getEventProtos().get(3).getLastMessage()).isTrue();
  }

  @Test
  public void testFinalEventsLate() throws Exception {
    // Verify that we correctly handle late events (i.e., events coming only after the
    // BuildCompleteEvent) that are sent to the streamer after the BuildCompleteEvent.
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));
    BuildEventId lateId = testId("late event");
    BuildEvent finishedEvent = new BuildCompleteEvent(new BuildResult(0), ImmutableList.of(lateId));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(finishedEvent);
    assertThat(streamer.isClosed()).isFalse();
    streamer.buildEvent(new GenericBuildEvent(lateId, ImmutableSet.of()));
    assertThat(streamer.isClosed()).isTrue();

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(4);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(eventsSeen.get(1).getEventId()).isEqualTo(BuildEventIdUtil.buildFinished());
    assertThat(ImmutableSet.of(eventsSeen.get(2).getEventId(), eventsSeen.get(3).getEventId()))
        .isEqualTo(ImmutableSet.of(lateId, ProgressEvent.INITIAL_PROGRESS_UPDATE));
  }

  @Test
  public void testFinalEventsEarly() throws Exception {
    // Verify that we correctly handle late events (i.e., events coming only after the
    // BuildCompleteEvent) that are sent to the streamer before the BuildCompleteEvent,
    // but with an order constraint to come afterwards.
    BuildEvent startEvent =
        new GenericBuildEvent(
            testId("Initial"),
            ImmutableSet.of(
                ProgressEvent.INITIAL_PROGRESS_UPDATE, BuildEventIdUtil.buildFinished()));
    BuildEventId lateId = testId("late event");
    BuildEvent finishedEvent = new BuildCompleteEvent(new BuildResult(0), ImmutableList.of(lateId));

    streamer.buildEvent(startEvent);
    streamer.buildEvent(
        new GenericOrderEvent(
            lateId, ImmutableSet.of(), ImmutableList.of(BuildEventIdUtil.buildFinished())));
    streamer.buildEvent(finishedEvent);
    assertThat(streamer.isClosed()).isTrue();

    List<BuildEvent> eventsSeen = transport.getEvents();
    assertThat(eventsSeen).hasSize(4);
    assertThat(eventsSeen.get(0).getEventId()).isEqualTo(startEvent.getEventId());
    assertThat(eventsSeen.get(1).getEventId()).isEqualTo(BuildEventIdUtil.buildFinished());
    assertThat(ImmutableSet.of(eventsSeen.get(2).getEventId(), eventsSeen.get(3).getEventId()))
        .isEqualTo(ImmutableSet.of(lateId, ProgressEvent.INITIAL_PROGRESS_UPDATE));
  }

  @Test
  public void testSuccessfulActionsAreNotPublishedByDefault() {
    EventBusHandler handler = new EventBusHandler();
    eventBus.register(handler);

    ActionExecutedEvent failedActionExecutedEvent =
        new ActionExecutedEvent(
            ActionsTestUtil.DUMMY_ARTIFACT.getExecPath(),
            new ActionsTestUtil.NullAction(),
            new ActionExecutionException("Exception", /* action= */ null, /* catastrophe= */ false),
            ActionsTestUtil.DUMMY_ARTIFACT.getPath(),
            /* stdout= */ null,
            /* stderr= */ null,
            /* actionMetadataLogs= */ ImmutableList.of(),
            ErrorTiming.BEFORE_EXECUTION,
            /* isInMemoryFs= */ false);

    streamer.buildEvent(SUCCESSFUL_ACTION_EXECUTED_EVENT);
    streamer.buildEvent(failedActionExecutedEvent);

    List<BuildEvent> transportedEvents = transport.getEvents();

    assertThat(transportedEvents).doesNotContain(SUCCESSFUL_ACTION_EXECUTED_EVENT);
    assertThat(transportedEvents).contains(failedActionExecutedEvent);
  }

  @Test
  public void testSuccessfulActionsCanBePublished() {
    EventBusHandler handler = new EventBusHandler();
    eventBus.register(handler);

    BuildEventStreamOptions options = new BuildEventStreamOptions();
    options.publishAllActions = true;

    BuildEventStreamer streamer =
        new BuildEventStreamer.Builder()
            .artifactGroupNamer(artifactGroupNamer)
            .besStreamOptions(options)
            .buildEventTransports(ImmutableSet.of(transport))
            .build();

    ActionExecutedEvent failedActionExecutedEvent =
        new ActionExecutedEvent(
            ActionsTestUtil.DUMMY_ARTIFACT.getExecPath(),
            new ActionsTestUtil.NullAction(),
            new ActionExecutionException("Exception", /* action= */ null, /* catastrophe= */ false),
            ActionsTestUtil.DUMMY_ARTIFACT.getPath(),
            /* stdout= */ null,
            /* stderr= */ null,
            /* actionMetadataLogs= */ ImmutableList.of(),
            ErrorTiming.BEFORE_EXECUTION,
            /* isInMemoryFs= */ false);

    streamer.buildEvent(SUCCESSFUL_ACTION_EXECUTED_EVENT);
    streamer.buildEvent(failedActionExecutedEvent);

    List<BuildEvent> transportedEvents = transport.getEvents();

    assertThat(transportedEvents).contains(SUCCESSFUL_ACTION_EXECUTED_EVENT);
    assertThat(transportedEvents).contains(failedActionExecutedEvent);
  }

  @Test
  public void testBuildIncomplete() {
    BuildEventId buildEventId = testId("abort_expected");
    BuildEvent startEvent =
        new GenericBuildEvent(
            BuildEventIdUtil.buildStartedId(),
            ImmutableSet.of(
                buildEventId,
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventIdUtil.buildFinished()));
    BuildCompleteEvent buildCompleteEvent =
        buildCompleteEvent(ExitCode.BUILD_FAILURE, true, null, false);

    streamer.buildEvent(startEvent);
    streamer.buildEvent(buildCompleteEvent);
    streamer.close();

    BuildEventStreamProtos.BuildEvent aborted = getBepEvent(buildEventId);
    assertThat(aborted).isNotNull();
    assertThat(aborted.hasAborted()).isNotNull();
    assertThat(aborted.getAborted().getReason()).isEqualTo(AbortReason.INCOMPLETE);
    assertThat(aborted.getAborted().getDescription()).isEmpty();
  }

  @Test
  public void testBuildCrash() {
    BuildEventId buildEventId = testId("abort_expected");
    BuildEvent startEvent =
        new GenericBuildEvent(
            BuildEventIdUtil.buildStartedId(),
            ImmutableSet.of(
                buildEventId,
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventIdUtil.buildFinished()));
    BuildCompleteEvent buildCompleteEvent =
        buildCompleteEvent(ExitCode.BUILD_FAILURE, true, new RuntimeException(), false);

    streamer.buildEvent(startEvent);
    streamer.buildEvent(buildCompleteEvent);
    streamer.close();

    BuildEventStreamProtos.BuildEvent aborted = getBepEvent(buildEventId);
    assertThat(aborted).isNotNull();
    assertThat(aborted.hasAborted()).isNotNull();
    assertThat(aborted.getAborted().getReason()).isEqualTo(AbortReason.INTERNAL);
    assertThat(aborted.getAborted().getDescription()).isEmpty();
  }

  @Test
  public void testBuildCatastrophe() {
    BuildEventId buildEventId = testId("abort_expected");
    BuildEvent startEvent =
        new GenericBuildEvent(
            BuildEventIdUtil.buildStartedId(),
            ImmutableSet.of(
                buildEventId,
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventIdUtil.buildFinished()));
    BuildCompleteEvent buildCompleteEvent =
        buildCompleteEvent(ExitCode.BUILD_FAILURE, true, null, true);

    streamer.buildEvent(startEvent);
    streamer.buildEvent(buildCompleteEvent);
    streamer.close();

    BuildEventStreamProtos.BuildEvent aborted = getBepEvent(buildEventId);
    assertThat(aborted).isNotNull();
    assertThat(aborted.hasAborted()).isNotNull();
    assertThat(aborted.getAborted().getReason()).isEqualTo(AbortReason.INTERNAL);
    assertThat(aborted.getAborted().getDescription()).isEmpty();
  }

  @Test
  public void testStreamAbortedWithTimeout() {
    BuildEventId buildEventId = testId("abort_expected");
    BuildEvent startEvent =
        new GenericBuildEvent(
            BuildEventIdUtil.buildStartedId(),
            ImmutableSet.of(
                buildEventId,
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventIdUtil.buildFinished()));

    streamer.buildEvent(startEvent);
    streamer.close(AbortReason.TIME_OUT);

    BuildEventStreamProtos.BuildEvent aborted0 = getBepEvent(buildEventId);
    assertThat(aborted0).isNotNull();
    assertThat(aborted0.hasAborted()).isNotNull();
    assertThat(aborted0.getAborted().getReason()).isEqualTo(AbortReason.TIME_OUT);
    assertThat(aborted0.getAborted().getDescription()).isEmpty();

    BuildEventStreamProtos.BuildEvent aborted1 = getBepEvent(BuildEventIdUtil.buildFinished());
    assertThat(aborted1).isNotNull();
    assertThat(aborted1.hasAborted()).isNotNull();
    assertThat(aborted1.getAborted().getReason()).isEqualTo(AbortReason.TIME_OUT);
    assertThat(aborted1.getAborted().getDescription()).isEmpty();
  }

  @Test
  public void testBuildFailureMultipleReasons() {
    BuildEventId buildEventId = testId("abort_expected");
    BuildEvent startEvent =
        new GenericBuildEvent(
            BuildEventIdUtil.buildStartedId(),
            ImmutableSet.of(
                buildEventId,
                ProgressEvent.INITIAL_PROGRESS_UPDATE,
                BuildEventIdUtil.buildFinished()));
    BuildCompleteEvent buildCompleteEvent =
        buildCompleteEvent(ExitCode.BUILD_FAILURE, false, new RuntimeException(), false);

    streamer.buildEvent(startEvent);
    streamer.noAnalyze(new NoAnalyzeEvent());
    streamer.buildEvent(buildCompleteEvent);
    streamer.close();

    BuildEventStreamProtos.BuildEvent aborted = getBepEvent(buildEventId);
    assertThat(aborted).isNotNull();
    assertThat(aborted.hasAborted()).isNotNull();
    assertThat(aborted.getAborted().getReason()).isEqualTo(AbortReason.INTERNAL);
    assertThat(aborted.getAborted().getDescription())
        .isEqualTo("Multiple abort reasons reported: [NO_ANALYZE, INTERNAL]");
  }

  @Nullable
  private BuildEventStreamProtos.BuildEvent getBepEvent(BuildEventId buildEventId) {
    return transport.getEventProtos().stream()
        .filter(e -> e.getId().equals(buildEventId))
        .findFirst()
        .orElse(null);
  }

  private BuildCompleteEvent buildCompleteEvent(
      ExitCode exitCode, boolean stopOnFailure, Throwable crash, boolean catastrophe) {
    BuildResult result = new BuildResult(0);
    result.setDetailedExitCode(DetailedExitCode.justExitCode(exitCode));
    result.setStopOnFirstFailure(stopOnFailure);
    if (catastrophe) {
      result.setCatastrophe();
    }
    if (crash != null) {
      result.setUnhandledThrowable(crash);
    }
    return new BuildCompleteEvent(result);
  }

  private static ActionExecutedEvent createActionExecutedEventWithLogs(
      ImmutableList<MetadataLog> metadataLogs) {
    return new ActionExecutedEvent(
        ActionsTestUtil.DUMMY_ARTIFACT.getExecPath(),
        new ActionsTestUtil.NullAction(),
        /* exception= */ null,
        ActionsTestUtil.DUMMY_ARTIFACT.getPath(),
        /* stdout= */ null,
        /* stderr= */ null,
        metadataLogs,
        ErrorTiming.NO_ERROR,
        /* isInMemoryFs= */ false);
  }

  @Test
  public void testActionExecutedEventLogsConstructor() {
    String metadataLogName = "action_metadata";
    Path testPath1 = FileSystems.getJavaIoFileSystem().getPath("/path/to/logs-1");
    Path testPath2 = FileSystems.getJavaIoFileSystem().getPath("/path/to/logs-2");
    MetadataLog testMetadataLog1 = new MetadataLog(metadataLogName, testPath1);
    MetadataLog testMetadataLog2 = new MetadataLog(metadataLogName, testPath2);

    ActionExecutedEvent withLogsEvent =
        createActionExecutedEventWithLogs(ImmutableList.of(testMetadataLog1, testMetadataLog2));
    ActionExecutedEvent withNoLogsEvent = SUCCESSFUL_ACTION_EXECUTED_EVENT;

    assertWithMessage("List parameter should return list of log path values")
        .that(withLogsEvent.getActionMetadataLogs())
        .containsExactly(testMetadataLog1, testMetadataLog2);
    assertWithMessage("Null logs parameter should return empty list.")
        .that(withNoLogsEvent.getActionMetadataLogs())
        .isEmpty();
  }

  @Test
  public void testActionExcutedEventProtoLogs() {
    String metadataLogName = "action_metadata";
    Path testPath1 = FileSystems.getJavaIoFileSystem().getPath("/path/to/logs-1");
    Path testPath2 = FileSystems.getJavaIoFileSystem().getPath("/path/to/logs-2");

    ActionExecutedEvent withLogsEvent =
        createActionExecutedEventWithLogs(
            ImmutableList.of(
                new MetadataLog(metadataLogName, testPath1),
                new MetadataLog(metadataLogName, testPath2)));
    ActionExecutedEvent withNoLogsEvents = SUCCESSFUL_ACTION_EXECUTED_EVENT;

    BuildEventStreamProtos.BuildEvent buildEventLogs =
        withLogsEvent.asStreamProto(getTestBuildEventContext(artifactGroupNamer));
    BuildEventStreamProtos.BuildEvent buildEventNoLogs =
        withNoLogsEvents.asStreamProto(getTestBuildEventContext(artifactGroupNamer));

    assertWithMessage("With logs build event action should contain 2 log files")
        .that(buildEventLogs.getAction().getActionMetadataLogsCount())
        .isEqualTo(2);
    assertWithMessage("No logs build event action should contain 0 log files")
        .that(buildEventNoLogs.getAction().getActionMetadataLogsCount())
        .isEqualTo(0);
    assertWithMessage("Event action should contains the two paths")
        .that(
            buildEventLogs.getAction().getActionMetadataLogsList().stream()
                .map(File::getUri)
                .collect(ImmutableList.toImmutableList()))
        .containsExactly(testPath1.toString(), testPath2.toString());
  }
}
