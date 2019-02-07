// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.nullToEmpty;
import static com.google.devtools.build.lib.events.Event.of;
import static com.google.devtools.build.lib.events.EventKind.PROGRESS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.actions.EventReportingArtifacts.ReportedArtifacts;
import com.google.devtools.build.lib.analysis.BuildInfoEvent;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.extra.ExtraAction;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildCompletingEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.ChainableEvent;
import com.google.devtools.build.lib.buildeventstream.LastBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.NoAnalyzeEvent;
import com.google.devtools.build.lib.buildtool.buildevent.NoExecutionEvent;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.pkgcache.TargetParsingCompleteEvent;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Listens for {@link BuildEvent}s and streams them to the provided {@link BuildEventTransport}s.
 *
 * <p>The streamer takes care of closing all {@link BuildEventTransport}s. It does so after having
 * received a {@link BuildCompleteEvent}. Furthermore, it emits two event types to the
 * {@code eventBus}. After having received the first {@link BuildEvent} it emits a
 * {@link AnnounceBuildEventTransportsEvent} that contains a list of all its transports.
 * Furthermore, after a transport has been closed, it emits
 * a {@link BuildEventTransportClosedEvent}.
 */
public class BuildEventStreamer implements EventHandler {

  private final Collection<BuildEventTransport> transports;
  private final Reporter reporter;
  private final BuildEventStreamOptions options;
  private Set<BuildEventId> announcedEvents;
  private final Set<BuildEventId> postedEvents = new HashSet<>();
  private final Set<BuildEventId> configurationsPosted = new HashSet<>();
  private List<Pair<String, String>> bufferedStdoutStderrPairs = new ArrayList<>();
  private final Multimap<BuildEventId, BuildEvent> pendingEvents = HashMultimap.create();
  private int progressCount;
  private final CountingArtifactGroupNamer artifactGroupNamer = new CountingArtifactGroupNamer();
  private OutErrProvider outErrProvider;
  private volatile AbortReason abortReason = AbortReason.UNKNOWN;
  // Will be set to true if the build was invoked through "bazel test" or "bazel coverage".
  private boolean isTestCommand;

  // After #buildComplete is called, contains the set of events that the streamer is expected to
  // process. The streamer will fully close after seeing them. This field is null until
  // #buildComplete is called.
  private Set<BuildEventId> finalEventsToCome = null;

  // True, if we already closed the stream.
  private boolean closed;

  private static final Logger logger = Logger.getLogger(BuildEventStreamer.class.getName());

  /**
   * Provider for stdout and stderr output.
   */
  public interface OutErrProvider {
    /**
     * Return the chunks of stdout that were produced since the last call to this function (or the
     * beginning of the build, for the first call). It is the responsibility of the class
     * implementing this interface to properly synchronize with simultaneously written output.
     */
    Iterable<String> getOut();

    /**
     * Return the chunks of stderr that were produced since the last call to this function (or the
     * beginning of the build, for the first call). It is the responsibility of the class
     * implementing this interface to properly synchronize with simultaneously written output.
     */
    Iterable<String> getErr();
  }

  @ThreadSafe
  private static class CountingArtifactGroupNamer implements ArtifactGroupNamer {
    @GuardedBy("this")
    private final Map<Object, Long> reportedArtifactNames = new HashMap<>();

    @GuardedBy("this")
    private long nextArtifactName;

    @Override
    public NamedSetOfFilesId apply(Object id) {
      Long name;
      synchronized (this) {
        name = reportedArtifactNames.get(id);
      }
      if (name == null) {
        return null;
      }
      return NamedSetOfFilesId.newBuilder().setId(name.toString()).build();
    }

    /**
     * If the {@link NestedSetView} has no name already, return a new name for it. Return null
     * otherwise.
     */
    synchronized String maybeName(NestedSetView<Artifact> view) {
      if (reportedArtifactNames.containsKey(view.identifier())) {
        return null;
      }
      Long name = nextArtifactName;
      nextArtifactName++;
      reportedArtifactNames.put(view.identifier(), name);
      return name.toString();
    }
  }

  /** Creates a new build event streamer. */
  public BuildEventStreamer(
      Collection<BuildEventTransport> transports,
      @Nullable Reporter reporter,
      BuildEventStreamOptions options) {
    checkArgument(transports.size() > 0);
    checkNotNull(options);
    this.transports = transports;
    this.reporter = reporter;
    this.options = options;
    this.announcedEvents = null;
    this.progressCount = 0;
  }

  /** Creates a new build event streamer with default options. */
  public BuildEventStreamer(
      Collection<BuildEventTransport> transports, @Nullable Reporter reporter) {
    this(transports, reporter, new BuildEventStreamOptions());
  }

  public void registerOutErrProvider(OutErrProvider outErrProvider) {
    this.outErrProvider = outErrProvider;
  }

  /**
   * Post a new event to all transports; simultaneously keep track of the events we announce to
   * still come.
   *
   * <p>Moreover, link unannounced events to the progress stream; we only expect failure events to
   * come before their parents.
   */
  private void post(BuildEvent event) {
    List<BuildEvent> linkEvents = null;
    BuildEventId id = event.getEventId();
    List<BuildEvent> flushEvents = null;
    boolean lastEvent = false;

    synchronized (this) {
      if (announcedEvents == null) {
        announcedEvents = new HashSet<>();
        // The very first event of a stream is implicitly announced by the convention that
        // a complete stream has to have at least one entry. In this way we keep the invariant
        // that the set of posted events is always a subset of the set of announced events.
        announcedEvents.add(id);
        if (!event.getChildrenEvents().contains(ProgressEvent.INITIAL_PROGRESS_UPDATE)) {
          BuildEvent progress = ProgressEvent.progressChainIn(progressCount, event.getEventId());
          linkEvents = ImmutableList.of(progress);
          progressCount++;
          announcedEvents.addAll(progress.getChildrenEvents());
          // the new first event in the stream, implicitly announced by the fact that complete
          // stream may not be empty.
          announcedEvents.add(progress.getEventId());
          postedEvents.add(progress.getEventId());
        }

        if (reporter != null) {
          reporter.post(new AnnounceBuildEventTransportsEvent(transports));
        }

        if (!bufferedStdoutStderrPairs.isEmpty()) {
          flushEvents = new ArrayList<>(bufferedStdoutStderrPairs.size());
          for (Pair<String, String> outErrPair : bufferedStdoutStderrPairs) {
            flushEvents.add(flushStdoutStderrEvent(outErrPair.getFirst(), outErrPair.getSecond()));
          }
        }
        bufferedStdoutStderrPairs = null;
      } else {
        if (!announcedEvents.contains(id)) {
          Iterable<String> allOut = ImmutableList.of();
          Iterable<String> allErr = ImmutableList.of();
          if (outErrProvider != null) {
            allOut = orEmpty(outErrProvider.getOut());
            allErr = orEmpty(outErrProvider.getErr());
          }
          linkEvents = new ArrayList<>();
          List<BuildEvent> finalLinkEvents = linkEvents;
          consumeAsPairsofStrings(
              allOut,
              allErr,
              (out, err) -> {
                BuildEvent progressEvent =
                    ProgressEvent.progressChainIn(progressCount, id, out, err);
                finalLinkEvents.add(progressEvent);
                progressCount++;
                announcedEvents.addAll(progressEvent.getChildrenEvents());
                postedEvents.add(progressEvent.getEventId());
              });
        }
      }

      if (event instanceof BuildInfoEvent) {
        // The specification for BuildInfoEvent says that there may be many such events,
        // but all except the first one should be ignored.
        if (postedEvents.contains(id)) {
          return;
        }
      }

      postedEvents.add(id);
      announcedEvents.addAll(event.getChildrenEvents());
      // We keep as an invariant that postedEvents is a subset of announced events, so this is a
      // cheaper test for equality
      if (announcedEvents.size() == postedEvents.size()) {
        lastEvent = true;
      }
    }

    BuildEvent mainEvent = event;
    if (lastEvent) {
      mainEvent = new LastBuildEvent(event);
    }

    for (BuildEventTransport transport : transports) {
      if (linkEvents != null) {
        for (BuildEvent linkEvent : linkEvents) {
          transport.sendBuildEvent(linkEvent, artifactGroupNamer);
        }
      }
      transport.sendBuildEvent(mainEvent, artifactGroupNamer);
    }

    if (flushEvents != null) {
      for (BuildEvent flushEvent : flushEvents) {
        for (BuildEventTransport transport : transports) {
          transport.sendBuildEvent(flushEvent, artifactGroupNamer);
        }
      }
    }
  }

  /**
   * If some events are blocked on the absence of a build_started event, generate such an event;
   * moreover, make that artificial start event announce all events blocked on it, as well as the
   * {@link BuildCompletingEvent} that caused the early end of the stream.
   */
  private void clearMissingStartEvent(BuildEventId id) {
    if (pendingEvents.containsKey(BuildEventId.buildStartedId())) {
      ImmutableSet.Builder<BuildEventId> children = ImmutableSet.<BuildEventId>builder();
      children.add(ProgressEvent.INITIAL_PROGRESS_UPDATE);
      children.add(id);
      children.addAll(
          pendingEvents
              .get(BuildEventId.buildStartedId())
              .stream()
              .map(BuildEvent::getEventId)
              .collect(ImmutableSet.<BuildEventId>toImmutableSet()));
      buildEvent(
          new AbortedEvent(BuildEventId.buildStartedId(), children.build(), abortReason, ""));
    }
  }

  /** Clear pending events by generating aborted events for all their requisits. */
  private void clearPendingEvents() {
    while (!pendingEvents.isEmpty()) {
      BuildEventId id = pendingEvents.keySet().iterator().next();
      buildEvent(new AbortedEvent(id, abortReason, ""));
    }
  }

  /**
   * Clear all events that are still announced; events not naturally closed by the expected event
   * normally only occur if the build is aborted.
   */
  private void clearAnnouncedEvents(Collection<BuildEventId> dontclear) {
    if (announcedEvents != null) {
      // create a copy of the identifiers to clear, as the post method
      // will change the set of already announced events.
      Set<BuildEventId> ids;
      synchronized (this) {
        ids = Sets.difference(announcedEvents, postedEvents);
      }
      for (BuildEventId id : ids) {
        if (!dontclear.contains(id)) {
          post(new AbortedEvent(id, abortReason, ""));
        }
      }
    }
  }

  private ScheduledFuture<?> bepUploadWaitEvent(ScheduledExecutorService executor) {
    final long startNanos = System.nanoTime();
    return executor.scheduleAtFixedRate(
        () -> {
          long deltaNanos = System.nanoTime() - startNanos;
          long deltaSeconds = TimeUnit.NANOSECONDS.toSeconds(deltaNanos);
          Event waitEvt =
              of(PROGRESS, null, "Waiting for Build Event Protocol upload: " + deltaSeconds + "s");
          if (reporter != null) {
            reporter.handle(waitEvt);
          }
        },
        0,
        1,
        TimeUnit.SECONDS);
  }

  public synchronized boolean isClosed() {
    return closed;
  }

  public void close() {
    close(null);
  }

  public void close(@Nullable AbortReason reason) {
    synchronized (this) {
      if (closed) {
        return;
      }
      closed = true;
      if (reason != null) {
        abortReason = reason;
      }

      if (finalEventsToCome == null) {
        // This should only happen if there's a crash. Try to clean up as best we can.
        clearEventsAndPostFinalProgress(null);
      }
    }


    ScheduledExecutorService executor = null;
    try {
      executor = Executors.newSingleThreadScheduledExecutor(
          new ThreadFactoryBuilder().setNameFormat("build-event-streamer-%d").build());
      List<ListenableFuture<Void>> closeFutures = new ArrayList<>(transports.size());
      for (final BuildEventTransport transport : transports) {
        ListenableFuture<Void> closeFuture = transport.close();
        closeFuture.addListener(
            () -> {
              if (reporter != null) {
                reporter.post(new BuildEventTransportClosedEvent(transport));
              }
            },
            executor);
        closeFutures.add(closeFuture);
      }

      try {
        if (closeFutures.isEmpty()) {
          // Don't spam events if there is nothing to close.
          return;
        }

        ScheduledFuture<?> f = bepUploadWaitEvent(executor);
        // Wait for all transports to close, ignoring interrupts.
        Uninterruptibles.getUninterruptibly(Futures.allAsList(closeFutures));
        f.cancel(true);
      } catch (ExecutionException e) {
        logger.log(Level.SEVERE, "Failed to close a build event transport", e);
        LoggingUtil.logToRemote(Level.SEVERE, "Failed to close a build event transport", e);
      }
    } finally {
      if (executor != null) {
        executor.shutdown();
      }
    }
  }

  private void maybeReportArtifactSet(
      ArtifactPathResolver pathResolver, NestedSetView<Artifact> view) {
    String name = artifactGroupNamer.maybeName(view);
    if (name == null) {
      return;
    }
    // We only split if the max number of entries is at least 2 (it must be at least a binary tree).
    // The method throws for smaller values.
    if (options.maxNamedSetEntries >= 2) {
      // We only split the event after naming it to avoid splitting the same node multiple times.
      // Note that the artifactGroupNames keeps references to the individual pieces, so this can
      // double the memory consumption of large nested sets.
      view = view.splitIfExceedsMaximumSize(options.maxNamedSetEntries);
    }
    for (NestedSetView<Artifact> transitive : view.transitives()) {
      maybeReportArtifactSet(pathResolver, transitive);
    }
    post(new NamedArtifactGroup(name, pathResolver, view));
  }

  private void maybeReportArtifactSet(ArtifactPathResolver pathResolver, NestedSet<Artifact> set) {
    maybeReportArtifactSet(pathResolver, new NestedSetView<Artifact>(set));
  }

  private void maybeReportConfiguration(BuildEvent configuration) {
    BuildEvent event = configuration;
    if (configuration == null) {
      event = new NullConfiguration();
    }
    BuildEventId id = event.getEventId();
    synchronized (this) {
      if (configurationsPosted.contains(id)) {
        return;
      }
      configurationsPosted.add(id);
    }
    post(event);
  }

  @Override
  public void handle(Event event) {}

  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    abortReason = AbortReason.USER_INTERRUPTED;
  }

  @Subscribe
  public void noAnalyze(NoAnalyzeEvent event) {
    abortReason = AbortReason.NO_ANALYZE;
  }

  @Subscribe
  public void noExecution(NoExecutionEvent event) {
    abortReason = AbortReason.NO_BUILD;
  }

  @Subscribe
  public void buildEvent(BuildEvent event) {
    if (finalEventsToCome != null) {
      synchronized (this) {
        BuildEventId id = event.getEventId();
        if (finalEventsToCome.contains(id)) {
          finalEventsToCome.remove(id);
        } else {
          return;
        }
      }
    }

    if (shouldIgnoreBuildEvent(event)) {
      return;
    }

    if (event instanceof BuildStartingEvent) {
      BuildRequest buildRequest = ((BuildStartingEvent) event).getRequest();
      isTestCommand = "test".equals(buildRequest.getCommandName())
          || "coverage".equals(buildRequest.getCommandName());
    }

    if (event instanceof BuildEventWithConfiguration) {
      for (BuildEvent configuration : ((BuildEventWithConfiguration) event).getConfigurations()) {
        maybeReportConfiguration(configuration);
      }
    }

    if (event instanceof EventReportingArtifacts) {
      ReportedArtifacts reportedArtifacts = ((EventReportingArtifacts) event).reportedArtifacts();
      for (NestedSet<Artifact> artifactSet : reportedArtifacts.artifacts) {
        maybeReportArtifactSet(reportedArtifacts.pathResolver, artifactSet);
      }
    }

    if (event instanceof BuildCompletingEvent
        && !event.getEventId().equals(BuildEventId.buildStartedId())) {
      clearMissingStartEvent(event.getEventId());
    }

    post(event);

    // Reconsider all events blocked by the event just posted.
    Collection<BuildEvent> toReconsider = pendingEvents.removeAll(event.getEventId());
    for (BuildEvent freedEvent : toReconsider) {
      buildEvent(freedEvent);
    }

    if (event instanceof BuildCompleteEvent && isCrash((BuildCompleteEvent) event)) {
      abortReason = AbortReason.INTERNAL;
    }

    if (event instanceof BuildCompletingEvent) {
      buildComplete(event);
    }

    if (event instanceof NoBuildEvent) {
      if (!((NoBuildEvent) event).separateFinishedEvent()) {
        buildComplete(event);
      }
    }

    if (finalEventsToCome != null && finalEventsToCome.isEmpty()) {
      close();
    }
  }

  private static boolean isCrash(BuildCompleteEvent event) {
    return event.getResult().getUnhandledThrowable() != null;
  }

  private synchronized BuildEvent flushStdoutStderrEvent(String out, String err) {
    BuildEvent updateEvent = ProgressEvent.progressUpdate(progressCount, out, err);
    progressCount++;
    announcedEvents.addAll(updateEvent.getChildrenEvents());
    postedEvents.add(updateEvent.getEventId());
    return updateEvent;
  }

  void flush() {
    List<BuildEvent> updateEvents = null;
    synchronized (this) {
      Iterable<String> allOut = ImmutableList.of();
      Iterable<String> allErr = ImmutableList.of();
      if (outErrProvider != null) {
        allOut = orEmpty(outErrProvider.getOut());
        allErr = orEmpty(outErrProvider.getErr());
      }
      if (Iterables.isEmpty(allOut) && Iterables.isEmpty(allErr)) {
        // Nothing to flush; avoid generating an unneeded progress event.
        return;
      }
      if (announcedEvents != null) {
        updateEvents = new ArrayList<>();
        List<BuildEvent> finalUpdateEvents = updateEvents;
        consumeAsPairsofStrings(
            allOut, allErr, (s1, s2) -> finalUpdateEvents.add(flushStdoutStderrEvent(s1, s2)));
      } else {
        consumeAsPairsofStrings(
            allOut, allErr, (s1, s2) -> bufferedStdoutStderrPairs.add(Pair.of(s1, s2)));
      }
    }
    if (updateEvents != null) {
      for (BuildEvent updateEvent : updateEvents) {
        for (BuildEventTransport transport : transports) {
          transport.sendBuildEvent(updateEvent, artifactGroupNamer);
        }
      }
    }
  }

  // Returns the given Iterable, or an empty list if null.
  private static <T> Iterable<T> orEmpty(Iterable<T> original) {
    return original == null ? ImmutableList.of() : original;
  }

  // Given a pair of iterables and {@link BiConsumer}s, emit a sequence of pairs to the consumers.
  // Given the leftIterables [L1, L2, ... LN], and the rightIterable [R1, R2, ... RM], the consumers
  // will see this sequence of calls:
  //  biConsumer.accept(L1, null);
  //  biConsumer.accept(L2, null);
  //  ....
  //  biConsumer.accept(L(N-1), null);
  //  biConsumer.accept(LN, R1);
  //  biConsumer.accept(R2, null);
  //  ...
  //  biConsumer.accept(R(M-1), null);
  //  lastConsumer.accept(RM, null);
  //
  // The lastConsumer is always called exactly once, even if both Iterables are empty.
  @VisibleForTesting
  static <T> void consumeAsPairs(
      Iterable<T> leftIterable,
      Iterable<T> rightIterable,
      BiConsumer<T, T> biConsumer,
      BiConsumer<T, T> lastConsumer) {
    if (Iterables.isEmpty(leftIterable) && Iterables.isEmpty(rightIterable)) {
      lastConsumer.accept(null, null);
      return;
    }

    Iterator<T> leftIterator = leftIterable.iterator();
    Iterator<T> rightIterator = rightIterable.iterator();
    while (leftIterator.hasNext()) {
      T left = leftIterator.next();
      boolean lastT = !leftIterator.hasNext();
      T right = (lastT && rightIterator.hasNext()) ? rightIterator.next() : null;
      boolean lastItem = lastT && !rightIterator.hasNext();
      (lastItem ? lastConsumer : biConsumer).accept(left, right);
    }

    while (rightIterator.hasNext()) {
      T right = rightIterator.next();
      (!rightIterator.hasNext() ? lastConsumer : biConsumer).accept(null, right);
    }
  }

  private static <T> void consumeAsPairsofStrings(
      Iterable<String> leftIterable,
      Iterable<String> rightIterable,
      BiConsumer<String, String> biConsumer,
      BiConsumer<String, String> lastConsumer) {
    consumeAsPairs(
        leftIterable,
        rightIterable,
        (s1, s2) -> biConsumer.accept(nullToEmpty(s1), nullToEmpty(s2)),
        (s1, s2) -> lastConsumer.accept(nullToEmpty(s1), nullToEmpty(s2)));
  }

  private static <T> void consumeAsPairsofStrings(
      Iterable<String> leftIterable,
      Iterable<String> rightIterable,
      BiConsumer<String, String> biConsumer) {
    consumeAsPairsofStrings(leftIterable, rightIterable, biConsumer, biConsumer);
  }

  @VisibleForTesting
  public ImmutableSet<BuildEventTransport> getTransports() {
    return ImmutableSet.copyOf(transports);
  }

  private synchronized void clearEventsAndPostFinalProgress(ChainableEvent event) {
    clearPendingEvents();
    Iterable<String> allOut = ImmutableList.of();
    Iterable<String> allErr = ImmutableList.of();
    if (outErrProvider != null) {
      allOut = orEmpty(outErrProvider.getOut());
      allErr = orEmpty(outErrProvider.getErr());
    }
    consumeAsPairsofStrings(
        allOut,
        allErr,
        (s1, s2) -> post(flushStdoutStderrEvent(s1, s2)),
        (s1, s2) -> post(ProgressEvent.finalProgressUpdate(progressCount, s1, s2)));
    clearAnnouncedEvents(event == null ? ImmutableList.of() : event.getChildrenEvents());
  }

  private synchronized void buildComplete(ChainableEvent event) {
    clearEventsAndPostFinalProgress(event);

    finalEventsToCome = new HashSet<>(announcedEvents);
    finalEventsToCome.removeAll(postedEvents);
    if (finalEventsToCome.isEmpty()) {
      close();
    }
  }

  /** Returns whether a {@link BuildEvent} should be ignored. */
  public boolean shouldIgnoreBuildEvent(BuildEvent event) {
    if (event instanceof ActionExecutedEvent
        && !shouldPublishActionExecutedEvent((ActionExecutedEvent) event)) {
      return true;
    }

    if (bufferUntilPrerequisitesReceived(event) || isVacuousTestSummary(event)) {
      return true;
    }

    if (isTestCommand && event instanceof BuildCompleteEvent) {
      // In case of "bazel test" ignore the BuildCompleteEvent, as it will be followed by a
      // TestingCompleteEvent that contains the correct exit code.
      return !isCrash((BuildCompleteEvent) event);
    }

    if (event instanceof TargetParsingCompleteEvent) {
      // If there is only one pattern and we have one failed pattern, then we already posted a
      // pattern expanded error, so we don't post the completion event.
      // TODO(ulfjack): This is brittle. It would be better to always post one PatternExpanded event
      // for each pattern given on the command line instead of one event for all of them combined.
      return ((TargetParsingCompleteEvent) event).getOriginalTargetPattern().size() == 1
          && !((TargetParsingCompleteEvent) event).getFailedTargetPatterns().isEmpty();
    }

    return false;
  }

  /** Returns whether an {@link ActionExecutedEvent} should be published. */
  private boolean shouldPublishActionExecutedEvent(ActionExecutedEvent event) {
    if (options.publishAllActions) {
      return true;
    }
    if (event.getException() != null) {
      // Publish failed actions
      return true;
    }
    return (event.getAction() instanceof ExtraAction);
  }

  private boolean bufferUntilPrerequisitesReceived(BuildEvent event) {
    if (!(event instanceof BuildEventWithOrderConstraint)) {
      return false;
    }
    // Check if all prerequisite events are posted already.
    for (BuildEventId prerequisiteId : ((BuildEventWithOrderConstraint) event).postedAfter()) {
      if (!postedEvents.contains(prerequisiteId)) {
        pendingEvents.put(prerequisiteId, event);
        return true;
      }
    }
    return false;
  }

  /** Return true if the test summary contains no actual test runs. */
  private boolean isVacuousTestSummary(BuildEvent event) {
    return event instanceof TestSummary && (((TestSummary) event).totalRuns() == 0);
  }
}
