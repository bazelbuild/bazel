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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
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
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
  private AbortReason abortReason = AbortReason.UNKNOWN;
  // Will be set to true if the build was invoked through "bazel test" or "bazel coverage".
  private boolean isTestCommand;

  // After a BuildCompetingEvent we might expect a whitelisted set of events. If non-null,
  // the streamer is restricted to only allow those events and fully close after having seen
  // them.
  private Set<BuildEventId> finalEventsToCome = null;

  // True, if we already closed the stream.
  private boolean closed;

  /**
   * Provider for stdout and stderr output.
   */
  public interface OutErrProvider {
    /**
     * Return the chunk of stdout that was produced since the last call to this function (or the
     * beginning of the build, for the first call). It is the responsibility of the class
     * implementing this interface to properly synchronize with simultaneously written output.
     */
    String getOut();

    /**
     * Return the chunk of stderr that was produced since the last call to this function (or the
     * beginning of the build, for the first call). It is the responsibility of the class
     * implementing this interface to properly synchronize with simultaneously written output.
     */
    String getErr();
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
    Preconditions.checkState(!isClosed(),
        String.format("Received event of type '%s' after close. This is a bug.",
            event.getClass().getName()));
    BuildEvent linkEvent = null;
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
          linkEvent = ProgressEvent.progressChainIn(progressCount, event.getEventId());
          progressCount++;
          announcedEvents.addAll(linkEvent.getChildrenEvents());
          // the new first event in the stream, implicitly announced by the fact that complete
          // stream may not be empty.
          announcedEvents.add(linkEvent.getEventId());
          postedEvents.add(linkEvent.getEventId());
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
          String out = null;
          String err = null;
          if (outErrProvider != null) {
            out = outErrProvider.getOut();
            err = outErrProvider.getErr();
          }
          linkEvent = ProgressEvent.progressChainIn(progressCount, id, out, err);
          progressCount++;
          announcedEvents.addAll(linkEvent.getChildrenEvents());
          postedEvents.add(linkEvent.getEventId());
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
      if (linkEvent != null) {
        transport.sendBuildEvent(linkEvent, artifactGroupNamer);
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

  @VisibleForTesting
  boolean isClosed() {
    return closed;
  }

  /**
   * Close marks the streamer as closed but does not release any resources. We only set
   * {@code closed} to {@code true} so that we can call {@link #isClosed()} in tests.
   */
  private void close() {
    closed = true;
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

  private synchronized BuildEvent flushStdoutStderrEvent(String out, String err) {
    BuildEvent updateEvent = ProgressEvent.progressUpdate(progressCount, out, err);
    progressCount++;
    announcedEvents.addAll(updateEvent.getChildrenEvents());
    postedEvents.add(updateEvent.getEventId());
    return updateEvent;
  }

  void flush() {
    BuildEvent updateEvent = null;
    synchronized (this) {
      String out = null;
      String err = null;
      if (outErrProvider != null) {
        out = outErrProvider.getOut();
        err = outErrProvider.getErr();
      }
      if (Strings.isNullOrEmpty(out) && Strings.isNullOrEmpty(err)) {
        // Nothing to flush; avoid generating an unneeded progress event.
        return;
      }
      if (announcedEvents != null) {
        updateEvent = flushStdoutStderrEvent(out, err);
      } else {
        bufferedStdoutStderrPairs.add(Pair.of(out, err));
      }
    }
    if (updateEvent != null) {
      for (BuildEventTransport transport : transports) {
        transport.sendBuildEvent(updateEvent, artifactGroupNamer);
      }
    }
  }

  @VisibleForTesting
  public ImmutableSet<BuildEventTransport> getTransports() {
    return ImmutableSet.copyOf(transports);
  }

  private void buildComplete(ChainableEvent event) {
    clearPendingEvents();
    String out = null;
    String err = null;
    if (outErrProvider != null) {
      out = outErrProvider.getOut();
      err = outErrProvider.getErr();
    }
    post(ProgressEvent.finalProgressUpdate(progressCount, out, err));
    clearAnnouncedEvents(event.getChildrenEvents());

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
      return true;
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
    if (event.getAction() instanceof ExtraAction) {
      return true;
    }
    return false;
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
