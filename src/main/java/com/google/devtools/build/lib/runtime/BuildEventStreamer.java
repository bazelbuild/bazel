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

import static com.google.devtools.build.lib.events.Event.of;
import static com.google.devtools.build.lib.events.EventKind.PROGRESS;
import static com.google.devtools.build.lib.util.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EventReportingArtifacts;
import com.google.devtools.build.lib.analysis.BuildInfoEvent;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.rules.extra.ExtraAction;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

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
  private Set<BuildEventId> announcedEvents;
  private final Set<BuildEventId> postedEvents = new HashSet<>();
  private final Set<BuildEventId> configurationsPosted = new HashSet<>();
  private final Multimap<BuildEventId, BuildEvent> pendingEvents = HashMultimap.create();
  private int progressCount;
  private final CountingArtifactGroupNamer artifactGroupNamer = new CountingArtifactGroupNamer();
  private OutErrProvider outErrProvider;
  private AbortReason abortReason = AbortReason.UNKNOWN;
  private static final Logger log = Logger.getLogger(BuildEventStreamer.class.getName());

  interface OutErrProvider {
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

  private static class CountingArtifactGroupNamer implements ArtifactGroupNamer {
    private final Map<Object, Long> reportedArtifactNames = new HashMap<>();
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

  public BuildEventStreamer(Collection<BuildEventTransport> transports, Reporter reporter) {
    checkArgument(transports.size() > 0);
    this.transports = transports;
    this.reporter = reporter;
    this.announcedEvents = null;
    this.progressCount = 0;
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
    BuildEvent linkEvent = null;
    BuildEventId id = event.getEventId();

    synchronized (this) {
      if (announcedEvents == null) {
        announcedEvents = new HashSet<>();
        if (!event.getChildrenEvents().contains(ProgressEvent.INITIAL_PROGRESS_UPDATE)) {
          linkEvent = ProgressEvent.progressChainIn(progressCount, event.getEventId());
          progressCount++;
          announcedEvents.addAll(linkEvent.getChildrenEvents());
          postedEvents.add(linkEvent.getEventId());
        }

        if (reporter != null) {
          reporter.post(new AnnounceBuildEventTransportsEvent(transports));
        }
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
    }

    for (BuildEventTransport transport : transports) {
      if (linkEvent != null) {
        transport.sendBuildEvent(linkEvent, artifactGroupNamer);
      }
      transport.sendBuildEvent(event, artifactGroupNamer);
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
  private void clearAnnouncedEvents() {
    if (announcedEvents != null) {
      // create a copy of the identifiers to clear, as the post method
      // will change the set of already announced events.
      Set<BuildEventId> ids;
      synchronized (this) {
        ids = Sets.difference(announcedEvents, postedEvents);
      }
      for (BuildEventId id : ids) {
        post(new AbortedEvent(id, abortReason, ""));
      }
    }
  }

  private ScheduledFuture<?> bepUploadWaitEvent(ScheduledExecutorService executor) {
    final long startNanos = System.nanoTime();
    return executor.scheduleAtFixedRate(new Runnable() {
      @Override
      public void run() {
        long deltaNanos = System.nanoTime() - startNanos;
        long deltaSeconds = TimeUnit.NANOSECONDS.toSeconds(deltaNanos);
        Event waitEvt =
            of(PROGRESS, null, "Waiting for build event protocol upload: " + deltaSeconds + "s");
        if (reporter != null) {
          reporter.handle(waitEvt);
        }
      }
    }, 0, 1, TimeUnit.SECONDS);
  }

  private void close() {
    ScheduledExecutorService executor = null;
    try {
      executor = Executors.newSingleThreadScheduledExecutor();
      List<ListenableFuture<Void>> closeFutures = new ArrayList<>(transports.size());
      for (final BuildEventTransport transport : transports) {
        ListenableFuture<Void> closeFuture = transport.close();
        closeFuture.addListener(new Runnable() {
          @Override
          public void run() {
            if (reporter != null) {
              reporter.post(new BuildEventTransportClosedEvent(transport));
            }
          }
        }, executor);
        closeFutures.add(closeFuture);
      }

      try {
        if (closeFutures.isEmpty()) {
          // Don't spam events if there is nothing to close.
          return;
        }

        ScheduledFuture<?> f = bepUploadWaitEvent(executor);
        // Wait for all transports to close.
        Futures.allAsList(closeFutures).get();
        f.cancel(true);
      } catch (Exception e) {
        log.severe("Failed to close a build event transport: " + e);
      }
    } finally {
      if (executor != null) {
        executor.shutdown();
      }
    }
  }

  private void maybeReportArtifactSet(NestedSetView<Artifact> view) {
    String name = artifactGroupNamer.maybeName(view);
    if (name == null) {
      return;
    }
    for (NestedSetView<Artifact> transitive : view.transitives()) {
      maybeReportArtifactSet(transitive);
    }
    post(new NamedArtifactGroup(name, view));
  }

  private void maybeReportArtifactSet(NestedSet<Artifact> set) {
    maybeReportArtifactSet(new NestedSetView<Artifact>(set));
  }

  private void maybeReportConfiguration(BuildConfiguration configuration) {
    BuildEventId id = configuration.getEventId();
    synchronized (this) {
      if (configurationsPosted.contains(id)) {
        return;
      }
      configurationsPosted.add(id);
    }
    post(configuration);
  }

  @Override
  public void handle(Event event) {}

  @Subscribe
  public void noBuild(NoBuildEvent event) {
    close();
  }

  @Subscribe
  public void buildInterrupted(BuildInterruptedEvent event) {
    abortReason = AbortReason.USER_INTERRUPTED;
  }

  @Subscribe
  public void buildEvent(BuildEvent event) {
    if (isActionWithoutError(event) || bufferUntilPrerequisitesReceived(event)) {
      return;
    }

    if (event instanceof BuildEventWithConfiguration) {
      for (BuildConfiguration configuration :
          ((BuildEventWithConfiguration) event).getConfigurations()) {
        maybeReportConfiguration(configuration);
      }
    }

    if (event instanceof EventReportingArtifacts) {
      for (NestedSet<Artifact> artifactSet :
          ((EventReportingArtifacts) event).reportedArtifacts()) {
        maybeReportArtifactSet(artifactSet);
      }
    }

    post(event);

    // Reconsider all events blocked by the event just posted.
    Collection<BuildEvent> toReconsider = pendingEvents.removeAll(event.getEventId());
    for (BuildEvent freedEvent : toReconsider) {
      buildEvent(freedEvent);
    }

    if (event instanceof BuildCompleteEvent) {
      buildComplete();
    }
  }

  @VisibleForTesting
  ImmutableSet<BuildEventTransport> getTransports() {
    return ImmutableSet.copyOf(transports);
  }

  private void buildComplete() {
    clearPendingEvents();
    String out = null;
    String err = null;
    if (outErrProvider != null) {
      out = outErrProvider.getOut();
      err = outErrProvider.getErr();
    }
    post(ProgressEvent.finalProgressUpdate(progressCount, out, err));
    clearAnnouncedEvents();
    close();
  }

  /**
   * Return true, if the action is not worth being reported. This is the case, if the action
   * executed successfully and is not an ExtraAction.
   */
  private static boolean isActionWithoutError(BuildEvent event) {
    return event instanceof ActionExecutedEvent
        && ((ActionExecutedEvent) event).getException() == null
        && (!(((ActionExecutedEvent) event).getAction() instanceof ExtraAction));
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
}
