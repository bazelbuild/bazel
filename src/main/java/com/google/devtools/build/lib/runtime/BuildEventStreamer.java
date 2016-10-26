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

import com.google.common.collect.Sets;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionExecutedEvent;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.buildeventstream.AbortedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Logger;

/** Listen for {@link BuildEvent} and stream them to the provided {@link BuildEventTransport}. */
public class BuildEventStreamer implements EventHandler {
  private final Collection<BuildEventTransport> transports;
  private Set<BuildEventId> announcedEvents;
  private Set<BuildEventId> postedEvents;
  private int progressCount;
  private AbortReason abortReason = AbortReason.UNKNOWN;
  private static final Logger LOG = Logger.getLogger(BuildEventStreamer.class.getName());

  public BuildEventStreamer(Collection<BuildEventTransport> transports) {
    this.transports = transports;
    this.announcedEvents = null;
    this.postedEvents = null;
    this.progressCount = 0;
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
        postedEvents = new HashSet<>();
        if (!event.getChildrenEvents().contains(ProgressEvent.INITIAL_PROGRESS_UPDATE)) {
          linkEvent = ProgressEvent.progressChainIn(progressCount, event.getEventId());
          progressCount++;
          announcedEvents.addAll(linkEvent.getChildrenEvents());
          postedEvents.add(linkEvent.getEventId());
        }
      } else {
        if (!announcedEvents.contains(id)) {
          linkEvent = ProgressEvent.progressChainIn(progressCount, id);
          progressCount++;
          announcedEvents.addAll(linkEvent.getChildrenEvents());
          postedEvents.add(linkEvent.getEventId());
        }
        postedEvents.add(id);
      }
      announcedEvents.addAll(event.getChildrenEvents());
    }

    for (BuildEventTransport transport : transports) {
      try {
        if (linkEvent != null) {
          transport.sendBuildEvent(linkEvent);
        }
        transport.sendBuildEvent(event);
      } catch (IOException e) {
        // TODO(aehlig): signal that the build ought to be aborted
        LOG.severe("Failed to write to build event transport: " + e);
      }
    }
  }

  /**
   * Clear all events that are still pending; events not naturally closed by the expected event
   * normally only occur if the build is aborted.
   */
  private void clearPendingEvents() {
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

  private void close() {
    for (BuildEventTransport transport : transports) {
      try {
        transport.close();
      } catch (IOException e) {
        // TODO(aehlig): signal that the build ought to be aborted
        LOG.warning("Failure while closing build event transport: " + e);
      }
    }
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
  };

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    post(ProgressEvent.finalProgressUpdate(progressCount));
    clearPendingEvents();
    close();
  }

  @Subscribe
  public void buildEvent(BuildEvent event) {
    if (event instanceof ActionExecutedEvent) {
      // We ignore events about action executions if the execution succeeded.
      if (((ActionExecutedEvent) event).getException() == null) {
        return;
      }
    }

    post(event);
  }
}
