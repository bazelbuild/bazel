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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.concurrent.TimeUnit.MINUTES;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import javax.management.ListenerNotFoundException;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

/**
 * Monitors the size of the retained heap and exit promptly if it grows too large.
 *
 * <p>Specifically, checks the size of the tenured space after each major GC; if it exceeds {@link
 * #occupiedHeapPercentageThreshold}%, call {@link System#gc()} to trigger a stop-the-world
 * collection; if it's still more than {@link #occupiedHeapPercentageThreshold}% full, exit with an
 * {@link OutOfMemoryError}.
 */
final class RetainedHeapLimiter implements NotificationListener {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static final long MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS = 60000;

  private final AtomicBoolean throwingOom = new AtomicBoolean(false);
  private final ImmutableList<NotificationEmitter> tenuredGcEmitters;
  private OptionalInt occupiedHeapPercentageThreshold = OptionalInt.empty();
  private String oomMessage = "";
  private EventHandler eventHandler = NullEventHandler.INSTANCE;
  private final AtomicLong lastTriggeredGcInMilliseconds = new AtomicLong();
  private final BugReporter bugReporter;

  static RetainedHeapLimiter create(BugReporter bugReporter) {
    return createFromBeans(ManagementFactory.getGarbageCollectorMXBeans(), bugReporter);
  }

  @VisibleForTesting
  static RetainedHeapLimiter createFromBeans(
      List<GarbageCollectorMXBean> gcBeans, BugReporter bugReporter) {
    ImmutableList<NotificationEmitter> tenuredGcEmitters = findTenuredCollectorBeans(gcBeans);
    if (tenuredGcEmitters.isEmpty()) {
      logger.atSevere().log(
          "Unable to find tenured collector from %s: names were %s. "
              + "--experimental_oom_more_eagerly_threshold cannot be specified for this JVM",
          gcBeans,
          gcBeans.stream()
              .map(GarbageCollectorMXBean::getMemoryPoolNames)
              .map(Arrays::asList)
              .collect(toImmutableList()));
    }
    return new RetainedHeapLimiter(tenuredGcEmitters, bugReporter);
  }

  private RetainedHeapLimiter(
      ImmutableList<NotificationEmitter> tenuredGcEmitters, BugReporter bugReporter) {
    this.tenuredGcEmitters = checkNotNull(tenuredGcEmitters);
    this.bugReporter = checkNotNull(bugReporter);
  }

  @ThreadSafety.ThreadCompatible // Can only be called on the logical main Bazel thread.
  void update(int occupiedHeapPercentageThreshold, String oomMessage, EventHandler eventHandler)
      throws AbruptExitException {
    if (tenuredGcEmitters.isEmpty() && occupiedHeapPercentageThreshold != 100) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      "No tenured GC collectors were found: unable to watch for GC events to exit"
                          + " JVM when "
                          + occupiedHeapPercentageThreshold
                          + "% of heap is used")
                  .setMemoryOptions(
                      FailureDetails.MemoryOptions.newBuilder()
                          .setCode(
                              FailureDetails.MemoryOptions.Code
                                  .EXPERIMENTAL_OOM_MORE_EAGERLY_NO_TENURED_COLLECTORS_FOUND)
                          .build())
                  .build()));
    }
    if (occupiedHeapPercentageThreshold < 0 || occupiedHeapPercentageThreshold > 100) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      "--experimental_oom_more_eagerly_threshold must be a percent between "
                          + "0 and 100 but was "
                          + occupiedHeapPercentageThreshold)
                  .setMemoryOptions(
                      FailureDetails.MemoryOptions.newBuilder()
                          .setCode(
                              FailureDetails.MemoryOptions.Code
                                  .EXPERIMENTAL_OOM_MORE_EAGERLY_THRESHOLD_INVALID_VALUE)
                          .build())
                  .build()));
    }
    boolean alreadyInstalled = this.occupiedHeapPercentageThreshold.isPresent();
    this.occupiedHeapPercentageThreshold =
        occupiedHeapPercentageThreshold < 100
            ? OptionalInt.of(occupiedHeapPercentageThreshold)
            : OptionalInt.empty();
    boolean shouldBeInstalled = this.occupiedHeapPercentageThreshold.isPresent();
    if (alreadyInstalled && !shouldBeInstalled) {
      for (NotificationEmitter emitter : tenuredGcEmitters) {
        try {
          emitter.removeNotificationListener(this, null, null);
        } catch (ListenerNotFoundException e) {
          logger.atWarning().withCause(e).log("Couldn't remove self as listener from %s", emitter);
        }
      }
    } else if (!alreadyInstalled && shouldBeInstalled) {
      tenuredGcEmitters.forEach(e -> e.addNotificationListener(this, null, null));
    }
    this.oomMessage = oomMessage;
    this.eventHandler = eventHandler;
  }

  /**
   * Resets the event handler to {@link NullEventHandler#INSTANCE}, releasing any other event
   * handler passed to {@link #update}.
   */
  void resetEventHandler() {
    eventHandler = NullEventHandler.INSTANCE;
  }

  @VisibleForTesting
  static ImmutableList<NotificationEmitter> findTenuredCollectorBeans(
      List<GarbageCollectorMXBean> gcBeans) {
    ImmutableList.Builder<NotificationEmitter> builder = ImmutableList.builder();
    // Examine all collectors and register for notifications from those which collect the tenured
    // space. Normally there is one such collector.
    for (GarbageCollectorMXBean gcBean : gcBeans) {
      for (String name : gcBean.getMemoryPoolNames()) {
        if (isTenuredSpace(name)) {
          builder.add((NotificationEmitter) gcBean);
        }
      }
    }
    return builder.build();
  }

  // Can be called concurrently, handles concurrent calls with #update gracefully.
  @ThreadSafety.ThreadSafe
  @Override
  public void handleNotification(Notification notification, Object handback) {
    if (!notification
        .getType()
        .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
      return;
    }
    // Get a local reference to guard against concurrent modifications.
    OptionalInt occupiedHeapPercentageThreshold = this.occupiedHeapPercentageThreshold;
    if (!occupiedHeapPercentageThreshold.isPresent()) {
      // Presumably failure above to uninstall this listener, or a racy GC.
      logger.atInfo().atMostEvery(1, MINUTES).log(
          "Got notification %s when should be disabled", notification);
      return;
    }
    GarbageCollectionNotificationInfo info =
        GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());
    Map<String, MemoryUsage> spaces = info.getGcInfo().getMemoryUsageAfterGc();
    for (Map.Entry<String, MemoryUsage> entry : spaces.entrySet()) {
      if (isTenuredSpace(entry.getKey())) {
        MemoryUsage space = entry.getValue();
        if (space.getMax() == 0) {
          // The collector sometimes passes us nonsense stats.
          continue;
        }

        long percentUsed = 100 * space.getUsed() / space.getMax();
        if (percentUsed > occupiedHeapPercentageThreshold.getAsInt()) {
          if (info.getGcCause().equals("System.gc()") && !throwingOom.getAndSet(true)) {
            // Assume we got here from a GC initiated by the other branch.
            String exitMsg = BugReport.constructOomExitMessage(oomMessage);
            eventHandler.handle(Event.error(exitMsg));
            String detailedExitMsg =
                String.format(
                    "%s RetainedHeapLimiter forcing exit due to GC thrashing: After back-to-back"
                        + " full GCs, the tenured space is more than %s%% occupied (%s out of"
                        + " a tenured space size of %s).",
                    exitMsg,
                    occupiedHeapPercentageThreshold.getAsInt(),
                    space.getUsed(),
                    space.getMax());
            logger.atSevere().log(detailedExitMsg);
            // Exits the runtime.
            throw bugReporter.handleCrash(new OutOfMemoryError(detailedExitMsg));
          }

          if (System.currentTimeMillis() - lastTriggeredGcInMilliseconds.get()
              > MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS) {
            logger.atInfo().log(
                "Triggering a full GC with %s tenured space used out of a tenured space size of %s",
                space.getUsed(), space.getMax());
            // Force a full stop-the-world GC and see if it can get us below the threshold.
            System.gc();
            lastTriggeredGcInMilliseconds.set(System.currentTimeMillis());
          }
        }
      }
    }
  }

  private static boolean isTenuredSpace(String name) {
    return "CMS Old Gen".equals(name)
        || "G1 Old Gen".equals(name)
        || "PS Old Gen".equals(name)
        || "Tenured Gen".equals(name);
  }
}
