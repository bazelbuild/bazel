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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.common.options.OptionsParsingException;
import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

/**
 * Monitor the size of the retained heap and exit promptly if it grows too large.  Specifically,
 * check the size of the tenured space after each major GC; if it exceeds 90%, call
 * {@code System.gc()} to trigger a stop-the-world collection; if it's still more than 90% full,
 * exit with an {@link OutOfMemoryError}.
 */
class RetainedHeapLimiter implements NotificationListener {
  private static final Logger logger = Logger.getLogger(RetainedHeapLimiter.class.getName());
  private static final long MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS = 60000;

  private static int registeredOccupiedHeapPercentageThreshold = -1;

  static void maybeInstallRetainedHeapLimiter(int occupiedHeapPercentageThreshold)
      throws OptionsParsingException {
    if (registeredOccupiedHeapPercentageThreshold == -1) {
      registeredOccupiedHeapPercentageThreshold = occupiedHeapPercentageThreshold;
      new RetainedHeapLimiter(occupiedHeapPercentageThreshold).install();
    }
    if (registeredOccupiedHeapPercentageThreshold != occupiedHeapPercentageThreshold) {
      throw new OptionsParsingException(
          "Old threshold of "
              + registeredOccupiedHeapPercentageThreshold
              + " not equal to new threshold of "
              + occupiedHeapPercentageThreshold
              + ". To change the threshold, shut down the server and restart it with the desired "
              + "value");
    }
  }

  private boolean installed = false;
  private final AtomicBoolean throwingOom = new AtomicBoolean(false);
  private long lastTriggeredGcInMilliseconds = 0;
  private final int occupiedHeapPercentageThreshold;

  RetainedHeapLimiter(int occupiedHeapPercentageThreshold) {
    this.occupiedHeapPercentageThreshold = occupiedHeapPercentageThreshold;
  }

  void install() {
    Preconditions.checkState(!installed, "RetainedHeapLimiter installed twice");
    installed = true;
    List<GarbageCollectorMXBean> gcbeans = ManagementFactory.getGarbageCollectorMXBeans();
    boolean foundTenured = false;
    // Examine all collectors and register for notifications from those which collect the tenured
    // space. Normally there is one such collector.
    for (GarbageCollectorMXBean gcbean : gcbeans) {
      boolean collectsTenured = false;
      for (String name : gcbean.getMemoryPoolNames()) {
        collectsTenured |= isTenuredSpace(name);
      }
      if (collectsTenured) {
        foundTenured = true;
        NotificationEmitter emitter = (NotificationEmitter) gcbean;
        emitter.addNotificationListener(this, null, null);
      }
    }
    if (!foundTenured) {
      throw new IllegalStateException(
          "Can't find tenured space; update this class for a new collector");
    }
  }

  @Override
  public void handleNotification(Notification notification, Object handback) {
    if (!notification
        .getType()
        .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
      return;
    }
    GarbageCollectionNotificationInfo info =
        GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());
    Map<String, MemoryUsage> spaces = info.getGcInfo().getMemoryUsageAfterGc();
    for (Map.Entry<String, MemoryUsage> entry : spaces.entrySet()) {
      if (isTenuredSpace(entry.getKey())) {
        MemoryUsage space = entry.getValue();
        if (space.getMax() == 0) {
          // The CMS collector sometimes passes us nonsense stats.
          continue;
        }

        long percentUsed = 100 * space.getUsed() / space.getMax();
        if (percentUsed > occupiedHeapPercentageThreshold) {
          if (info.getGcCause().equals("System.gc()") && !throwingOom.getAndSet(true)) {
            // Assume we got here from a GC initiated by the other branch.
            String exitMsg =
                String.format(
                    "RetainedHeapLimiter forcing exit due to GC thrashing: tenured space "
                        + "%s out of %s (>%s%%) occupied after back-to-back full GCs",
                    space.getUsed(),
                    space.getMax(),
                    occupiedHeapPercentageThreshold);
            System.err.println(exitMsg);
            logger.info(exitMsg);
            // Exits the runtime.
            BugReport.handleCrash(new OutOfMemoryError(exitMsg));
          } else if (System.currentTimeMillis() - lastTriggeredGcInMilliseconds
              > MIN_TIME_BETWEEN_TRIGGERED_GC_MILLISECONDS) {
            logger.info(
                "Triggering a full GC with "
                    + space.getUsed()
                    + " out of "
                    + space.getMax()
                    + " used");
            // Force a full stop-the-world GC and see if it can get us below the threshold.
            System.gc();
            lastTriggeredGcInMilliseconds = System.currentTimeMillis();
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
