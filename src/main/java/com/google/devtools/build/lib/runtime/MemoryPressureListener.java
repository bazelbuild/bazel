// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.Arrays;
import java.util.Map;
import javax.annotation.Nullable;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

@ThreadSafe
class MemoryPressureListener implements NotificationListener {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ImmutableList<MemoryPressureHandler> handlers;

  private MemoryPressureListener(ImmutableList<MemoryPressureHandler> handlers) {
    this.handlers = handlers;
  }

  @Nullable
  static MemoryPressureListener create(ImmutableList<MemoryPressureHandler> handlers) {
    return createFromBeans(
        ImmutableList.copyOf(ManagementFactory.getGarbageCollectorMXBeans()),
        handlers);
  }

  @VisibleForTesting
  @Nullable
  static MemoryPressureListener createFromBeans(
      ImmutableList<GarbageCollectorMXBean> gcBeans,
      ImmutableList<MemoryPressureHandler> handlers)  {
    ImmutableList<NotificationEmitter> tenuredGcEmitters = findTenuredCollectorBeans(gcBeans);
    if (tenuredGcEmitters.isEmpty()) {
      logger.atSevere().log(
          "Unable to find tenured collector from %s: names were %s.",
          gcBeans,
          gcBeans.stream()
              .map(GarbageCollectorMXBean::getMemoryPoolNames)
              .map(Arrays::asList)
              .collect(toImmutableList()));
      return null;
    }
    MemoryPressureListener memoryPressureListener = new MemoryPressureListener(handlers);
    tenuredGcEmitters.forEach(e -> e.addNotificationListener(memoryPressureListener, null, null));
    return memoryPressureListener;
  }

  @VisibleForTesting
  static ImmutableList<NotificationEmitter> findTenuredCollectorBeans(
      Iterable<GarbageCollectorMXBean> gcBeans) {
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

  @Override
  public void handleNotification(Notification notification, Object handback) {
    if (!notification
        .getType()
        .equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION)) {
      return;
    }

    GarbageCollectionNotificationInfo gcInfo =
        GarbageCollectionNotificationInfo.from((CompositeData) notification.getUserData());

    long tenuredSpaceUsedBytes = 0L;
    long tenuredSpaceMaxBytes = 0L;
    for (Map.Entry<String, MemoryUsage> memoryUsageEntry :
        gcInfo.getGcInfo().getMemoryUsageAfterGc().entrySet()) {
      if (!isTenuredSpace(memoryUsageEntry.getKey())) {
        continue;
      }
      MemoryUsage space = memoryUsageEntry.getValue();
      if (space.getMax() == 0L) {
        // The collector sometimes passes us nonsense stats.
        continue;
      }
      tenuredSpaceUsedBytes = space.getUsed();
      tenuredSpaceMaxBytes = space.getMax();
      break;
    }
    if (tenuredSpaceMaxBytes == 0L) {
      return;
    }

    MemoryPressureHandler.Event event =
        MemoryPressureHandler.Event.newBuilder()
            .setWasManualGc(gcInfo.getGcCause().equals("System.gc()"))
            .setTenuredSpaceUsedBytes(tenuredSpaceUsedBytes)
            .setTenuredSpaceMaxBytes(tenuredSpaceMaxBytes)
            .build();
    for (MemoryPressureHandler handler : handlers) {
      handler.handle(event);
    }
  }

  private static boolean isTenuredSpace(String name) {
    return "CMS Old Gen".equals(name)
        || "G1 Old Gen".equals(name)
        || "PS Old Gen".equals(name)
        || "Tenured Gen".equals(name)
        || "Shenandoah".equals(name)
        || "ZHeap".equals(name);
  }
}