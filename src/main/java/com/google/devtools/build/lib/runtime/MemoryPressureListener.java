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
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.metrics.GarbageCollectionMetricsUtils;
import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

@ThreadSafe
final class MemoryPressureListener implements NotificationListener {

  private final AtomicReference<EventBus> eventBus = new AtomicReference<>();
  private final RetainedHeapLimiter retainedHeapLimiter;
  private final AtomicReference<GcThrashingDetector> gcThrashingDetector = new AtomicReference<>();

  private MemoryPressureListener(RetainedHeapLimiter retainedHeapLimiter) {
    this.retainedHeapLimiter = retainedHeapLimiter;
  }

  static MemoryPressureListener create(RetainedHeapLimiter retainedHeapLimiter) {
    return createFromBeans(ManagementFactory.getGarbageCollectorMXBeans(), retainedHeapLimiter);
  }

  @VisibleForTesting
  static MemoryPressureListener createFromBeans(
      List<GarbageCollectorMXBean> gcBeans, RetainedHeapLimiter retainedHeapLimiter) {
    ImmutableList<NotificationEmitter> tenuredGcEmitters = findTenuredCollectorBeans(gcBeans);
    if (tenuredGcEmitters.isEmpty()) {
      var names =
          gcBeans.stream()
              .map(GarbageCollectorMXBean::getMemoryPoolNames)
              .map(Arrays::asList)
              .collect(toImmutableList());
      throw new IllegalStateException(
          String.format(
              "Unable to find tenured collector from %s: names were %s.", gcBeans, names));
    }

    MemoryPressureListener memoryPressureListener = new MemoryPressureListener(retainedHeapLimiter);
    tenuredGcEmitters.forEach(e -> e.addNotificationListener(memoryPressureListener, null, null));
    return memoryPressureListener;
  }

  @VisibleForTesting
  static ImmutableList<NotificationEmitter> findTenuredCollectorBeans(
      List<GarbageCollectorMXBean> gcBeans) {
    ImmutableList.Builder<NotificationEmitter> builder = ImmutableList.builder();
    // Examine all collectors and register for notifications from those which collect the tenured
    // space. Normally there is one such collector.
    for (GarbageCollectorMXBean gcBean : gcBeans) {
      for (String name : gcBean.getMemoryPoolNames()) {
        if (GarbageCollectionMetricsUtils.isTenuredSpace(name)) {
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
      if (!GarbageCollectionMetricsUtils.isTenuredSpace(memoryUsageEntry.getKey())) {
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

    MemoryPressureEvent event =
        MemoryPressureEvent.newBuilder()
            .setWasManualGc(gcInfo.getGcCause().equals("System.gc()"))
            .setWasGcLockerInitiatedGc(gcInfo.getGcCause().equals("GCLocker Initiated GC"))
            .setWasFullGc(GarbageCollectionMetricsUtils.isFullGc(gcInfo))
            .setTenuredSpaceUsedBytes(tenuredSpaceUsedBytes)
            .setTenuredSpaceMaxBytes(tenuredSpaceMaxBytes)
            .build();

    GcThrashingDetector gcThrashingDetector = this.gcThrashingDetector.get();
    if (gcThrashingDetector != null) {
      gcThrashingDetector.handle(event);
    }

    // A null EventBus implies memory pressure event between commands with no active EventBus.
    // In such cases, notify RetainedHeapLimiter but do not publish event.
    EventBus eventBus = this.eventBus.get();
    if (eventBus != null) {
      eventBus.post(event);
    }
    // Post to EventBus first so memory pressure subscribers have a chance to make things
    // eligible for GC before RetainedHeapLimiter would trigger a full GC.
    this.retainedHeapLimiter.handle(event);
  }

  void setEventBus(@Nullable EventBus eventBus) {
    this.eventBus.set(eventBus);
  }

  void setGcThrashingDetector(@Nullable GcThrashingDetector gcThrashingDetector) {
    this.gcThrashingDetector.set(gcThrashingDetector);
  }
}
